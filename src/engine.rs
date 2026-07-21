//! PyO3 bridge to vllm-omni's AsyncOmni.
//!
//! Creates a Python asyncio event loop in a background thread and stores
//! it as TaskLocals. All async Python calls go through this event loop
//! via `into_future_with_locals`.

use std::sync::OnceLock;
use std::thread;

use anyhow::{Context, Result};
use pyo3::exceptions::PyStopAsyncIteration;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_async_runtimes::TaskLocals;
use tokenizers::Tokenizer;
use tracing::info;

static TASK_LOCALS: OnceLock<TaskLocals> = OnceLock::new();

/// Start a Python asyncio event loop in a background thread and store
/// the TaskLocals for use by `anext()`.
pub fn init_python_event_loop() -> Result<()> {
    let locals = Python::with_gil(|py| -> PyResult<TaskLocals> {
        let asyncio = py.import("asyncio")?;
        let event_loop = asyncio.call_method0("new_event_loop")?;

        let locals =
            TaskLocals::new(event_loop.clone()).copy_context(py)?;

        let loop_ref = event_loop.unbind();
        thread::spawn(move || {
            Python::with_gil(|py| {
                let event_loop = loop_ref.bind(py);
                if let Err(e) = event_loop.call_method0("run_forever") {
                    e.print(py);
                }
            });
        });

        Ok(locals)
    })
    .context("Failed to create Python event loop")?;

    TASK_LOCALS
        .set(locals)
        .map_err(|_| anyhow::anyhow!("Python event loop already initialized"))?;

    info!("Python asyncio event loop started in background thread");
    Ok(())
}

fn get_task_locals() -> &'static TaskLocals {
    TASK_LOCALS
        .get()
        .expect("Python event loop not initialized -- call init_python_event_loop() first")
}

/// Handle to the Python AsyncOmni engine client.
pub struct OmniEngine {
    engine: PyObject,
    pub model_name: String,
    tokenizer: Option<Tokenizer>,
}

unsafe impl Send for OmniEngine {}
unsafe impl Sync for OmniEngine {}

impl OmniEngine {
    pub fn new(
        model: &str,
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Self> {
        Python::with_gil(|py| {
            let module = py
                .import("vllm_omni.entrypoints.async_omni")
                .context(
                    "Failed to import vllm_omni.entrypoints.async_omni",
                )?;
            let cls = module
                .getattr("AsyncOmni")
                .context("Failed to get AsyncOmni class")?;

            let py_kwargs = PyDict::new(py);
            py_kwargs.set_item("model", model)?;
            for (k, v) in kwargs {
                let py_v = pythonize::pythonize(py, v)?;
                py_kwargs.set_item(k, py_v)?;
            }

            info!("Creating AsyncOmni for model: {model}");
            let engine = cls.call((), Some(&py_kwargs)).context(
                "Failed to create AsyncOmni. Is vllm-omni installed?",
            )?;

            info!("AsyncOmni ready");

            let tokenizer = Tokenizer::from_pretrained(model, None)
                .map_err(|e| {
                    info!("Could not load tokenizer from HF: {e}. Prompt length estimation will use fallback.");
                })
                .ok();

            Ok(Self {
                engine: engine.into(),
                model_name: model.to_string(),
                tokenizer,
            })
        })
    }

    pub fn generate(
        &self,
        py: Python<'_>,
        prompt: &Bound<'_, PyDict>,
        kwargs: &Bound<'_, PyDict>,
    ) -> Result<PyObject> {
        let generator = self
            .engine
            .call_method(py, "generate", (prompt,), Some(kwargs))
            .context("AsyncOmni.generate() failed")?;
        Ok(generator)
    }

    /// Advance a Python async generator by one step.
    ///
    /// Uses the background Python event loop via TaskLocals.
    pub async fn anext(generator: &PyObject) -> Result<Option<PyObject>> {
        let locals = get_task_locals();

        let coro = Python::with_gil(|py| -> PyResult<PyObject> {
            generator.call_method0(py, "__anext__")
        })?;

        let result = Python::with_gil(|py| {
            pyo3_async_runtimes::into_future_with_locals(
                &locals.clone_ref(py),
                coro.into_bound(py),
            )
        })?
        .await;

        match result {
            Ok(val) => Ok(Some(val)),
            Err(e) => Python::with_gil(|py| {
                if e.is_instance_of::<PyStopAsyncIteration>(py) {
                    Ok(None)
                } else {
                    Err(e.into())
                }
            }),
        }
    }

    /// Estimate the placeholder prompt length for Qwen3-TTS CustomVoice.
    ///
    /// Mirrors `Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information`.
    pub fn estimate_tts_prompt_len(
        &self,
        text: &str,
        instruct: Option<&str>,
    ) -> usize {
        let Some(ref tokenizer) = self.tokenizer else {
            return 2048;
        };

        let tokenize = |s: &str| -> usize {
            tokenizer
                .encode(s, false)
                .map(|enc| enc.get_ids().len())
                .unwrap_or(0)
        };

        let instruct_len = match instruct {
            Some(i) if !i.trim().is_empty() => {
                tokenize(&format!("<|im_start|>user\n{i}<|im_end|>\n"))
            }
            _ => 0,
        };

        let assistant_text = format!(
            "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        );
        let assistant_len = tokenize(&assistant_text);

        let role_len = 3;
        let codec_prefix_len = 5; // (prefill=3 + speaker=1 + 2) - 1

        // CustomVoice non_streaming_mode=True: full text ids + eos + codec_bos
        instruct_len + role_len + codec_prefix_len + assistant_len.saturating_sub(6)
    }

    pub fn shutdown(&self) -> Result<()> {
        Python::with_gil(|py| {
            self.engine
                .call_method0(py, "shutdown")
                .context("shutdown failed")?;
            Ok(())
        })
    }
}

/// Extract audio tensor bytes from an OmniRequestOutput.
///
/// For Qwen3 TTS, the audio lives at:
///   output.multimodal_output["model_outputs"] -> list of float32 tensors
///   output.multimodal_output["sr"] -> list of sample rates
pub fn extract_audio(
    py: Python<'_>,
    output: &PyObject,
) -> Option<(Vec<u8>, u32)> {
    let obj = output.bind(py);

    let mm_output = obj.getattr("multimodal_output").ok()?;
    if mm_output.is_none() {
        return None;
    }

    let model_outputs = mm_output.get_item("model_outputs").ok()?;
    let sr_list = mm_output.get_item("sr").ok()?;

    let outputs_list =
        model_outputs.downcast::<pyo3::types::PyList>().ok()?;
    if outputs_list.is_empty() {
        return None;
    }

    let tensor = outputs_list.get_item(0).ok()?;
    let sr: u32 = sr_list
        .downcast::<pyo3::types::PyList>()
        .ok()?
        .get_item(0)
        .ok()?
        .extract()
        .ok()?;

    let np_array = tensor
        .call_method0("detach")
        .ok()?
        .call_method0("cpu")
        .ok()?
        .call_method0("float")
        .ok()?
        .call_method0("numpy")
        .ok()?;
    let raw_bytes: Vec<u8> =
        np_array.call_method0("tobytes").ok()?.extract().ok()?;

    Some((raw_bytes, sr))
}

/// Check if an OmniRequestOutput is finished.
pub fn is_finished(py: Python<'_>, output: &PyObject) -> bool {
    output
        .bind(py)
        .getattr("finished")
        .and_then(|v| v.extract())
        .unwrap_or(false)
}
