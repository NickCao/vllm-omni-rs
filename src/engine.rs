//! PyO3 bridge to vllm-omni's AsyncOmni + Rust tokenizer for prompt estimation.

use std::sync::OnceLock;
use std::thread;

use anyhow::{Context, Result};
use pyo3::exceptions::PyStopAsyncIteration;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_async_runtimes::TaskLocals;
use tracing::info;
use vllm_tokenizer::HuggingFaceTokenizer;

static TASK_LOCALS: OnceLock<TaskLocals> = OnceLock::new();

pub fn init_python_event_loop() -> Result<()> {
    let locals = Python::with_gil(|py| -> PyResult<TaskLocals> {
        let asyncio = py.import("asyncio")?;
        let event_loop = asyncio.call_method0("new_event_loop")?;
        let locals = TaskLocals::new(event_loop.clone()).copy_context(py)?;
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
        .expect("init_python_event_loop() not called")
}

pub struct OmniEngine {
    engine: PyObject,
    pub model_name: String,
    pub supported_speakers: Vec<String>,
    tokenizer: Option<HuggingFaceTokenizer>,
}

unsafe impl Send for OmniEngine {}
unsafe impl Sync for OmniEngine {}

impl OmniEngine {
    pub fn new(
        model: &str,
        kwargs: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Self> {
        let (engine, supported_speakers, tokenizer_json_path) =
            Python::with_gil(|py| -> Result<(PyObject, Vec<String>, Option<String>)> {
                let module = py
                    .import("vllm_omni.entrypoints.async_omni")
                    .context("Failed to import vllm_omni.entrypoints.async_omni")?;
                let cls = module.getattr("AsyncOmni")?;

                let py_kwargs = PyDict::new(py);
                py_kwargs.set_item("model", model)?;
                for (k, v) in kwargs {
                    let py_v = pythonize::pythonize(py, v)?;
                    py_kwargs.set_item(k, py_v)?;
                }

                info!("Creating AsyncOmni for model: {model}");
                let engine = cls
                    .call((), Some(&py_kwargs))
                    .context("Failed to create AsyncOmni. Is vllm-omni installed?")?;
                info!("AsyncOmni ready");

                let supported_speakers = (|| -> PyResult<Vec<String>> {
                    let hf_config =
                        engine.getattr("model_config")?.getattr("hf_config")?;
                    let talker_config = hf_config.getattr("talker_config")?;
                    let spk_id = talker_config.getattr("spk_id")?;
                    let dict = spk_id.downcast::<PyDict>()?;
                    Ok(dict
                        .keys()
                        .iter()
                        .filter_map(|k| k.extract::<String>().ok())
                        .collect())
                })()
                .unwrap_or_default();

                // Extract tokenizer.json from the Python tokenizer so we can
                // load it in Rust. The HF repo doesn't ship tokenizer.json but
                // the Python tokenizer builds one from vocab.json + merges.txt.
                let tok_path = (|| -> PyResult<String> {
                    let path = "/tmp/_vllm_omni_rs_tokenizer.json";
                    py.run(
                        &std::ffi::CString::new(format!(
                            "from transformers import AutoTokenizer\n\
                             _t = AutoTokenizer.from_pretrained('{}', trust_remote_code=True)\n\
                             _t.backend_tokenizer.save('{}')\n",
                            model, path
                        ))
                        .unwrap(),
                        None,
                        None,
                    )?;
                    Ok(path.to_string())
                })()
                .ok();

                if supported_speakers.is_empty() {
                    info!("No supported speakers found");
                } else {
                    info!(
                        "Supported speakers: {}",
                        supported_speakers.join(", ")
                    );
                }

                Ok((engine.into(), supported_speakers, tok_path))
            })?;

        let tokenizer = tokenizer_json_path
            .and_then(|path| {
                HuggingFaceTokenizer::new(std::path::Path::new(&path))
                    .map_err(|e| {
                        info!("Failed to load Rust tokenizer: {e}. Will fall back to Python.");
                    })
                    .ok()
            });

        if tokenizer.is_some() {
            info!("Rust tokenizer loaded successfully");
        }

        Ok(Self {
            engine,
            model_name: model.to_string(),
            supported_speakers,
            tokenizer,
        })
    }

    pub fn engine_ref<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        self.engine.bind(py)
    }

    pub fn generate(
        &self,
        py: Python<'_>,
        prompt: &Bound<'_, PyAny>,
        kwargs: &Bound<'_, PyDict>,
    ) -> Result<PyObject> {
        self.engine
            .call_method(py, "generate", (prompt,), Some(kwargs))
            .context("AsyncOmni.generate() failed")
    }

    pub async fn anext(generator: &PyObject) -> Result<Option<PyObject>> {
        let locals = get_task_locals();

        // Single GIL acquisition: get __anext__ coroutine and submit to event loop
        let fut = Python::with_gil(|py| -> anyhow::Result<_> {
            let coro = generator.call_method0(py, "__anext__")?;
            let fut = pyo3_async_runtimes::into_future_with_locals(
                &locals.clone_ref(py),
                coro.into_bound(py),
            )?;
            Ok(fut)
        })?;

        match fut.await {
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

    /// Estimate prompt placeholder length for Qwen3-TTS CustomVoice.
    ///
    /// Pure Rust when the tokenizer loaded; falls back to Python otherwise.
    pub fn estimate_tts_prompt_len(
        &self,
        text: &str,
        instruct: Option<&str>,
        task_type: &str,
    ) -> usize {
        if let Some(ref tok) = self.tokenizer {
            self.estimate_prompt_len_rust(tok, text, instruct, task_type)
        } else {
            self.estimate_prompt_len_python(text, instruct, task_type)
        }
    }

    fn estimate_prompt_len_rust(
        &self,
        tok: &HuggingFaceTokenizer,
        text: &str,
        instruct: Option<&str>,
        task_type: &str,
    ) -> usize {
        use vllm_tokenizer::Tokenizer;

        let tokenize = |s: &str| -> usize {
            tok.encode(s, false)
                .map(|ids| ids.len())
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
        // prefill_len=3 (no language_id), speaker_len=1 for CustomVoice/Base
        let speaker_len: usize =
            if task_type == "CustomVoice" || task_type == "Base" {
                1
            } else {
                0
            };
        let codec_input_len = 3 + speaker_len + 2;
        let codec_prefix_len = codec_input_len - 1;

        let mut prompt_len = instruct_len + role_len + codec_prefix_len;

        if task_type == "CustomVoice" || task_type == "VoiceDesign" {
            // non_streaming_mode=True (default for CustomVoice/VoiceDesign)
            prompt_len += assistant_len.saturating_sub(6);
        }

        prompt_len
    }

    fn estimate_prompt_len_python(
        &self,
        text: &str,
        instruct: Option<&str>,
        task_type: &str,
    ) -> usize {
        Python::with_gil(|py| -> PyResult<usize> {
            let locals = PyDict::new(py);
            locals.set_item("model_name", &self.model_name)?;
            locals.set_item("engine", self.engine.bind(py))?;
            locals.set_item("text", text)?;
            locals.set_item("instruct", instruct.unwrap_or(""))?;
            locals.set_item("task_type", task_type)?;

            let code = std::ffi::CString::new(concat!(
                "import sys, types\n",
                "from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder\n",
                "if '_omni_rs_cache' not in sys.modules:\n",
                "    from transformers import AutoTokenizer\n",
                "    _m = types.ModuleType('_omni_rs_cache')\n",
                "    _m.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')\n",
                "    sys.modules['_omni_rs_cache'] = _m\n",
                "_tok = sys.modules['_omni_rs_cache'].tok\n",
                "info = {'text': [text], 'task_type': [task_type], 'language': ['Auto']}\n",
                "if instruct: info['instruct'] = [instruct]\n",
                "talker_config = engine.model_config.hf_config.talker_config\n",
                "result = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(\n",
                "    additional_information=info,\n",
                "    task_type=task_type,\n",
                "    tokenize_prompt=lambda t: _tok(t, padding=False)['input_ids'],\n",
                "    codec_language_id=getattr(talker_config, 'codec_language_id', None),\n",
                "    spk_is_dialect=getattr(talker_config, 'spk_is_dialect', None),\n",
                ")\n",
            )).unwrap();
            py.run(&code, Some(&locals), Some(&locals))?;
            locals.get_item("result")?.unwrap().extract()
        })
        .unwrap_or_else(|e| {
            info!("Python prompt estimation failed: {e}, using fallback");
            2048
        })
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

/// Advance the generator and extract audio + finished status in one GIL acquisition.
/// Returns None when the generator is exhausted.
pub async fn anext_audio(
    generator: &PyObject,
) -> Result<Option<(Option<(Vec<u8>, u32)>, bool)>> {
    let output = match OmniEngine::anext(generator).await? {
        Some(o) => o,
        None => return Ok(None),
    };

    // Single GIL: extract audio + check finished
    Ok(Some(Python::with_gil(|py| {
        let audio = extract_audio(py, &output);
        let finished = is_finished(py, &output);
        (audio, finished)
    })))
}

pub fn extract_audio(py: Python<'_>, output: &PyObject) -> Option<(Vec<u8>, u32)> {
    let obj = output.bind(py);
    let mm = obj.getattr("multimodal_output").ok()?;
    if mm.is_none() { return None; }

    let audio_key = ["audio", "model_outputs"]
        .into_iter()
        .find(|k| mm.contains(k).unwrap_or(false))?;
    let audio_val = mm.get_item(audio_key).ok()?;

    let tensor = if let Ok(list) = audio_val.downcast::<pyo3::types::PyList>() {
        (0..list.len()).rev().find_map(|i| {
            let item = list.get_item(i).ok()?;
            let n: i64 = item.call_method0("numel").and_then(|v| v.extract()).unwrap_or(0);
            if n > 0 { Some(item) } else { None }
        })?
    } else {
        let n: i64 = audio_val.call_method0("numel").and_then(|v| v.extract()).unwrap_or(0);
        if n == 0 { return None; }
        audio_val
    };

    let sr: u32 = (|| -> Option<u32> {
        let sr_val = mm.get_item("sr").ok()?;
        sr_val.extract::<u32>().ok()
            .or_else(|| sr_val.call_method0("item").ok()?.extract::<u32>().ok())
    })()
    .unwrap_or(24000);

    let np_array = tensor
        .call_method0("float").ok()?
        .call_method0("detach").ok()?
        .call_method0("cpu").ok()?
        .call_method0("numpy").ok()?;
    let ndim: i64 = np_array.getattr("ndim").and_then(|v| v.extract()).unwrap_or(1);
    let final_array = if ndim > 1 { np_array.call_method0("squeeze").ok()? } else { np_array };
    let raw_bytes: Vec<u8> = final_array.call_method0("tobytes").ok()?.extract().ok()?;
    if raw_bytes.is_empty() { return None; }

    Some((raw_bytes, sr))
}

pub fn is_finished(py: Python<'_>, output: &PyObject) -> bool {
    output.bind(py).getattr("finished").and_then(|v| v.extract()).unwrap_or(false)
}
