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
use tracing::info;

static TASK_LOCALS: OnceLock<TaskLocals> = OnceLock::new();

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
        .expect("init_python_event_loop() not called")
}

pub struct OmniEngine {
    engine: PyObject,
    pub model_name: String,
    pub supported_speakers: Vec<String>,
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
                let hf_config = engine.getattr("model_config")?.getattr("hf_config")?;
                let talker_config = hf_config.getattr("talker_config")?;
                let spk_id = talker_config.getattr("spk_id")?;
                let dict = spk_id.downcast::<PyDict>()?;
                Ok(dict.keys().iter().filter_map(|k| k.extract::<String>().ok()).collect())
            })()
            .unwrap_or_default();

            if !supported_speakers.is_empty() {
                info!("Supported speakers: {}", supported_speakers.join(", "));
            }

            Ok(Self {
                engine: engine.into(),
                model_name: model.to_string(),
                supported_speakers,
            })
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

    pub fn shutdown(&self) -> Result<()> {
        Python::with_gil(|py| {
            self.engine
                .call_method0(py, "shutdown")
                .context("shutdown failed")?;
            Ok(())
        })
    }
}

pub fn extract_audio(py: Python<'_>, output: &PyObject) -> Option<(Vec<u8>, u32)> {
    let obj = output.bind(py);
    let mm = obj.getattr("multimodal_output").ok()?;
    if mm.is_none() {
        return None;
    }

    let audio_key = ["audio", "model_outputs"]
        .into_iter()
        .find(|k| mm.contains(k).unwrap_or(false))?;
    let audio_val = mm.get_item(audio_key).ok()?;

    let tensor = if let Ok(list) = audio_val.downcast::<pyo3::types::PyList>() {
        let mut best = None;
        for i in (0..list.len()).rev() {
            if let Ok(item) = list.get_item(i) {
                let n: i64 = item.call_method0("numel").and_then(|v| v.extract()).unwrap_or(0);
                if n > 0 {
                    best = Some(item);
                    break;
                }
            }
        }
        best?
    } else {
        let n: i64 = audio_val.call_method0("numel").and_then(|v| v.extract()).unwrap_or(0);
        if n == 0 { return None; }
        audio_val
    };

    let sr: u32 = (|| -> Option<u32> {
        let sr_val = mm.get_item("sr").ok()?;
        if let Ok(v) = sr_val.extract::<u32>() { return Some(v); }
        sr_val.call_method0("item").ok()?.extract::<u32>().ok()
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
