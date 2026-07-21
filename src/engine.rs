//! PyO3 bridge to vllm-omni's AsyncOmni.
//!
//! Calls AsyncOmni.generate() -- a Python async generator -- directly from
//! Rust via pyo3-async-runtimes. The GIL is acquired only when interacting
//! with Python objects; the actual orchestration runs in Python background
//! threads with the GIL released.


use anyhow::{Context, Result};
use pyo3::exceptions::PyStopAsyncIteration;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tracing::info;

/// Handle to the Python AsyncOmni engine client.
///
/// This wraps the high-level `AsyncOmni` (not the low-level
/// `AsyncOmniEngine`), so we get the full generate() async generator
/// with prompt building, sampling params, and output processing.
pub struct OmniEngine {
    /// The Python `AsyncOmni` instance.
    engine: PyObject,
    pub model_name: String,
}

// Safety: OmniEngine holds a PyObject which is Send.
// All access goes through Python::with_gil().
unsafe impl Send for OmniEngine {}
unsafe impl Sync for OmniEngine {}

impl OmniEngine {
    /// Create an AsyncOmni instance in the embedded Python interpreter.
    pub fn new(model: &str, kwargs: &serde_json::Map<String, serde_json::Value>) -> Result<Self> {
        Python::with_gil(|py| {
            let module = py
                .import("vllm_omni.entrypoints.async_omni")
                .context("Failed to import vllm_omni.entrypoints.async_omni")?;
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
            let engine = cls
                .call((), Some(&py_kwargs))
                .context("Failed to create AsyncOmni. Is vllm-omni installed?")?;

            info!("AsyncOmni ready");
            Ok(Self {
                engine: engine.into(),
                model_name: model.to_string(),
            })
        })
    }

    /// Call AsyncOmni.generate() and return the async generator as a PyObject.
    ///
    /// The caller iterates this with `anext()`.
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
    /// Returns Ok(Some(item)) for the next value, Ok(None) on
    /// StopAsyncIteration, or Err on failure.
    pub async fn anext(generator: &PyObject) -> Result<Option<PyObject>> {
        let coro = Python::with_gil(|py| -> PyResult<PyObject> {
            generator.call_method0(py, "__anext__")
        })?;

        let result = Python::with_gil(|py| {
            pyo3_async_runtimes::tokio::into_future(coro.into_bound(py))
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

/// Extract incremental text from an OmniRequestOutput (for future use).
pub fn extract_text(py: Python<'_>, output: &PyObject) -> Option<String> {
    let obj = output.bind(py);
    let request_output = obj.getattr("request_output").ok()?;
    if request_output.is_none() {
        return None;
    }
    let outputs = request_output.getattr("outputs").ok()?;
    let list = outputs.downcast::<pyo3::types::PyList>().ok()?;
    if list.is_empty() {
        return None;
    }
    let first = list.get_item(0).ok()?;
    first.getattr("text").ok()?.extract().ok()
}

/// Extract audio tensor bytes from an OmniRequestOutput.
///
/// For Qwen3 TTS, the audio lives at:
///   output.multimodal_output["model_outputs"] -> list of float32 tensors
///   output.multimodal_output["sr"] -> list of sample rates
///
/// Returns (pcm_f32_bytes, sample_rate) for the first audio output.
pub fn extract_audio(py: Python<'_>, output: &PyObject) -> Option<(Vec<u8>, u32)> {
    let obj = output.bind(py);

    let mm_output = obj.getattr("multimodal_output").ok()?;
    if mm_output.is_none() {
        return None;
    }

    let model_outputs = mm_output.get_item("model_outputs").ok()?;
    let sr_list = mm_output.get_item("sr").ok()?;

    let outputs_list = model_outputs.downcast::<pyo3::types::PyList>().ok()?;
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

    // Convert torch tensor to numpy bytes:
    //   tensor.detach().cpu().float().numpy().tobytes()
    let np_array = tensor
        .call_method0("detach")
        .ok()?
        .call_method0("cpu")
        .ok()?
        .call_method0("float")
        .ok()?
        .call_method0("numpy")
        .ok()?;
    let raw_bytes: Vec<u8> = np_array.call_method0("tobytes").ok()?.extract().ok()?;

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
