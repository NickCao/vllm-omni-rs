//! POST /v1/audio/speech -- Qwen3 TTS endpoint.

use std::io::Cursor;
use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::Engine as Base64Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::error;
use uuid::Uuid;

use crate::engine::{OmniEngine, anext_audio};
use crate::server::AppState;

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    pub input: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default = "default_response_format")]
    pub response_format: String,
    #[serde(default)]
    pub speed: Option<f64>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_format: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

fn default_response_format() -> String {
    "wav".to_string()
}

#[derive(Serialize)]
struct SseAudioDelta {
    #[serde(rename = "type")]
    event_type: &'static str,
    audio: String,
    response_format: &'static str,
}

#[derive(Serialize)]
struct SseAudioDone {
    #[serde(rename = "type")]
    event_type: &'static str,
}

pub async fn create_speech(
    State(state): State<AppState>,
    Json(req): Json<SpeechRequest>,
) -> Response {
    let request_id = format!("speech-{}", Uuid::new_v4());

    if let Some(ref voice) = req.voice {
        if !state.engine.supported_speakers.is_empty() {
            let voice_lower = voice.to_lowercase();
            if !state.engine.supported_speakers.iter().any(|s| s.to_lowercase() == voice_lower) {
                let valid = state.engine.supported_speakers.join(", ");
                return error_response(StatusCode::BAD_REQUEST, &format!("Invalid voice '{voice}'. Supported: {valid}"));
            }
        }
    }

    let is_sse = req.stream || req.stream_format.as_deref() == Some("sse");
    let is_raw_stream = req.stream_format.as_deref() == Some("audio");

    if is_sse {
        create_speech_sse(state, req, request_id).await
    } else if is_raw_stream {
        create_speech_raw_stream(state, req, request_id).await
    } else {
        create_speech_full(state, req, request_id).await
    }
}

async fn create_speech_full(state: AppState, req: SpeechRequest, request_id: String) -> Response {
    let engine = Arc::clone(&state.engine);
    let response_format = req.response_format.clone();

    let generator = match Python::with_gil(|py| start_generate(py, &engine, &req, &request_id)) {
        Ok(g) => g,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}")),
    };

    let mut last_audio: Option<(Vec<u8>, u32)> = None;
    loop {
        match anext_audio(&generator).await {
            Ok(Some((audio, finished))) => {
                if audio.is_some() { last_audio = audio; }
                if finished { break; }
            }
            Ok(None) => break,
            Err(e) => {
                error!("generate error: {e:#}");
                return error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}"));
            }
        }
    }

    let Some((pcm_bytes, sample_rate)) = last_audio else {
        return error_response(StatusCode::INTERNAL_SERVER_ERROR, "No audio output");
    };

    let (encoded, content_type) = encode_audio(&pcm_bytes, sample_rate, &response_format);
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, content_type.parse().unwrap());
    (headers, encoded).into_response()
}

async fn create_speech_sse(state: AppState, req: SpeechRequest, request_id: String) -> Response {
    let engine = Arc::clone(&state.engine);
    let generator = match Python::with_gil(|py| start_generate(py, &engine, &req, &request_id)) {
        Ok(g) => g,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}")),
    };

    let stream = async_stream::stream! {
        loop {
            match anext_audio(&generator).await {
                Ok(Some((audio, finished))) => {
                    if let Some((pcm_bytes, _)) = audio {
                        if !pcm_bytes.is_empty() {
                            let b64 = BASE64.encode(&pcm_bytes);
                            let delta = SseAudioDelta { event_type: "speech.audio.delta", audio: b64, response_format: "pcm" };
                            yield Ok::<_, std::convert::Infallible>(Event::default().event("speech.audio.delta").json_data(&delta).unwrap());
                        }
                    }
                    if finished {
                        let done = SseAudioDone { event_type: "speech.audio.done" };
                        yield Ok(Event::default().event("speech.audio.done").json_data(&done).unwrap());
                        break;
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    error!("generate error: {e:#}");
                    yield Ok(Event::default().event("speech.audio.error").json_data(&serde_json::json!({"type":"speech.audio.error","error":{"message":format!("{e:#}")}})).unwrap());
                    break;
                }
            }
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}

async fn create_speech_raw_stream(state: AppState, req: SpeechRequest, request_id: String) -> Response {
    let engine = Arc::clone(&state.engine);
    let generator = match Python::with_gil(|py| start_generate(py, &engine, &req, &request_id)) {
        Ok(g) => g,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}")),
    };

    let stream = async_stream::stream! {
        loop {
            match anext_audio(&generator).await {
                Ok(Some((audio, finished))) => {
                    if let Some((pcm_bytes, _)) = audio {
                        if !pcm_bytes.is_empty() {
                            yield Ok::<_, std::io::Error>(bytes::Bytes::from(pcm_bytes));
                        }
                    }
                    if finished { break; }
                }
                Ok(None) => break,
                Err(e) => { error!("raw stream error: {e:#}"); break; }
            }
        }
    };
    let body = axum::body::Body::from_stream(stream);
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, "audio/pcm".parse().unwrap());
    (headers, body).into_response()
}

/// Build TTS prompt and call engine.generate().
///
/// Prompt building and length estimation are done in Rust (using vllm-tokenizer).
/// Only generate() and sampling params coercion call into Python.
fn start_generate(
    py: Python<'_>,
    engine: &OmniEngine,
    req: &SpeechRequest,
    request_id: &str,
) -> anyhow::Result<PyObject> {
    (|| -> PyResult<PyObject> {
        let info = PyDict::new(py);
        info.set_item("text", pyo3::types::PyList::new(py, &[&req.input])?)?;

        let task_type = req.extra.get("task_type").and_then(|v| v.as_str()).unwrap_or("CustomVoice");
        info.set_item("task_type", pyo3::types::PyList::new(py, &[task_type])?)?;

        let language = req.extra.get("language").and_then(|v| v.as_str()).unwrap_or("Auto");
        info.set_item("language", pyo3::types::PyList::new(py, &[language])?)?;

        let speaker = req.voice.as_deref().unwrap_or("Vivian");
        info.set_item("speaker", pyo3::types::PyList::new(py, &[speaker])?)?;

        if let Some(ref inst) = req.instructions {
            info.set_item("instruct", pyo3::types::PyList::new(py, &[inst.as_str()])?)?;
        }
        for (k, v) in &req.extra {
            if !matches!(k.as_str(), "task_type" | "language") {
                info.set_item(k, pythonize::pythonize(py, v)?)?;
            }
        }

        // Prompt length estimation -- pure Rust when tokenizer loaded
        let prompt_len = engine.estimate_tts_prompt_len(
            &req.input,
            req.instructions.as_deref(),
            task_type,
        );

        let ones: Vec<i64> = vec![1; prompt_len];
        let prompt = PyDict::new(py);
        prompt.set_item("prompt_token_ids", pyo3::types::PyList::new(py, &ones)?)?;
        prompt.set_item("additional_information", &info)?;

        // Sampling params: clone + set output_kind directly from Rust.
        // RequestOutputKind: CUMULATIVE=0, DELTA=1, FINAL_ONLY=2
        let is_streaming = req.stream || req.stream_format.as_deref() == Some("sse");
        let target_kind: i32 = if is_streaming { 1 } else { 2 }; // DELTA or FINAL_ONLY
        let copy = py.import("copy")?;
        let default_spl = engine.engine_ref(py).getattr("default_sampling_params_list")?;
        let spl = copy.call_method1("deepcopy", (&default_spl,))?;
        let spl_list = spl.downcast::<pyo3::types::PyList>()?;
        let sampling_params_cls = py.import("vllm.sampling_params")?
            .getattr("SamplingParams")?;
        for i in 0..spl_list.len() {
            let sp = spl_list.get_item(i)?;
            if sp.is_instance(&sampling_params_cls)? {
                let clone = sp.call_method0("clone")?;
                clone.setattr("output_kind", target_kind)?;
                clone.setattr("skip_clone", true)?;
                spl_list.set_item(i, clone)?;
            }
        }

        let kwargs = PyDict::new(py);
        kwargs.set_item("request_id", request_id)?;
        kwargs.set_item("output_modalities", pyo3::types::PyList::new(py, &["audio"])?)?;
        kwargs.set_item("sampling_params_list", &spl)?;

        engine.generate(py, &prompt, &kwargs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
    })()
    .map_err(|e: PyErr| anyhow::anyhow!("{e}"))
}

fn encode_audio(pcm_f32_bytes: &[u8], sample_rate: u32, format: &str) -> (Vec<u8>, &'static str) {
    let samples_f32: &[f32] = bytemuck::cast_slice(pcm_f32_bytes);
    let samples_i16: Vec<i16> = samples_f32
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect();

    match format {
        "pcm" => {
            let bytes: Vec<u8> = samples_i16.iter().flat_map(|s| s.to_le_bytes()).collect();
            (bytes, "audio/pcm")
        }
        _ => {
            let mut cursor = Cursor::new(Vec::new());
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            for &s in &samples_i16 { writer.write_sample(s).unwrap(); }
            writer.finalize().unwrap();
            (cursor.into_inner(), "audio/wav")
        }
    }
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (status, Json(serde_json::json!({"error": {"message": message, "type": "server_error"}}))).into_response()
}
