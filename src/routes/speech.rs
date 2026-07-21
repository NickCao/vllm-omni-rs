//! POST /v1/audio/speech -- Qwen3 TTS endpoint.
//!
//! Supports two response modes:
//! - Non-streaming: returns complete audio as raw bytes (audio/wav, audio/pcm, etc.)
//! - SSE streaming: returns base64-encoded PCM chunks as SSE events

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

use crate::engine::{OmniEngine, extract_audio, is_finished};
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
            if !state
                .engine
                .supported_speakers
                .iter()
                .any(|s| s.to_lowercase() == voice_lower)
            {
                let valid = state.engine.supported_speakers.join(", ");
                return error_response(
                    StatusCode::BAD_REQUEST,
                    &format!(
                        "Invalid voice '{voice}'. Supported: {valid}"
                    ),
                );
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

/// Non-streaming: collect all audio, encode, return as single response.
async fn create_speech_full(
    state: AppState,
    req: SpeechRequest,
    request_id: String,
) -> Response {
    let engine = Arc::clone(&state.engine);
    let response_format = req.response_format.clone();

    let generator = match Python::with_gil(|py| {
        start_generate(py, &engine, &req, &request_id)
    }) {
        Ok(g) => g,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("{e:#}"),
            )
        }
    };

    let mut last_audio: Option<(Vec<u8>, u32)> = None;
    let mut yield_count = 0u32;

    loop {
        match OmniEngine::anext(&generator).await {
            Ok(Some(output)) => {
                yield_count += 1;
                let (audio, finished, debug_info) =
                    Python::with_gil(|py| {
                        let obj = output.bind(py);
                        let out_type: String = obj
                            .getattr("final_output_type")
                            .and_then(|v| v.extract())
                            .unwrap_or_default();
                        let mm = obj
                            .getattr("multimodal_output")
                            .ok();
                        let has_mm = mm
                            .as_ref()
                            .map(|v| !v.is_none() && v.is_truthy().unwrap_or(false))
                            .unwrap_or(false);
                        let mm_repr: String = mm
                            .as_ref()
                            .map(|v| format!("{}", v.repr().map(|r| r.to_string()).unwrap_or_default()))
                            .unwrap_or_default();
                        let has_req = obj
                            .getattr("request_output")
                            .map(|v| !v.is_none())
                            .unwrap_or(false);
                        let fin = is_finished(py, &output);
                        let audio = extract_audio(py, &output);
                        let debug = format!(
                            "yield#{yield_count}: type={out_type} finished={fin} \
                             has_mm={has_mm} mm={mm_repr} \
                             has_req_output={has_req} audio_extracted={}",
                            audio.is_some()
                        );
                        (audio, fin, debug)
                    });
                tracing::info!("{debug_info}");
                if audio.is_some() {
                    last_audio = audio;
                }
                if finished {
                    break;
                }
            }
            Ok(None) => break,
            Err(e) => {
                error!("generate error: {e:#}");
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("{e:#}"),
                );
            }
        }
    }

    let Some((pcm_bytes, sample_rate)) = last_audio else {
        return error_response(StatusCode::INTERNAL_SERVER_ERROR, "No audio output");
    };

    let (encoded, content_type) =
        encode_audio(&pcm_bytes, sample_rate, &response_format);

    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, content_type.parse().unwrap());
    (headers, encoded).into_response()
}

/// SSE streaming: send base64 PCM chunks as events.
async fn create_speech_sse(
    state: AppState,
    req: SpeechRequest,
    request_id: String,
) -> Response {
    let engine = Arc::clone(&state.engine);

    let generator = match Python::with_gil(|py| {
        start_generate(py, &engine, &req, &request_id)
    }) {
        Ok(g) => g,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("{e:#}"),
            )
        }
    };

    let stream = async_stream::stream! {
        loop {
            match OmniEngine::anext(&generator).await {
                Ok(Some(output)) => {
                    let (audio, finished) = Python::with_gil(|py| {
                        let audio = extract_audio(py, &output);
                        let fin = is_finished(py, &output);
                        (audio, fin)
                    });

                    if let Some((pcm_bytes, _sr)) = audio {
                        if !pcm_bytes.is_empty() {
                            let b64 = BASE64.encode(&pcm_bytes);
                            let delta = SseAudioDelta {
                                event_type: "speech.audio.delta",
                                audio: b64,
                                response_format: "pcm",
                            };
                            yield Ok::<_, std::convert::Infallible>(
                                Event::default()
                                    .event("speech.audio.delta")
                                    .json_data(&delta)
                                    .unwrap()
                            );
                        }
                    }

                    if finished {
                        let done = SseAudioDone {
                            event_type: "speech.audio.done",
                        };
                        yield Ok(
                            Event::default()
                                .event("speech.audio.done")
                                .json_data(&done)
                                .unwrap()
                        );
                        break;
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    error!("generate error: {e:#}");
                    let err = serde_json::json!({
                        "type": "speech.audio.error",
                        "error": {"message": format!("{e:#}")}
                    });
                    yield Ok(Event::default()
                        .event("speech.audio.error")
                        .json_data(&err)
                        .unwrap());
                    break;
                }
            }
        }
    };

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Raw audio streaming: send audio bytes directly.
async fn create_speech_raw_stream(
    state: AppState,
    req: SpeechRequest,
    request_id: String,
) -> Response {
    let engine = Arc::clone(&state.engine);

    let generator = match Python::with_gil(|py| {
        start_generate(py, &engine, &req, &request_id)
    }) {
        Ok(g) => g,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("{e:#}"),
            )
        }
    };

    let stream = async_stream::stream! {
        loop {
            match OmniEngine::anext(&generator).await {
                Ok(Some(output)) => {
                    let (audio, finished) = Python::with_gil(|py| {
                        let audio = extract_audio(py, &output);
                        let fin = is_finished(py, &output);
                        (audio, fin)
                    });
                    if let Some((pcm_bytes, _sr)) = audio {
                        if !pcm_bytes.is_empty() {
                            yield Ok::<_, std::io::Error>(bytes::Bytes::from(pcm_bytes));
                        }
                    }
                    if finished {
                        break;
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    error!("raw stream error: {e:#}");
                    break;
                }
            }
        }
    };

    let body = axum::body::Body::from_stream(stream);
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, "audio/pcm".parse().unwrap());
    (headers, body).into_response()
}

/// Build the Qwen3-TTS prompt and call engine.generate().
///
/// The prompt format is:
///   prompt_token_ids: [1] * estimated_len  (placeholders)
///   additional_information: {text, task_type, language, speaker, ...}
fn start_generate(
    py: Python<'_>,
    engine: &OmniEngine,
    req: &SpeechRequest,
    request_id: &str,
) -> anyhow::Result<PyObject> {
    let additional_info = PyDict::new(py);
    additional_info.set_item(
        "text",
        pyo3::types::PyList::new(py, &[&req.input])?,
    )?;

    let task_type = req
        .extra
        .get("task_type")
        .and_then(|v| v.as_str())
        .unwrap_or("CustomVoice");
    additional_info.set_item(
        "task_type",
        pyo3::types::PyList::new(py, &[task_type])?,
    )?;

    let language = req
        .extra
        .get("language")
        .and_then(|v| v.as_str())
        .unwrap_or("Auto");
    additional_info.set_item(
        "language",
        pyo3::types::PyList::new(py, &[language])?,
    )?;

    let speaker = req.voice.as_deref().unwrap_or("Chelsie");
    additional_info.set_item(
        "speaker",
        pyo3::types::PyList::new(py, &[speaker])?,
    )?;

    if let Some(ref instructions) = req.instructions {
        additional_info.set_item(
            "instruct",
            pyo3::types::PyList::new(py, &[instructions.as_str()])?,
        )?;
    }

    // Pass any extra fields into additional_information.
    for (k, v) in &req.extra {
        if matches!(k.as_str(), "task_type" | "language") {
            continue;
        }
        let py_v = pythonize::pythonize(py, v)?;
        additional_info.set_item(k, py_v)?;
    }

    // Build prompt with placeholder token IDs.
    // The model replaces these with computed embeddings at forward time.
    let placeholder_len = engine.estimate_tts_prompt_len(
        &req.input,
        req.instructions.as_deref(),
    );
    let ones: Vec<i64> = vec![1; placeholder_len];
    let prompt_token_ids = pyo3::types::PyList::new(py, &ones)?;

    let prompt = PyDict::new(py);
    prompt.set_item("prompt_token_ids", prompt_token_ids)?;
    prompt.set_item("additional_information", additional_info)?;

    let kwargs = PyDict::new(py);
    kwargs.set_item("request_id", request_id)?;
    kwargs.set_item(
        "output_modalities",
        pyo3::types::PyList::new(py, &[&"audio"])?,
    )?;

    engine.generate(py, &prompt, &kwargs)
}

/// Encode float32 PCM bytes into the requested format.
///
/// pcm_bytes is raw float32 LE samples. We convert to 16-bit PCM for WAV/PCM.
fn encode_audio(pcm_f32_bytes: &[u8], sample_rate: u32, format: &str) -> (Vec<u8>, &'static str) {
    let samples_f32: &[f32] = bytemuck::cast_slice(pcm_f32_bytes);
    let samples_i16: Vec<i16> = samples_f32
        .iter()
        .map(|&s| {
            let clamped = s.clamp(-1.0, 1.0);
            (clamped * 32767.0) as i16
        })
        .collect();

    match format {
        "pcm" => {
            let bytes: Vec<u8> = samples_i16
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect();
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
            for &s in &samples_i16 {
                writer.write_sample(s).unwrap();
            }
            writer.finalize().unwrap();
            (cursor.into_inner(), "audio/wav")
        }
    }
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({
            "error": {"message": message, "type": "server_error"}
        })),
    )
        .into_response()
}
