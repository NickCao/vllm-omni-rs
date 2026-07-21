//! POST /v1/audio/speech -- Qwen3 TTS endpoint.

use std::ffi::CString;
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
        match OmniEngine::anext(&generator).await {
            Ok(Some(output)) => {
                let (audio, finished) = Python::with_gil(|py| {
                    (extract_audio(py, &output), is_finished(py, &output))
                });
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
            match OmniEngine::anext(&generator).await {
                Ok(Some(output)) => {
                    let (audio, finished) = Python::with_gil(|py| {
                        (extract_audio(py, &output), is_finished(py, &output))
                    });
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
            match OmniEngine::anext(&generator).await {
                Ok(Some(output)) => {
                    let (audio, finished) = Python::with_gil(|py| {
                        (extract_audio(py, &output), is_finished(py, &output))
                    });
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
/// Prompt building is in Rust. Only the prompt length estimation calls
/// into Python (needs the model's tokenizer which has no tokenizer.json).
fn start_generate(
    py: Python<'_>,
    engine: &OmniEngine,
    req: &SpeechRequest,
    request_id: &str,
) -> anyhow::Result<PyObject> {
    (|| -> PyResult<PyObject> {
        // Build additional_information
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

        // Estimate prompt length via Python (the only Python call in prompt building).
        // Uses the engine's already-loaded model config + Python tokenizer.
        let prompt_len = estimate_prompt_len_py(py, engine, &info, task_type)?;

        // Build prompt: {prompt_token_ids: [1]*N, additional_information: {...}}
        let ones: Vec<i64> = vec![1; prompt_len];
        let prompt = PyDict::new(py);
        prompt.set_item("prompt_token_ids", pyo3::types::PyList::new(py, &ones)?)?;
        prompt.set_item("additional_information", &info)?;

        // Coerce sampling params: FINAL_ONLY for non-streaming, DELTA for streaming
        let is_streaming = req.stream || req.stream_format.as_deref() == Some("sse");
        let spl = py.import("copy")?.call_method1(
            "deepcopy",
            (engine.engine_ref(py).getattr("default_sampling_params_list")?,),
        )?;
        py.import("vllm_omni.entrypoints.utils")?.call_method1(
            "coerce_param_message_types",
            (&spl, is_streaming),
        )?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("request_id", request_id)?;
        kwargs.set_item("output_modalities", pyo3::types::PyList::new(py, &["audio"])?)?;
        kwargs.set_item("sampling_params_list", &spl)?;

        engine.generate(py, &prompt, &kwargs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
    })()
    .map_err(|e: PyErr| anyhow::anyhow!("{e}"))
}

// unused function removed

/// Call Python to estimate prompt length and return the result.
fn estimate_prompt_len_py(
    py: Python<'_>,
    engine: &OmniEngine,
    info: &Bound<'_, PyDict>,
    task_type: &str,
) -> PyResult<usize> {
    let locals = PyDict::new(py);
    locals.set_item("model_name", &engine.model_name)?;
    locals.set_item("info", info)?;
    locals.set_item("task_type", task_type)?;
    locals.set_item("engine", engine.engine_ref(py))?;

    let code = CString::new(concat!(
        "import sys, types\n",
        "from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder\n",
        "if '_omni_rs_cache' not in sys.modules:\n",
        "    from transformers import AutoTokenizer\n",
        "    _m = types.ModuleType('_omni_rs_cache')\n",
        "    _m.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')\n",
        "    sys.modules['_omni_rs_cache'] = _m\n",
        "_tok = sys.modules['_omni_rs_cache'].tok\n",
        "talker_config = engine.model_config.hf_config.talker_config\n",
        "result = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(\n",
        "    additional_information=dict(info),\n",
        "    task_type=task_type,\n",
        "    tokenize_prompt=lambda t: _tok(t, padding=False)['input_ids'],\n",
        "    codec_language_id=getattr(talker_config, 'codec_language_id', None),\n",
        "    spk_is_dialect=getattr(talker_config, 'spk_is_dialect', None),\n",
        ")\n",
    )).unwrap();
    py.run(&code, None, Some(&locals))?;

    locals.get_item("result")?.unwrap().extract()
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
