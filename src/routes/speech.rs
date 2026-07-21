//! POST /v1/audio/speech -- Pure Rust TTS endpoint.
//!
//! Uses PipelineRouter for ZMQ-native routing.
//! Zero Python per request.

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::{debug, error};
use uuid::Uuid;
use vllm_engine_core_client::protocol::OpaqueValue;

use crate::audio::{extract_and_concat_audio, text_entry};
use crate::server::AppState;

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    pub input: String,
    /// Accepted for OpenAI API compatibility; this server hosts a single model.
    #[serde(default)]
    #[allow(dead_code)]
    pub model: Option<String>,
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default = "default_response_format")]
    pub response_format: String,
    #[serde(default)]
    pub stream: bool,
    /// Only meaningful when `stream` is set, which is rejected below.
    #[serde(default)]
    #[allow(dead_code)]
    pub stream_format: Option<String>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

fn default_response_format() -> String {
    "wav".to_string()
}

pub async fn create_speech(
    State(state): State<AppState>,
    Json(req): Json<SpeechRequest>,
) -> Response {
    if !state.router.supports_speech() {
        return error_response(
            StatusCode::NOT_IMPLEMENTED,
            "this model has no audio-output stage",
        );
    }
    if req.stream {
        return error_response(
            StatusCode::BAD_REQUEST,
            "stream=true is not supported; this endpoint always returns a complete audio file",
        );
    }

    let request_id = format!("speech-{}", Uuid::new_v4());
    let response_format = req.response_format.clone();

    let task_type = req
        .extra
        .get("task_type")
        .and_then(|v| v.as_str())
        .unwrap_or("CustomVoice");
    let language = req
        .extra
        .get("language")
        .and_then(|v| v.as_str())
        .unwrap_or("Auto");
    let speaker = req.voice.as_deref().unwrap_or("Vivian");

    let mut entries = serde_json::Map::new();
    entries.insert("text".to_string(), text_entry(req.input.clone()));
    entries.insert("task_type".to_string(), text_entry(task_type));
    entries.insert("language".to_string(), text_entry(language));
    entries.insert("speaker".to_string(), text_entry(speaker));
    if let Some(ref inst) = req.instructions {
        entries.insert("instruct".to_string(), text_entry(inst.clone()));
    }

    let additional_info: OpaqueValue = match rmpv::ext::to_value(&json!({ "entries": entries })) {
        Ok(v) => v,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Failed to encode additional_information: {e:#}"),
            );
        }
    };

    let prompt_len = state
        .router
        .estimate_prompt_len(&req.input, req.instructions.as_deref());
    debug!(
        "Prompt length estimate: {prompt_len} for input: {:?}",
        &req.input
    );
    let prompt_token_ids: Vec<u32> = vec![1; prompt_len];

    match state
        .router
        .generate_speech(&request_id, prompt_token_ids, additional_info)
        .await
    {
        Ok(chunks) if !chunks.is_empty() => {
            debug!("Received {} audio chunks", chunks.len());
            match extract_and_concat_audio(&chunks, &response_format) {
                Ok((audio_bytes, content_type)) => {
                    let mut headers = HeaderMap::new();
                    headers.insert(header::CONTENT_TYPE, content_type.parse().unwrap());
                    (headers, audio_bytes).into_response()
                }
                Err(e) => error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("Audio extraction: {e:#}"),
                ),
            }
        }
        Ok(_) => error_response(StatusCode::INTERNAL_SERVER_ERROR, "No audio output"),
        Err(e) => {
            error!("[{request_id}] Generate failed: {e:#}");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}"))
        }
    }
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({"error": {"message": message, "type": "server_error"}})),
    )
        .into_response()
}
