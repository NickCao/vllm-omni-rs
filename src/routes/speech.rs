//! POST /v1/audio/speech -- Pure Rust TTS endpoint.
//!
//! Uses TtsRouter for ZMQ-native 2-stage routing.
//! Zero Python per request.


use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::Value;
use tracing::error;
use uuid::Uuid;
use vllm_engine_core_client::protocol::OpaqueValue;

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
    pub stream: bool,
    #[serde(default)]
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
    let request_id = format!("speech-{}", Uuid::new_v4());
    let _response_format = req.response_format.clone();

    // Build additional_information for stage 0
    let task_type = req.extra.get("task_type").and_then(|v| v.as_str()).unwrap_or("CustomVoice");
    let language = req.extra.get("language").and_then(|v| v.as_str()).unwrap_or("Auto");
    let speaker = req.voice.as_deref().unwrap_or("Vivian");

    // Build AdditionalInformationPayload format:
    // { "entries": { "key": { "list_data": [values] }, ... } }
    let mut entries = serde_json::Map::new();

    let make_list_entry = |vals: Vec<Value>| -> Value {
        let mut entry = serde_json::Map::new();
        entry.insert("list_data".into(), Value::Array(vals));
        // All other fields null
        entry.insert("tensor_data".into(), Value::Null);
        entry.insert("tensor_shape".into(), Value::Null);
        entry.insert("tensor_dtype".into(), Value::Null);
        entry.insert("scalar_data".into(), Value::Null);
        Value::Object(entry)
    };

    entries.insert("text".into(), make_list_entry(vec![Value::String(req.input.clone())]));
    entries.insert("task_type".into(), make_list_entry(vec![Value::String(task_type.to_string())]));
    entries.insert("language".into(), make_list_entry(vec![Value::String(language.to_string())]));
    entries.insert("speaker".into(), make_list_entry(vec![Value::String(speaker.to_string())]));

    if let Some(ref inst) = req.instructions {
        entries.insert("instruct".into(), make_list_entry(vec![Value::String(inst.clone())]));
    }

    let mut payload = serde_json::Map::new();
    payload.insert("entries".into(), Value::Object(entries));

    let additional_info = match rmp_serde::to_vec_named(&payload) {
        Ok(bytes) => match rmp_serde::from_slice::<rmpv::Value>(&bytes) {
            Ok(val) => OpaqueValue::from(val),
            Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("Failed to encode: {e}")),
        },
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("Failed to serialize: {e}")),
    };

    // Estimate prompt length (placeholder for now -- will use tokenizer)
    let prompt_len = 2048usize; // TODO: use vllm-tokenizer
    let prompt_token_ids: Vec<u32> = vec![1; prompt_len];

    // Generate speech via TtsRouter
    match state.router.generate_speech(
        &request_id,
        prompt_token_ids,
        additional_info,
        None, // use default sampling params
    ).await {
        Ok(Some(_audio_opaque)) => {
            // TODO: extract audio bytes from OpaqueValue
            // For now, return a placeholder
            error_response(StatusCode::INTERNAL_SERVER_ERROR, "Audio extraction from ZMQ output not yet implemented")
        }
        Ok(None) => {
            error_response(StatusCode::INTERNAL_SERVER_ERROR, "No audio output")
        }
        Err(e) => {
            error!("[{request_id}] Generate failed: {e:#}");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}"))
        }
    }
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (status, Json(serde_json::json!({"error": {"message": message, "type": "server_error"}}))).into_response()
}
