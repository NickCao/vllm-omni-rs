//! POST /v1/chat/completions -- Qwen3-Omni chat endpoint.
//!
//! Scope: text-only input (image/audio/video content parts rejected),
//! non-streaming only. Uses the same PipelineRouter as /v1/audio/speech;
//! zero Python per request (the chat template is rendered in Rust via
//! `chat_template::ChatTemplateRenderer`, loaded once at startup).

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;

use anyhow::Result;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::error;
use uuid::Uuid;
use vllm_engine_core_client::protocol::OpaqueValue;
use vllm_engine_core_client::protocol::output::EngineCoreFinishReason;
use vllm_tokenizer::Tokenizer;

use crate::audio::{extract_and_concat_audio, text_entry};
use crate::chat_template;
use crate::introspect::SamplingOverrides;
use crate::server::AppState;

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
struct ContentPart {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
}

impl MessageContent {
    /// Flattens to plain text; `Err` carries the 400 message for any
    /// non-"text" part, since input is text-only for this endpoint.
    fn into_text(self) -> Result<String, String> {
        match self {
            Self::Text(s) => Ok(s),
            Self::Parts(parts) => {
                let mut buf = String::new();
                for part in parts {
                    if part.kind != "text" {
                        return Err(format!(
                            "unsupported content part type {:?}; only text input is supported",
                            part.kind
                        ));
                    }
                    buf.push_str(part.text.as_deref().unwrap_or(""));
                }
                Ok(buf)
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct ChatMessageIn {
    role: String,
    #[serde(default)]
    content: Option<MessageContent>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StopSequences {
    Single(String),
    Multiple(Vec<String>),
}

impl StopSequences {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::Multiple(v) => v,
        }
    }
}

#[derive(Debug, Deserialize)]
struct AudioParam {
    #[serde(default)]
    voice: Option<String>,
    #[serde(default = "default_audio_format")]
    format: String,
}

fn default_audio_format() -> String {
    "wav".to_string()
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatCompletionRequest {
    /// Accepted for OpenAI API compatibility; this server hosts a single model.
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,
    messages: Vec<ChatMessageIn>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i64>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    min_tokens: Option<u32>,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    stop: Option<StopSequences>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    modalities: Option<Vec<String>>,
    #[serde(default)]
    audio: Option<AudioParam>,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: i64,
    model: String,
    choices: Vec<ChatCompletionChoice>,
    usage: UsageInfo,
}

#[derive(Serialize)]
struct ChatCompletionChoice {
    index: u32,
    message: ChatCompletionMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct ChatCompletionMessage {
    role: &'static str,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio: Option<ChatCompletionAudio>,
}

#[derive(Serialize)]
struct ChatCompletionAudio {
    id: String,
    data: String,
    expires_at: i64,
    transcript: String,
}

#[derive(Serialize)]
struct UsageInfo {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

pub async fn create_chat_completion(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let Some(chat_template) = &state.chat_template else {
        return error_response(
            StatusCode::NOT_IMPLEMENTED,
            "this model has no text-output stage; /v1/chat/completions is unavailable",
        );
    };
    if !state.router.supports_chat() {
        return error_response(
            StatusCode::NOT_IMPLEMENTED,
            "this model has no text-output stage; /v1/chat/completions is unavailable",
        );
    }
    if req.stream {
        return error_response(StatusCode::BAD_REQUEST, "stream=true is not supported yet");
    }
    if let Some(modalities) = &req.modalities {
        if modalities.iter().any(|m| m != "text" && m != "audio") {
            return error_response(
                StatusCode::BAD_REQUEST,
                "only \"text\" and \"audio\" output modalities are supported",
            );
        }
    }

    let mut messages = Vec::with_capacity(req.messages.len());
    for m in req.messages {
        let text = match m.content {
            Some(c) => match c.into_text() {
                Ok(t) => t,
                Err(msg) => return error_response(StatusCode::BAD_REQUEST, &msg),
            },
            None => String::new(),
        };
        messages.push(chat_template::ChatMessage {
            role: m.role,
            content: text,
        });
    }

    let rendered = match chat_template.render(&messages, true) {
        Ok(s) => s,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("chat template render failed: {e:#}"),
            );
        }
    };
    let prompt_token_ids = match state.tokenizer.encode(&rendered, false) {
        Ok(ids) => ids,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("tokenize failed: {e:#}"),
            );
        }
    };
    let prompt_tokens = prompt_token_ids.len() as u32;

    // additional_information only carries a voice override, if requested;
    // a bare chat completion has nothing to say here, matching how
    // vllm-omni's own thinker stage treats it as fully optional.
    let voice = req.audio.as_ref().and_then(|a| a.voice.clone());
    let additional_info: OpaqueValue = match voice {
        Some(v) => {
            let mut entries = serde_json::Map::new();
            entries.insert("speaker".to_string(), text_entry(v));
            match rmpv::ext::to_value(&json!({ "entries": entries })) {
                Ok(v) => v,
                Err(e) => {
                    return error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        &format!("Failed to encode additional_information: {e:#}"),
                    );
                }
            }
        }
        None => rmpv::Value::Nil,
    };

    let overrides = SamplingOverrides {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        max_tokens: req.max_tokens.or(req.max_completion_tokens),
        min_tokens: req.min_tokens,
        seed: req.seed,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        stop: req.stop.map(StopSequences::into_vec),
        ignore_eos: None,
    };

    let request_id = format!("chatcmpl-{}", Uuid::new_v4());
    let result = match state
        .router
        .generate_chat(
            &request_id,
            prompt_token_ids,
            additional_info,
            Some(&overrides),
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            error!("[{request_id}] Generate failed: {e:#}");
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}"));
        }
    };

    let content = match state.tokenizer.decode(&result.text_token_ids, true) {
        Ok(s) => s,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("decode failed: {e:#}"),
            );
        }
    };
    let completion_tokens = result.text_token_ids.len() as u32;
    let finish_reason = finish_reason_str(result.finish_reason);

    let want_audio = req
        .modalities
        .as_ref()
        .map(|m| m.iter().any(|x| x == "audio"))
        .unwrap_or(false);
    let audio = if want_audio && !result.audio_chunks.is_empty() {
        let format = req
            .audio
            .as_ref()
            .map(|a| a.format.clone())
            .unwrap_or_else(default_audio_format);
        match extract_and_concat_audio(&result.audio_chunks, &format) {
            Ok((bytes, _content_type)) => Some(ChatCompletionAudio {
                id: format!("audio-{}", Uuid::new_v4()),
                data: BASE64.encode(bytes),
                expires_at: unix_now() + 24 * 3600,
                transcript: String::new(),
            }),
            Err(e) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("audio extraction: {e:#}"),
                );
            }
        }
    } else {
        None
    };

    let response = ChatCompletionResponse {
        id: request_id,
        object: "chat.completion",
        created: unix_now(),
        model: state.model_name.clone(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant",
                content,
                audio,
            },
            finish_reason,
        }],
        usage: UsageInfo {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };

    Json(response).into_response()
}

fn finish_reason_str(reason: Option<EngineCoreFinishReason>) -> &'static str {
    match reason {
        Some(EngineCoreFinishReason::Length) => "length",
        // Abort/Error/Repetition have no OpenAI equivalent; Stop and the
        // "no reason recorded" case both read as a normal completion.
        _ => "stop",
    }
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({"error": {"message": message, "type": "server_error"}})),
    )
        .into_response()
}
