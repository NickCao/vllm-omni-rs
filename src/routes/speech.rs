//! POST /v1/audio/speech -- Pure Rust TTS endpoint.
//!
//! Uses TtsRouter for ZMQ-native 2-stage routing.
//! Zero Python per request.

use std::io::Cursor;

use anyhow::{Context, Result};
use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::{debug, error};
use uuid::Uuid;
use vllm_engine_core_client::protocol::OpaqueValue;
use vllm_engine_core_client::protocol::tensor::WireTensor;

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

/// One entry of vllm-omni's `AdditionalInformationPayload` wire struct.
///
/// Built through `serde_json::Value` rather than a `#[derive(Serialize)]`
/// struct: `rmpv::ext`'s serializer encodes plain structs as msgpack
/// arrays (matching vLLM's `array_like=True` convention elsewhere), but
/// `AdditionalInformationPayload` is a plain dict on the Python side and
/// must round-trip as a msgpack map. `serde_json::Value` serializes
/// objects through `serialize_map`, which `rmpv::ext` does encode as a map.
fn text_entry(value: impl Into<String>) -> Value {
    json!({
        "tensor_data": null,
        "tensor_shape": null,
        "tensor_dtype": null,
        "list_data": [value.into()],
        "scalar_data": null,
    })
}

pub async fn create_speech(
    State(state): State<AppState>,
    Json(req): Json<SpeechRequest>,
) -> Response {
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

/// Stage 1's `multimodal_output`. Qwen3-TTS code2wav emits a flat
/// `{"model_outputs": tensor, "sr": tensor}` map; other omni models nest
/// tensors under `"tensors"` or place the sample rate under `"metadata"`,
/// so both are checked as fallbacks.
#[derive(Debug, Default, Deserialize)]
struct MultimodalOutput {
    #[serde(default)]
    audio: Option<WireTensor>,
    #[serde(default)]
    model_outputs: Option<WireTensor>,
    #[serde(default)]
    sr: Option<SampleRate>,
    #[serde(default)]
    tensors: Option<Box<MultimodalOutput>>,
    #[serde(default)]
    metadata: Option<Metadata>,
}

#[derive(Debug, Default, Deserialize)]
struct Metadata {
    #[serde(default)]
    sr: Option<SampleRate>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum SampleRate {
    Scalar(u32),
    Tensor(WireTensor),
}

impl SampleRate {
    fn resolve(&self) -> Result<u32> {
        match self {
            Self::Scalar(v) => Ok(*v),
            Self::Tensor(t) => scalar_from_tensor(t),
        }
    }
}

impl MultimodalOutput {
    fn audio_tensor(&self) -> Option<&WireTensor> {
        self.audio
            .as_ref()
            .or(self.model_outputs.as_ref())
            .or_else(|| self.tensors.as_deref().and_then(Self::audio_tensor))
    }

    fn sample_rate(&self) -> Option<&SampleRate> {
        self.sr
            .as_ref()
            .or_else(|| self.metadata.as_ref().and_then(|m| m.sr.as_ref()))
            .or_else(|| self.tensors.as_deref().and_then(Self::sample_rate))
    }
}

/// Extract PCM bytes from each chunk (DELTA output_kind -- each chunk is a
/// new audio slice) and concatenate them in order, matching the Python
/// frontend's `torch.cat` over all streamed audio deltas.
fn extract_and_concat_audio(
    chunks: &[OpaqueValue],
    response_format: &str,
) -> Result<(Vec<u8>, &'static str)> {
    let mut sr = 24000u32;
    let mut pcm_f32_bytes: Vec<u8> = Vec::new();
    for chunk in chunks {
        let mm: MultimodalOutput =
            rmpv::ext::from_value(chunk.clone()).context("failed to decode multimodal_output")?;
        let Some(audio) = mm.audio_tensor() else {
            continue;
        };
        if let Some(rate) = mm.sample_rate() {
            sr = rate.resolve()?;
        }
        pcm_f32_bytes.extend_from_slice(raw_view_bytes(audio)?);
    }
    if pcm_f32_bytes.is_empty() {
        anyhow::bail!("No audio tensor found in any multimodal output chunk");
    }
    encode_audio(&pcm_f32_bytes, sr, response_format)
}

fn raw_view_bytes(tensor: &WireTensor) -> Result<&[u8]> {
    tensor
        .data
        .as_raw_view()
        .map(Vec::as_slice)
        .with_context(|| {
            format!(
                "tensor '{}' uses an unresolved aux-frame reference",
                tensor.dtype
            )
        })
}

fn scalar_from_tensor(tensor: &WireTensor) -> Result<u32> {
    let bytes = raw_view_bytes(tensor)?;
    match tensor.dtype.as_str() {
        "int32" | "uint32" => Ok(i32::from_le_bytes(bytes[..4].try_into()?) as u32),
        "int64" | "uint64" => Ok(i64::from_le_bytes(bytes[..8].try_into()?) as u32),
        other => anyhow::bail!("unsupported scalar dtype for sample rate: {other}"),
    }
}

fn encode_audio(
    pcm_f32_bytes: &[u8],
    sample_rate: u32,
    format: &str,
) -> Result<(Vec<u8>, &'static str)> {
    let samples_f32: &[f32] = bytemuck::cast_slice(pcm_f32_bytes);
    let samples_i16: Vec<i16> = samples_f32
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect();

    Ok(match format {
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
            let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
            for &s in &samples_i16 {
                writer.write_sample(s)?;
            }
            writer.finalize()?;
            (cursor.into_inner(), "audio/wav")
        }
    })
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({"error": {"message": message, "type": "server_error"}})),
    )
        .into_response()
}
