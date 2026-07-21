//! POST /v1/audio/speech -- Pure Rust TTS endpoint.
//!
//! Uses TtsRouter for ZMQ-native 2-stage routing.
//! Zero Python per request.


use std::io::Cursor;

use anyhow::Result;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::Value;
use tracing::{error, debug};
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
    let response_format = req.response_format.clone();

    // Build additional_information for stage 0
    let task_type = req.extra.get("task_type").and_then(|v| v.as_str()).unwrap_or("CustomVoice");
    let language = req.extra.get("language").and_then(|v| v.as_str()).unwrap_or("Auto");
    let speaker = req.voice.as_deref().unwrap_or("Vivian");

    // Build AdditionalInformationPayload as rmpv::Value directly
    // Format: {"entries": {"key": {"list_data": [val], "tensor_data": nil, ...}, ...}}
    let make_entry = |vals: Vec<rmpv::Value>| -> rmpv::Value {
        rmpv::Value::Map(vec![
            (rmpv::Value::String("tensor_data".into()), rmpv::Value::Nil),
            (rmpv::Value::String("tensor_shape".into()), rmpv::Value::Nil),
            (rmpv::Value::String("tensor_dtype".into()), rmpv::Value::Nil),
            (rmpv::Value::String("list_data".into()), rmpv::Value::Array(vals)),
            (rmpv::Value::String("scalar_data".into()), rmpv::Value::Nil),
        ])
    };

    let mut entries = vec![
        (rmpv::Value::String("text".into()), make_entry(vec![rmpv::Value::String(req.input.clone().into())])),
        (rmpv::Value::String("task_type".into()), make_entry(vec![rmpv::Value::String(task_type.to_string().into())])),
        (rmpv::Value::String("language".into()), make_entry(vec![rmpv::Value::String(language.to_string().into())])),
        (rmpv::Value::String("speaker".into()), make_entry(vec![rmpv::Value::String(speaker.to_string().into())])),
    ];

    if let Some(ref inst) = req.instructions {
        entries.push((rmpv::Value::String("instruct".into()), make_entry(vec![rmpv::Value::String(inst.clone().into())])));
    }

    let additional_info: OpaqueValue = rmpv::Value::Map(vec![
        (rmpv::Value::String("entries".into()), rmpv::Value::Map(entries)),
    ]);

    let prompt_len = state.router.estimate_prompt_len(&req.input, req.instructions.as_deref());
    debug!("Prompt length estimate: {prompt_len} for input: {:?}", &req.input);
    let prompt_token_ids: Vec<u32> = vec![1; prompt_len];

    match state.router.generate_speech(
        &request_id,
        prompt_token_ids,
        additional_info,
        None,
    ).await {
        Ok(Some(audio_opaque)) => {
            debug!("Audio OpaqueValue: {:?}", audio_opaque);
                match extract_audio_from_opaque(&audio_opaque, &response_format) {
                Ok((audio_bytes, content_type)) => {
                    let mut headers = HeaderMap::new();
                    headers.insert(header::CONTENT_TYPE, content_type.parse().unwrap());
                    (headers, audio_bytes).into_response()
                }
                Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("Audio extraction: {e:#}")),
            }
        }
        Ok(None) => error_response(StatusCode::INTERNAL_SERVER_ERROR, "No audio output"),
        Err(e) => {
            error!("[{request_id}] Generate failed: {e:#}");
            error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("{e:#}"))
        }
    }
}

/// Extract audio PCM bytes from the multimodal_output OpaqueValue.
///
/// The wire format for tensors is (dtype_str, shape_tuple, raw_data).
/// For MultimodalPayload, it's a map with "audio" and "sr" tensor entries.
/// The raw_data may be inline bytes or a msgpack Ext type.
fn extract_audio_from_opaque(value: &OpaqueValue, response_format: &str) -> Result<(Vec<u8>, &'static str)> {
    let val = value;
    debug!("multimodal_output type: {:?}", val);

    // Navigate the structure to find audio tensor data.
    // MultimodalPayload serializes as a map with "tensors" and "metadata" keys,
    // where tensors contains {"audio": (dtype, shape, data), "sr": (dtype, shape, data)}.
    // Or it may be a flat map {"audio": tensor, "sr": tensor}.
    let audio_tensor = find_tensor_in_value(val, "audio")
        .or_else(|| find_tensor_in_value(val, "model_outputs"))
        .ok_or_else(|| anyhow::anyhow!("No audio tensor found in multimodal output"))?;
    let sr = find_scalar_in_value(val, "sr").unwrap_or(24000);

    // audio_tensor is (dtype_str, shape, raw_bytes)
    let pcm_f32_bytes = decode_tensor_bytes(&audio_tensor)?;

    encode_audio(&pcm_f32_bytes, sr, response_format)
}

/// Find a tensor value by key in a potentially nested rmpv structure.
fn find_tensor_in_value(val: &rmpv::Value, key: &str) -> Option<rmpv::Value> {
    match val {
        rmpv::Value::Map(entries) => {
            for (k, v) in entries {
                if let rmpv::Value::String(s) = k {
                    if s.as_str() == Some(key) {
                        return Some(v.clone());
                    }
                    if s.as_str() == Some("tensors") {
                        return find_tensor_in_value(v, key);
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Find a scalar integer value by key.
fn find_scalar_in_value(val: &rmpv::Value, key: &str) -> Option<u32> {
    match val {
        rmpv::Value::Map(entries) => {
            for (k, v) in entries {
                if let rmpv::Value::String(s) = k {
                    if s.as_str() == Some(key) {
                        return extract_scalar_u32(v);
                    }
                    if s.as_str() == Some("tensors") || s.as_str() == Some("metadata") {
                        if let Some(r) = find_scalar_in_value(v, key) {
                            return Some(r);
                        }
                    }
                }
            }
            None
        }
        _ => None,
    }
}

fn extract_scalar_u32(val: &rmpv::Value) -> Option<u32> {
    match val {
        rmpv::Value::Integer(i) => i.as_u64().map(|v| v as u32),
        // Tensor format: (dtype, shape, data) -- extract scalar from data
        rmpv::Value::Array(arr) if arr.len() == 3 => {
            // data is raw bytes, dtype tells us the type
            if let rmpv::Value::Binary(bytes) = &arr[2] {
                if bytes.len() == 4 {
                    return Some(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32);
                }
            }
            // Could be an Ext type
            if let rmpv::Value::Ext(_, bytes) = &arr[2] {
                if bytes.len() == 4 {
                    return Some(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32);
                }
            }
            None
        }
        _ => None,
    }
}

/// Decode raw bytes from a tensor wire format (dtype, shape, data).
fn decode_tensor_bytes(val: &rmpv::Value) -> Result<Vec<u8>> {
    match val {
        rmpv::Value::Array(arr) if arr.len() == 3 => {
            match &arr[2] {
                rmpv::Value::Binary(bytes) => Ok(bytes.clone()),
                rmpv::Value::Ext(_, bytes) => Ok(bytes.clone()),
                // Could be an integer index into aux buffers (zero-copy path)
                rmpv::Value::Integer(idx) => {
                    anyhow::bail!("Tensor uses aux buffer index {idx} -- zero-copy not supported in this path")
                }
                other => anyhow::bail!("Unexpected tensor data type: {:?}", other),
            }
        }
        rmpv::Value::Binary(bytes) => Ok(bytes.clone()),
        other => anyhow::bail!("Cannot decode tensor from: {:?}", other),
    }
}

fn encode_audio(pcm_f32_bytes: &[u8], sample_rate: u32, format: &str) -> Result<(Vec<u8>, &'static str)> {
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
            for &s in &samples_i16 { writer.write_sample(s)?; }
            writer.finalize()?;
            (cursor.into_inner(), "audio/wav")
        }
    })
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (status, Json(serde_json::json!({"error": {"message": message, "type": "server_error"}}))).into_response()
}
