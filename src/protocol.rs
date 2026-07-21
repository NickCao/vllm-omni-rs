//! Extended wire protocol types for vllm-omni.
//!
//! Extends upstream EngineCoreOutput with omni-specific fields
//! (multimodal_output, is_segment_finished, new_prompt_len_snapshot).

use serde::{Deserialize, Serialize};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

/// Omni extension of EngineCoreOutput.
///
/// The upstream struct has 14 fields (positions 0-13).
/// Omni adds 3 more at positions 14-16.
/// We only need to decode the fields we use.
#[derive(Debug, Clone, Deserialize_tuple)]
pub struct OmniEngineCoreOutput {
    // Upstream fields (positions 0-13)
    pub request_id: String,
    pub new_token_ids: Vec<u32>,
    #[serde(default)]
    pub new_logprobs: Option<rmpv::Value>,
    #[serde(default)]
    pub new_prompt_logprobs_tensors: Option<rmpv::Value>,
    #[serde(default)]
    pub pooling_output: Option<rmpv::Value>,
    #[serde(default)]
    pub finish_reason: Option<u8>, // FinishReason as int
    #[serde(default)]
    pub stop_reason: Option<rmpv::Value>,
    #[serde(default)]
    pub events: Option<rmpv::Value>,
    #[serde(default)]
    pub kv_transfer_params: Option<rmpv::Value>,
    #[serde(default)]
    pub ec_transfer_params: Option<rmpv::Value>,
    #[serde(default)]
    pub trace_headers: Option<rmpv::Value>,
    #[serde(default)]
    pub prefill_stats: Option<rmpv::Value>,
    #[serde(default)]
    pub routed_experts: Option<rmpv::Value>,
    #[serde(default)]
    pub num_nans_in_logits: u32,

    // Omni extension fields (positions 14-16)
    #[serde(default)]
    pub multimodal_output: Option<rmpv::Value>,
    #[serde(default)]
    pub is_segment_finished: Option<bool>,
    #[serde(default)]
    pub new_prompt_len_snapshot: Option<u32>,
}

impl OmniEngineCoreOutput {
    pub fn finished(&self) -> bool {
        self.finish_reason.is_some()
    }
}
