// SPDX-License-Identifier: Apache-2.0
// Vendored from vLLM v0.25.0 and modified for vllm-omni.
// Field order MUST match Python's SamplingParams.__struct_fields__ exactly
// since both sides use array_like=True (tuple serialization).

use std::collections::{BTreeSet, HashMap};

use serde::{Deserialize, Serialize};
use serde_default::DefaultFromSerde;

use crate::protocol::OpaqueValue;

/// Sampling parameters matching Python's SamplingParams field order exactly.
///
/// Python field order (v0.25.0):
///  0: n, 1: presence_penalty, 2: frequency_penalty, 3: repetition_penalty,
///  4: temperature, 5: top_p, 6: top_k, 7: min_p, 8: seed, 9: stop,
/// 10: stop_token_ids, 11: ignore_eos, 12: max_tokens, 13: min_tokens,
/// 14: logprobs, 15: prompt_logprobs, 16: logprob_token_ids, 17: flat_logprobs,
/// 18: detokenize, 19: skip_special_tokens, 20: spaces_between_special_tokens,
/// 21: include_stop_str_in_output, 22: output_kind, 23: skip_clone,
/// 24: output_text_buffer_length, 25: _eos_token_id, 26: _all_stop_token_ids,
/// 27: structured_outputs, 28: logit_bias, 29: allowed_token_ids,
/// 30: extra_args, 31: routed_experts_prompt_start, 32: bad_words,
/// 33: _bad_words_token_ids, 34: skip_reading_prefix_cache,
/// 35: thinking_token_budget, 36: repetition_detection
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, DefaultFromSerde)]
pub struct EngineCoreSamplingParams {
    #[serde(default = "default_n")]
    pub n: u32,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: u32,
    #[serde(default)]
    pub min_p: f32,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default)]
    pub stop_token_ids: Vec<u32>,
    #[serde(default)]
    pub ignore_eos: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub min_tokens: u32,
    #[serde(default)]
    pub logprobs: Option<i32>,
    #[serde(default)]
    pub prompt_logprobs: Option<i32>,
    #[serde(default)]
    pub logprob_token_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub flat_logprobs: bool,
    #[serde(default = "default_true")]
    pub detokenize: bool,
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,
    #[serde(default = "default_true")]
    pub spaces_between_special_tokens: bool,
    #[serde(default)]
    pub include_stop_str_in_output: bool,
    #[serde(default)]
    pub output_kind: u8, // 0=CUMULATIVE, 1=DELTA, 2=FINAL_ONLY
    #[serde(default)]
    pub skip_clone: bool,
    #[serde(default)]
    pub output_text_buffer_length: u32,
    #[serde(rename = "_eos_token_id", default)]
    pub eos_token_id: Option<u32>,
    #[serde(rename = "_all_stop_token_ids", default)]
    pub all_stop_token_ids: BTreeSet<u32>,
    #[serde(default)]
    pub structured_outputs: Option<OpaqueValue>,
    #[serde(default)]
    pub logit_bias: Option<HashMap<u32, f32>>,
    #[serde(default)]
    pub allowed_token_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub extra_args: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub routed_experts_prompt_start: u32,
    #[serde(default)]
    pub bad_words: Vec<String>,
    #[serde(rename = "_bad_words_token_ids", default)]
    pub bad_words_token_ids: Option<Vec<Vec<u32>>>,
    #[serde(default)]
    pub skip_reading_prefix_cache: bool,
    #[serde(default)]
    pub thinking_token_budget: Option<u64>,
    #[serde(default)]
    pub repetition_detection: Option<OpaqueValue>,
}

impl EngineCoreSamplingParams {
    pub fn for_test() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            max_tokens: 65536,
            ..Default::default()
        }
    }
}

fn default_n() -> u32 { 1 }
fn default_temperature() -> f32 { 1.0 }
fn default_top_p() -> f32 { 1.0 }
fn default_repetition_penalty() -> f32 { 1.0 }
fn default_max_tokens() -> u32 { 16 }
fn default_true() -> bool { true }

// Keep old types for backward compatibility with other vendored code
pub type RepetitionDetectionParams = OpaqueValue;
pub type StructuredOutputsParams = OpaqueValue;
