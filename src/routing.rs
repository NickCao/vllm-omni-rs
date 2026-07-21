//! 2-stage TTS routing: Talker (stage 0) -> Code2Wav (stage 1).
//!
//! With async_chunk enabled, workers handle inter-stage codec transfer
//! via SharedMemoryConnector. This router submits requests to both stages
//! and collects audio from stage 1.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use tracing::{debug, info, warn};
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::OpaqueValue;
use vllm_engine_core_client::protocol::request::EngineCoreRequest;
use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;
use vllm_tokenizer::{HuggingFaceTokenizer, Tokenizer};

use crate::master::ConnectedStage;

pub struct TtsRouter {
    pub stage0: EngineCoreClient,
    pub stage1: EngineCoreClient,
    tokenizer: Option<HuggingFaceTokenizer>,
}

impl TtsRouter {
    pub fn new(stages: Vec<ConnectedStage>, tokenizer_path: Option<&str>) -> Result<Self> {
        if stages.len() != 2 {
            bail!("TTS requires 2 stages, got {}", stages.len());
        }
        let mut s0 = None;
        let mut s1 = None;
        for s in stages {
            match s.stage_id {
                0 => s0 = Some(s.client),
                1 => s1 = Some(s.client),
                id => bail!("Unexpected stage_id: {id}"),
            }
        }

        let tokenizer_path = tokenizer_path.context("--tokenizer-path is required")?;
        let tokenizer = HuggingFaceTokenizer::new(Path::new(tokenizer_path))
            .context(format!("Failed to load tokenizer from {tokenizer_path}"))?;
        info!("Tokenizer loaded from {tokenizer_path}");

        Ok(Self {
            stage0: s0.context("Stage 0 not found")?,
            stage1: s1.context("Stage 1 not found")?,
            tokenizer: Some(tokenizer),
        })
    }

    /// Generate speech. Submit to both stages concurrently (async_chunk mode).
    /// Stage 0 generates codec tokens, workers transfer via /dev/shm,
    /// stage 1 generates audio.
    pub async fn generate_speech(
        &self,
        request_id: &str,
        prompt_token_ids: Vec<u32>,
        additional_info: OpaqueValue,
        _sampling_params: Option<EngineCoreSamplingParams>,
    ) -> Result<Option<OpaqueValue>> {
        let start = Instant::now();

        // Submit stage 0 (talker)
        let stage0_req = EngineCoreRequest {
            request_id: request_id.to_string(),
            prompt_token_ids: Some(prompt_token_ids),
            sampling_params: Some(default_tts_sampling_params()),
            arrival_time: now_secs(),
            additional_information: Some(additional_info.clone()),
            external_req_id: Some(request_id.to_string()),
            ..Default::default()
        };

        debug!("[{request_id}] Submitting to stage 0 (talker)");
        let mut stream0 = self.stage0.call(stage0_req).await
            .context("Stage 0 call failed")?;

        // Submit stage 1 (code2wav) with a minimal placeholder.
        // Python's _prewarm_async_chunk_stages uses
        // compute_talker_prompt_ids_length(stage0_prompt) which returns 1
        // for placeholder prompts. The actual codec data arrives via the
        // connector and the scheduler dynamically extends the request.
        let stage1_prompt_len = 1usize;
        let stage1_req = EngineCoreRequest {
            request_id: request_id.to_string(),
            prompt_token_ids: Some(vec![0; stage1_prompt_len]),
            sampling_params: Some(default_code2wav_sampling_params()),
            arrival_time: now_secs(),
            additional_information: Some(additional_info),
            external_req_id: Some(request_id.to_string()),
            resumable: true, // async_chunk streaming input
            ..Default::default()
        };

        debug!("[{request_id}] Submitting to stage 1 (code2wav)");
        let mut stream1 = self.stage1.call(stage1_req).await
            .context("Stage 1 call failed")?;

        // Wait for stage 0 to finish (generates codec tokens)
        let mut token_count = 0u64;
        while let Some(result) = stream0.next().await {
            let output = result.context("Stage 0 stream error")?;
            token_count += output.new_token_ids.len() as u64;
            if output.finished() { break; }
        }
        let stage0_ms = start.elapsed().as_millis();
        info!("[{request_id}] Stage 0 done: {token_count} tokens, {stage0_ms}ms");

        // Collect stage 1 output (audio)
        let mut audio_output: Option<OpaqueValue> = None;
        while let Some(result) = stream1.next().await {
            let output = result.context("Stage 1 stream error")?;
            if output.multimodal_output.is_some() {
                audio_output = output.multimodal_output.clone();
            }
            if output.finished() { break; }
        }

        let total_ms = start.elapsed().as_millis();
        info!("[{request_id}] Done: {total_ms}ms, has_audio={}", audio_output.is_some());

        Ok(audio_output)
    }

    pub fn estimate_prompt_len(&self, text: &str, instruct: Option<&str>) -> usize {
        let Some(ref tok) = self.tokenizer else { return 2048; };
        let tokenize = |s: &str| -> usize {
            tok.encode(s, false).map(|ids| ids.len()).unwrap_or(0)
        };
        let instruct_len = match instruct {
            Some(i) if !i.trim().is_empty() =>
                tokenize(&format!("<|im_start|>user\n{i}<|im_end|>\n")),
            _ => 0,
        };
        let assistant_len = tokenize(
            &format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n")
        );
        instruct_len + 3 + 5 + assistant_len.saturating_sub(6)
    }

    pub fn shutdown(self) -> Result<()> {
        tokio::spawn(async move {
            let _ = self.stage0.shutdown().await;
            let _ = self.stage1.shutdown().await;
        });
        Ok(())
    }
}

fn default_tts_sampling_params() -> EngineCoreSamplingParams {
    EngineCoreSamplingParams {
        temperature: 0.9,
        top_p: 1.0,
        top_k: 50,
        repetition_penalty: 1.05,
        max_tokens: 4096,
        min_tokens: 2,
        stop_token_ids: vec![2150],
        all_stop_token_ids: [2150].into(),
        output_kind: 2, // FINAL_ONLY
        ..Default::default()
    }
}

fn default_code2wav_sampling_params() -> EngineCoreSamplingParams {
    EngineCoreSamplingParams {
        max_tokens: 65536,
        output_kind: 2, // FINAL_ONLY
        ..Default::default()
    }
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}
