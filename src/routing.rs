//! 2-stage TTS routing: Talker (stage 0) -> Code2Wav (stage 1).
//!
//! Workers handle inter-stage data transfer via SharedMemoryConnector.
//! This router only manages request lifecycle and prompt construction.

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
            bail!(
                "TTS routing requires exactly 2 stages, got {}",
                stages.len()
            );
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
        let tokenizer_path = tokenizer_path
            .context("--tokenizer-path is required")?;
        let tokenizer = HuggingFaceTokenizer::new(Path::new(tokenizer_path))
            .context(format!("Failed to load tokenizer from {tokenizer_path}"))?;
        info!("Tokenizer loaded from {tokenizer_path}");

        Ok(Self {
            stage0: s0.context("Stage 0 not found")?,
            stage1: s1.context("Stage 1 not found")?,
            tokenizer: Some(tokenizer),
        })
    }

    /// Generate speech end-to-end.
    ///
    /// Stage 0 (talker) generates codec tokens autoregressively.
    /// Stage 1 (code2wav) converts codec tokens to audio waveform.
    /// Inter-stage codec data flows through SharedMemoryConnector
    /// (handled by workers, not by us).
    ///
    /// Returns raw audio bytes from stage 1's multimodal_output.
    pub async fn generate_speech(
        &self,
        request_id: &str,
        prompt_token_ids: Vec<u32>,
        additional_info: OpaqueValue,
        sampling_params: Option<EngineCoreSamplingParams>,
    ) -> Result<Option<OpaqueValue>> {
        let start = Instant::now();

        // === Stage 0: Talker ===
        let sp = sampling_params.unwrap_or_else(default_tts_sampling_params);
        let stage0_req = EngineCoreRequest {
            request_id: request_id.to_string(),
            prompt_token_ids: Some(prompt_token_ids),
            sampling_params: Some(sp),
            arrival_time: now_secs(),
            additional_information: Some(additional_info.clone()),
            external_req_id: Some(request_id.to_string()),
            ..Default::default()
        };

        debug!("[{request_id}] Submitting to stage 0 (talker)");
        let mut stream0 = self
            .stage0
            .call(stage0_req)
            .await
            .context("Stage 0 call failed")?;

        let mut token_count = 0u64;
        let mut stage0_mm: Option<OpaqueValue> = None;

        while let Some(result) = stream0.next().await {
            let output = result.context("Stage 0 stream error")?;
            token_count += output.new_token_ids.len() as u64;
            if output.multimodal_output.is_some() {
                stage0_mm = output.multimodal_output.clone();
            }
            if output.finished() {
                break;
            }
        }

        let stage0_ms = start.elapsed().as_millis();
        info!(
            "[{request_id}] Stage 0 done: {token_count} tokens, {stage0_ms}ms, has_mm={}",
            stage0_mm.is_some()
        );
        if let Some(ref mm) = stage0_mm {
            debug!("[{request_id}] Stage 0 multimodal_output: {mm:?}");
        }

        // === Bridge: build stage 1 prompt ===
        // Stage 1 needs prompt_token_ids with the right length
        // for KV cache allocation. The actual codec data flows
        // through the SharedMemoryConnector.
        //
        // For non-async-chunk, talker2code2wav_token_only computes:
        //   prompt_len = num_quantizers * (ref_frames + audio_frames)
        // We estimate from stage 0's token count.
        let num_quantizers = 16u64; // Qwen3 TTS default
        let audio_frames = token_count.saturating_sub(1); // -1 for stop token
        let stage1_prompt_len = (num_quantizers * audio_frames) as usize;

        if stage1_prompt_len == 0 {
            warn!("[{request_id}] Stage 0 produced 0 audio frames");
            return Ok(None);
        }

        // Build stage 1 additional_information with speaker/language
        // (pass through from the original request)
        let stage1_req = EngineCoreRequest {
            request_id: request_id.to_string(),
            prompt_token_ids: Some(vec![0; stage1_prompt_len]),
            sampling_params: Some(default_tts_sampling_params()),
            arrival_time: now_secs(),
            additional_information: Some(additional_info),
            external_req_id: Some(request_id.to_string()),
            ..Default::default()
        };

        // === Stage 1: Code2Wav ===
        debug!(
            "[{request_id}] Submitting to stage 1 (code2wav), prompt_len={stage1_prompt_len}"
        );
        let mut stream1 = self
            .stage1
            .call(stage1_req)
            .await
            .context("Stage 1 call failed")?;

        let mut audio_output: Option<OpaqueValue> = None;

        while let Some(result) = stream1.next().await {
            let output = result.context("Stage 1 stream error")?;
            if output.multimodal_output.is_some() {
                audio_output = output.multimodal_output.clone();
            }
            if output.finished() {
                break;
            }
        }

        let total_ms = start.elapsed().as_millis();
        info!(
            "[{request_id}] Done: {total_ms}ms total, has_audio={}",
            audio_output.is_some()
        );

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
        // CustomVoice non_streaming_mode=True
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
        ..Default::default()
    }
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}
