//! 2-stage TTS routing: Talker (stage 0) -> Code2Wav (stage 1).
//!
//! Workers handle inter-stage data transfer via SharedMemoryConnector.
//! This router only manages request lifecycle and prompt construction.

use std::time::Instant;

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use tracing::{debug, info, warn};
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::OpaqueValue;
use vllm_engine_core_client::protocol::request::EngineCoreRequest;
use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;

use crate::master::ConnectedStage;

pub struct TtsRouter {
    pub stage0: EngineCoreClient,
    pub stage1: EngineCoreClient,
}

impl TtsRouter {
    pub fn new(stages: Vec<ConnectedStage>) -> Result<Self> {
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
        Ok(Self {
            stage0: s0.context("Stage 0 not found")?,
            stage1: s1.context("Stage 1 not found")?,
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
        let stage0_req = EngineCoreRequest {
            request_id: request_id.to_string(),
            prompt_token_ids: Some(prompt_token_ids),
            sampling_params,
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
            "[{request_id}] Stage 0 done: {token_count} tokens, {stage0_ms}ms"
        );

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
            request_id: format!("{request_id}-s1"),
            prompt_token_ids: Some(vec![0; stage1_prompt_len]),
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

    pub fn shutdown(self) -> Result<()> {
        // EngineCoreClient::shutdown is async, but we need sync here
        // The clients will be dropped which closes the ZMQ sockets
        tokio::spawn(async move {
            let _ = self.stage0.shutdown().await;
            let _ = self.stage1.shutdown().await;
        });
        Ok(())
    }
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}
