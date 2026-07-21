//! Topology-driven pipeline routing.
//!
//! Handles any linear chain of stages wired via vllm-omni's async_chunk
//! connector (Qwen3-TTS's 2-stage Talker -> Code2Wav, Qwen3-Omni's 3-stage
//! Thinker -> Talker -> Code2Wav, etc.) -- not just a hardcoded 2-stage
//! pair. Root stages (no upstream source) get the request's real prompt;
//! downstream stages get a placeholder the connector extends as upstream
//! output arrives. Workers handle the actual codec transfer via
//! SharedMemoryConnector; this router only submits requests and collects
//! output from whichever stage produces audio.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use tracing::{debug, info};
use vllm_engine_core_client::protocol::OpaqueValue;
use vllm_engine_core_client::protocol::request::EngineCoreRequest;
use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;
use vllm_engine_core_client::{EngineCoreClient, EngineCoreOutputStream};
use vllm_tokenizer::{HuggingFaceTokenizer, Tokenizer};

use crate::introspect::StageTopology;
use crate::master::ConnectedStage;

// EngineCoreOutput.output_kind wire values.
const OUTPUT_KIND_DELTA: u8 = 1;
const OUTPUT_KIND_FINAL_ONLY: u8 = 2;

struct Stage {
    stage_id: u32,
    client: EngineCoreClient,
    /// No upstream source -- gets the request's real prompt directly,
    /// rather than a connector-fed placeholder.
    is_root: bool,
    sampling: EngineCoreSamplingParams,
}

pub struct TtsRouter {
    /// Ordered by stage_id, which is also submission/drain order.
    stages: Vec<Stage>,
    audio_stage_id: u32,
    tokenizer: HuggingFaceTokenizer,
}

impl TtsRouter {
    pub fn new(
        stages: Vec<ConnectedStage>,
        tokenizer_path: &str,
        topology: &HashMap<u32, StageTopology>,
    ) -> Result<Self> {
        if stages.is_empty() {
            bail!("pipeline requires at least 1 stage, got 0");
        }
        if stages.len() != topology.len() {
            bail!(
                "connected {} stage(s) but pipeline topology declares {}",
                stages.len(),
                topology.len()
            );
        }

        let audio_stages: Vec<u32> = topology
            .iter()
            .filter(|(_, t)| t.final_output_type.as_deref() == Some("audio"))
            .map(|(id, _)| *id)
            .collect();
        let audio_stage_id = match audio_stages.as_slice() {
            [id] => *id,
            [] => bail!("no stage in the pipeline has final_output_type=\"audio\""),
            ids => bail!("expected exactly one audio-output stage, found {ids:?}"),
        };

        let mut built = Vec::with_capacity(stages.len());
        for s in stages {
            let t = topology.get(&s.stage_id).with_context(|| {
                format!(
                    "stage {} connected but missing from introspected topology",
                    s.stage_id
                )
            })?;
            // The audio stage streams incrementally, so its output must be
            // DELTA for generate_speech's chunk accumulation to see each
            // new slice. Every other stage only needs its final token
            // count, so FINAL_ONLY avoids per-step deltas we'd discard
            // anyway. This is a choice about how *we* consume the stream,
            // not part of the model's own config, so it's applied here
            // rather than looked up from the introspected topology.
            let output_kind = if s.stage_id == audio_stage_id {
                OUTPUT_KIND_DELTA
            } else {
                OUTPUT_KIND_FINAL_ONLY
            };
            built.push(Stage {
                stage_id: s.stage_id,
                client: s.client,
                is_root: t.is_root(),
                sampling: t.default_sampling_params.to_sampling_params(output_kind),
            });
        }
        built.sort_by_key(|s| s.stage_id);

        let tokenizer = HuggingFaceTokenizer::new(Path::new(tokenizer_path))
            .context(format!("Failed to load tokenizer from {tokenizer_path}"))?;
        info!("Tokenizer loaded from {tokenizer_path}");

        Ok(Self {
            stages: built,
            audio_stage_id,
            tokenizer,
        })
    }

    /// Generate speech. Submits every stage in the pipeline concurrently
    /// (async_chunk mode), matching what vllm-omni's Python orchestrator
    /// does in `_prewarm_async_chunk_stages`. Root stages get the real
    /// prompt; downstream stages get a placeholder the connector extends
    /// as upstream output arrives via /dev/shm -- Rust never touches that
    /// transfer directly.
    pub async fn generate_speech(
        &self,
        request_id: &str,
        prompt_token_ids: Vec<u32>,
        additional_info: OpaqueValue,
    ) -> Result<Vec<OpaqueValue>> {
        let start = Instant::now();

        let mut streams: Vec<(u32, EngineCoreOutputStream)> = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            // Python's _prewarm_async_chunk_stages uses
            // compute_talker_prompt_ids_length(stage0_prompt), which
            // returns 1 for placeholder prompts on any downstream stage --
            // this isn't specific to code2wav.
            let (prompt, resumable) = if stage.is_root {
                (prompt_token_ids.clone(), false)
            } else {
                (vec![0u32; 1], true)
            };
            let req = EngineCoreRequest {
                request_id: request_id.to_string(),
                prompt_token_ids: Some(prompt),
                sampling_params: Some(stage.sampling.clone()),
                arrival_time: now_secs(),
                additional_information: Some(additional_info.clone()),
                external_req_id: Some(request_id.to_string()),
                resumable,
                ..Default::default()
            };
            debug!("[{request_id}] Submitting to stage {}", stage.stage_id);
            let stream = stage
                .client
                .call(req)
                .await
                .with_context(|| format!("Stage {} call failed", stage.stage_id))?;
            streams.push((stage.stage_id, stream));
        }

        // Drain every non-audio stage to completion first (only used for
        // logging), then the audio stage last. This preserves the original
        // 2-stage drain order that was verified to work end to end; the
        // actual codec transfer between stages happens out-of-band via
        // /dev/shm regardless of the order Rust polls each stream in.
        let mut audio_stream: Option<EngineCoreOutputStream> = None;
        for (stage_id, mut stream) in streams {
            if stage_id == self.audio_stage_id {
                audio_stream = Some(stream);
                continue;
            }
            let mut token_count = 0u64;
            while let Some(result) = stream.next().await {
                let output = result.with_context(|| format!("Stage {stage_id} stream error"))?;
                token_count += output.new_token_ids.len() as u64;
                if output.finished() {
                    break;
                }
            }
            info!(
                "[{request_id}] Stage {stage_id} done: {token_count} tokens, {}ms",
                start.elapsed().as_millis()
            );
        }

        // Stage output uses DELTA output_kind, so each multimodal_output is
        // a NEW chunk of audio, not the full buffer -- these must be
        // accumulated, not overwritten, matching the Python frontend's
        // torch.cat over all streamed chunks.
        let mut audio_stream =
            audio_stream.context("audio stage stream missing after submission")?;
        let mut audio_chunks: Vec<OpaqueValue> = Vec::new();
        while let Some(result) = audio_stream.next().await {
            let output = result.context("audio stage stream error")?;
            if let Some(mm) = output.multimodal_output.clone() {
                audio_chunks.push(mm);
            }
            if output.finished() {
                break;
            }
        }

        let total_ms = start.elapsed().as_millis();
        info!(
            "[{request_id}] Done: {total_ms}ms, chunks={}",
            audio_chunks.len()
        );

        Ok(audio_chunks)
    }

    pub fn estimate_prompt_len(&self, text: &str, instruct: Option<&str>) -> usize {
        let tokenize = |s: &str| -> usize {
            self.tokenizer
                .encode(s, false)
                .map(|ids| ids.len())
                .unwrap_or(0)
        };
        let instruct_len = match instruct {
            Some(i) if !i.trim().is_empty() => {
                tokenize(&format!("<|im_start|>user\n{i}<|im_end|>\n"))
            }
            _ => 0,
        };
        let assistant_len = tokenize(&format!(
            "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        ));
        instruct_len + 3 + 5 + assistant_len.saturating_sub(6)
    }

    pub fn shutdown(self) -> Result<()> {
        tokio::spawn(async move {
            for stage in self.stages {
                let _ = stage.client.shutdown().await;
            }
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
