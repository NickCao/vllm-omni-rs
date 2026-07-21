//! Topology-driven pipeline routing.
//!
//! Handles any linear chain of stages wired via vllm-omni's async_chunk
//! connector (Qwen3-TTS's 2-stage Talker -> Code2Wav, Qwen3-Omni's 3-stage
//! Thinker -> Talker -> Code2Wav, etc.). Root stages (no upstream source)
//! get the request's real prompt; downstream stages get a placeholder the
//! connector extends as upstream output arrives. Workers handle the actual
//! codec/embedding transfer via SharedMemoryConnector; this router only
//! submits requests and collects output from whichever stage(s) produce
//! text and/or audio.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use tracing::{debug, info};
use vllm_engine_core_client::protocol::OpaqueValue;
use vllm_engine_core_client::protocol::output::EngineCoreFinishReason;
use vllm_engine_core_client::protocol::request::EngineCoreRequest;
use vllm_engine_core_client::{EngineCoreClient, EngineCoreOutputStream};
use vllm_tokenizer::{HuggingFaceTokenizer, Tokenizer};

use crate::introspect::{SamplingOverrides, StageSamplingDefaults, StageTopology};
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
    /// Kept raw (not pre-baked into `EngineCoreSamplingParams`) so a chat
    /// completion's per-request overrides can be re-applied fresh on top
    /// for the root stage on every request, without needing to invert
    /// whatever was baked in at construction time.
    sampling_defaults: StageSamplingDefaults,
    output_kind: u8,
}

/// Output of a chat-completion generation: accumulated thinker token ids
/// (decode with the router's tokenizer) and, if audio was produced,
/// DELTA-mode multimodal_output chunks in the same shape `generate_speech`
/// returns.
pub struct ChatGenerationResult {
    pub text_token_ids: Vec<u32>,
    pub audio_chunks: Vec<OpaqueValue>,
    pub finish_reason: Option<EngineCoreFinishReason>,
}

pub struct PipelineRouter {
    /// Ordered by stage_id, which is also submission/drain order.
    stages: Vec<Stage>,
    audio_stage_id: Option<u32>,
    text_stage_id: Option<u32>,
    tokenizer: Arc<HuggingFaceTokenizer>,
}

impl PipelineRouter {
    pub fn new(
        stages: Vec<ConnectedStage>,
        tokenizer: Arc<HuggingFaceTokenizer>,
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

        let audio_stage_id = single_stage_with_output_type(topology, "audio")?;
        let text_stage_id = single_stage_with_output_type(topology, "text")?;
        if audio_stage_id.is_none() && text_stage_id.is_none() {
            bail!("pipeline has neither a text-output nor an audio-output stage");
        }

        let mut built = Vec::with_capacity(stages.len());
        for s in stages {
            let t = topology.get(&s.stage_id).with_context(|| {
                format!(
                    "stage {} connected but missing from introspected topology",
                    s.stage_id
                )
            })?;
            // The audio stage streams incrementally, so its output must be
            // DELTA for chunk accumulation to see each new slice. Every
            // other stage (including the text stage) only needs its final
            // accumulated tokens, so FINAL_ONLY avoids per-step deltas we'd
            // discard anyway. This is a choice about how *we* consume the
            // stream, not part of the model's own config, so it's applied
            // here rather than looked up from the introspected topology.
            let output_kind = if Some(s.stage_id) == audio_stage_id {
                OUTPUT_KIND_DELTA
            } else {
                OUTPUT_KIND_FINAL_ONLY
            };
            built.push(Stage {
                stage_id: s.stage_id,
                client: s.client,
                is_root: t.is_root(),
                sampling_defaults: t.default_sampling_params.clone(),
                output_kind,
            });
        }
        built.sort_by_key(|s| s.stage_id);

        Ok(Self {
            stages: built,
            audio_stage_id,
            text_stage_id,
            tokenizer,
        })
    }

    pub fn supports_speech(&self) -> bool {
        self.audio_stage_id.is_some()
    }

    pub fn supports_chat(&self) -> bool {
        self.text_stage_id.is_some()
    }

    /// Generate speech: collects audio from the pipeline's audio-output
    /// stage. `additional_information` is not overridable per request
    /// (TTS steering fields ride along in it directly), so there's no
    /// sampling-overrides parameter here.
    pub async fn generate_speech(
        &self,
        request_id: &str,
        prompt_token_ids: Vec<u32>,
        additional_info: OpaqueValue,
    ) -> Result<Vec<OpaqueValue>> {
        self.audio_stage_id
            .context("pipeline has no audio-output stage")?;
        let start = Instant::now();
        let streams = self
            .submit_all(request_id, prompt_token_ids, additional_info, None)
            .await?;
        let (_, audio_chunks, _) = self.drain_pipeline(request_id, start, streams).await?;
        info!(
            "[{request_id}] Done: {}ms, chunks={}",
            start.elapsed().as_millis(),
            audio_chunks.len()
        );
        Ok(audio_chunks)
    }

    /// Generate a chat completion: collects decoded text from the
    /// pipeline's text-output stage, and audio from its audio-output stage
    /// if one exists (the caller decides whether to use it based on the
    /// request's requested modalities). `overrides` applies only to the
    /// root stage, matching vllm-omni's own behavior of never letting a
    /// client-set sampling field reach downstream stages.
    pub async fn generate_chat(
        &self,
        request_id: &str,
        prompt_token_ids: Vec<u32>,
        additional_info: OpaqueValue,
        overrides: Option<&SamplingOverrides>,
    ) -> Result<ChatGenerationResult> {
        self.text_stage_id
            .context("pipeline has no text-output stage")?;
        let start = Instant::now();
        let streams = self
            .submit_all(request_id, prompt_token_ids, additional_info, overrides)
            .await?;
        let (text_token_ids, audio_chunks, finish_reason) =
            self.drain_pipeline(request_id, start, streams).await?;
        info!(
            "[{request_id}] Done: {}ms, text_tokens={}, audio_chunks={}",
            start.elapsed().as_millis(),
            text_token_ids.len(),
            audio_chunks.len()
        );
        Ok(ChatGenerationResult {
            text_token_ids,
            audio_chunks,
            finish_reason,
        })
    }

    /// Submits every stage in the pipeline concurrently (async_chunk
    /// mode), matching what vllm-omni's Python orchestrator does in
    /// `_prewarm_async_chunk_stages`. Root stages get the real prompt;
    /// downstream stages get a placeholder the connector extends as
    /// upstream output arrives via /dev/shm -- Rust never touches that
    /// transfer directly, regardless of what it's carrying (codec tokens
    /// for Qwen3-TTS, embeddings/hidden states for Qwen3-Omni's
    /// thinker->talker hop).
    async fn submit_all(
        &self,
        request_id: &str,
        prompt_token_ids: Vec<u32>,
        additional_info: OpaqueValue,
        overrides: Option<&SamplingOverrides>,
    ) -> Result<Vec<(u32, EngineCoreOutputStream)>> {
        let mut streams = Vec::with_capacity(self.stages.len());
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
            let sampling = stage.sampling_defaults.to_sampling_params(
                stage.output_kind,
                if stage.is_root { overrides } else { None },
            );
            let req = EngineCoreRequest {
                request_id: request_id.to_string(),
                prompt_token_ids: Some(prompt),
                sampling_params: Some(sampling),
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
        Ok(streams)
    }

    /// Drains every non-audio stage to completion first (in stage-id
    /// order), accumulating the text stage's tokens along the way, then
    /// drains the audio stage last. This preserves the original 2-stage
    /// drain order that was verified to work end to end; the actual
    /// inter-stage transfer happens out-of-band via /dev/shm regardless of
    /// the order Rust polls each stream in.
    async fn drain_pipeline(
        &self,
        request_id: &str,
        start: Instant,
        streams: Vec<(u32, EngineCoreOutputStream)>,
    ) -> Result<(Vec<u32>, Vec<OpaqueValue>, Option<EngineCoreFinishReason>)> {
        let mut text_token_ids: Vec<u32> = Vec::new();
        let mut text_finish_reason: Option<EngineCoreFinishReason> = None;
        let mut audio_stream: Option<EngineCoreOutputStream> = None;
        for (stage_id, mut stream) in streams {
            if Some(stage_id) == self.audio_stage_id {
                audio_stream = Some(stream);
                continue;
            }
            let mut token_count = 0u64;
            while let Some(result) = stream.next().await {
                let output = result.with_context(|| format!("Stage {stage_id} stream error"))?;
                if Some(stage_id) == self.text_stage_id {
                    text_token_ids.extend_from_slice(&output.new_token_ids);
                    if let Some(reason) = output.finish_reason {
                        text_finish_reason = Some(reason);
                    }
                }
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

        // Audio output uses DELTA output_kind, so each multimodal_output is
        // a NEW chunk of audio, not the full buffer -- these must be
        // accumulated, not overwritten, matching the Python frontend's
        // torch.cat over all streamed chunks.
        let mut audio_chunks: Vec<OpaqueValue> = Vec::new();
        if let Some(mut stream) = audio_stream {
            while let Some(result) = stream.next().await {
                let output = result.context("audio stage stream error")?;
                if let Some(mm) = output.multimodal_output.clone() {
                    audio_chunks.push(mm);
                }
                if output.finished() {
                    break;
                }
            }
        }

        Ok((text_token_ids, audio_chunks, text_finish_reason))
    }

    /// Qwen3-TTS-specific: reimplements
    /// `Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information`'s
    /// chat-template token-counting arithmetic. Not reusable for
    /// chat-completion prompts, which are rendered and tokenized directly.
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

/// Returns the single stage_id whose `final_output_type` matches, `None` if
/// no stage matches (that modality just isn't supported by this pipeline),
/// or an error if more than one stage claims the same output type.
fn single_stage_with_output_type(
    topology: &HashMap<u32, StageTopology>,
    output_type: &str,
) -> Result<Option<u32>> {
    let matches: Vec<u32> = topology
        .iter()
        .filter(|(_, t)| t.final_output_type.as_deref() == Some(output_type))
        .map(|(id, _)| *id)
        .collect();
    match matches.as_slice() {
        [] => Ok(None),
        [id] => Ok(Some(*id)),
        ids => bail!("expected at most one {output_type:?}-output stage, found {ids:?}"),
    }
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}
