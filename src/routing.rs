//! 2-stage TTS routing: Talker (stage 0) -> Code2Wav (stage 1).
//!
//! Implements the orchestrator's request lifecycle using EngineCoreClient
//! for direct ZMQ communication. Zero Python in the request path.

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use tracing::{debug, info};
use vllm_engine_core_client::EngineCoreClient;
use vllm_engine_core_client::protocol::request::EngineCoreRequest;

use crate::master::ConnectedStage;

/// Manages 2-stage TTS routing.
pub struct TtsRouter {
    pub stage0: Arc<EngineCoreClient>,
    pub stage1: Arc<EngineCoreClient>,
}

impl TtsRouter {
    pub fn new(stages: Vec<ConnectedStage>) -> Result<Self> {
        if stages.len() != 2 {
            bail!("TTS routing requires exactly 2 stages, got {}", stages.len());
        }

        let mut stage0 = None;
        let mut stage1 = None;
        for stage in stages {
            match stage.stage_id {
                0 => stage0 = Some(Arc::new(stage.client)),
                1 => stage1 = Some(Arc::new(stage.client)),
                id => bail!("Unexpected stage_id: {id}"),
            }
        }

        Ok(Self {
            stage0: stage0.context("Stage 0 not found")?,
            stage1: stage1.context("Stage 1 not found")?,
        })
    }

    // TODO: Implement generate_speech() that:
    // 1. Builds EngineCoreRequest for stage 0 (talker)
    // 2. Sends to stage 0 via ZMQ
    // 3. Collects stage 0 outputs (codec tokens)
    // 4. Bridges talker output to code2wav input
    // 5. Builds EngineCoreRequest for stage 1
    // 6. Sends to stage 1 via ZMQ
    // 7. Collects stage 1 output (audio tensor bytes)
    // 8. Returns audio PCM bytes
}
