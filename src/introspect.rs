//! One-time startup calls into Python to pull model-specific config out of
//! vllm-omni instead of hardcoding it in Rust.
//!
//! Both helpers here shell out to `python3` exactly once at startup, never
//! per-request -- the zero-per-request-Python invariant only applies to the
//! request path itself.

use std::collections::HashMap;
use std::process::Command;

use anyhow::{Context, Result, bail};
use tracing::info;
use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;

/// Extract tokenizer.json from the model using a one-time Python call.
/// The HF repo doesn't ship tokenizer.json but the Python tokenizer
/// builds one from vocab.json + merges.txt.
pub fn extract_tokenizer(model: &str) -> Result<String> {
    let path = format!(
        "/tmp/_vllm_omni_rs_tokenizer_{}.json",
        model.replace('/', "_")
    );
    if std::path::Path::new(&path).exists() {
        info!("Using cached tokenizer: {path}");
        return Ok(path);
    }
    // model/path are passed as argv, not interpolated into the script, so
    // neither can break out of the Python string literal they'd otherwise sit in.
    const SCRIPT: &str = "import sys; \
        from transformers import AutoTokenizer; \
        t = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True); \
        t.backend_tokenizer.save(sys.argv[2])";
    let output = Command::new("python3")
        .args(["-c", SCRIPT, model, &path])
        .output()
        .context("failed to run python3 for tokenizer extraction")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("tokenizer extraction failed: {stderr}");
    }
    info!("Extracted tokenizer to {path}");
    Ok(path)
}

/// Per-stage sampling defaults as vllm-omni itself resolves them: the deploy
/// YAML's `default_sampling_params` merged with the pipeline topology's
/// `sampling_constraints` (e.g. Qwen3-TTS's talker `stop_token_ids: [2150]`).
/// Every field is optional because different stages/models set different
/// subsets -- anything unset here keeps `EngineCoreSamplingParams`'s own
/// default.
#[derive(Debug, Default, serde::Deserialize)]
pub struct StageSamplingDefaults {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    /// vLLM's convention allows `top_k <= 0` to mean "disabled"; negative
    /// values can't fit `EngineCoreSamplingParams::top_k: u32` so they're
    /// normalized to `0` in `to_sampling_params`, matching what vLLM itself
    /// does internally whenever greedy sampling (`temperature == 0`) already
    /// forces `top_k` back to `0` regardless of the configured value.
    #[serde(default)]
    pub top_k: Option<i64>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub min_tokens: Option<u32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub stop_token_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub detokenize: Option<bool>,
}

impl StageSamplingDefaults {
    /// Build sampling params for one request. `output_kind` is not part of
    /// the introspected model config -- it's this frontend's own choice of
    /// how to consume the output stream (FINAL_ONLY vs DELTA), so it's
    /// passed in rather than looked up.
    pub fn to_sampling_params(&self, output_kind: u8) -> EngineCoreSamplingParams {
        let mut params = EngineCoreSamplingParams {
            output_kind,
            ..Default::default()
        };
        if let Some(v) = self.temperature {
            params.temperature = v;
        }
        if let Some(v) = self.top_p {
            params.top_p = v;
        }
        if let Some(v) = self.top_k {
            params.top_k = v.max(0) as u32;
        }
        if let Some(v) = self.max_tokens {
            params.max_tokens = v;
        }
        if let Some(v) = self.min_tokens {
            params.min_tokens = v;
        }
        if let Some(v) = self.repetition_penalty {
            params.repetition_penalty = v;
        }
        if let Some(ref ids) = self.stop_token_ids {
            params.all_stop_token_ids = ids.iter().copied().collect();
            params.stop_token_ids = ids.clone();
        }
        if let Some(v) = self.detokenize {
            params.detokenize = v;
        }
        params
    }
}

/// Resolve each stage's `default_sampling_params` the same way vllm-omni's
/// own `run_headless()` does: `load_and_resolve_stage_configs` merges the
/// deploy YAML with the model's pipeline-topology sampling constraints, so
/// this reads whatever the model actually declares instead of guessing at
/// per-model tuning values from Rust.
pub fn introspect_stage_sampling_params(
    model: &str,
) -> Result<HashMap<u32, StageSamplingDefaults>> {
    // Log noise from vllm-omni's own imports can land on either stream, so
    // the JSON payload is wrapped in unambiguous markers on stderr instead
    // of relying on stdout being clean.
    const SCRIPT: &str = "import sys, json; \
        from omegaconf import OmegaConf; \
        from vllm_omni.entrypoints.utils import load_and_resolve_stage_configs; \
        _, stage_configs, _ = load_and_resolve_stage_configs(sys.argv[1], None, {}); \
        out = {str(cfg.stage_id): OmegaConf.to_container(cfg.default_sampling_params, resolve=True) for cfg in stage_configs}; \
        sys.stderr.write('===VLLM_OMNI_RS_JSON_START===\\n' + json.dumps(out) + '\\n===VLLM_OMNI_RS_JSON_END===\\n')";
    let output = Command::new("python3")
        .args(["-c", SCRIPT, model])
        .output()
        .context("failed to run python3 for stage config introspection")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("stage config introspection failed: {stderr}");
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let json_str = stderr
        .split("===VLLM_OMNI_RS_JSON_START===\n")
        .nth(1)
        .and_then(|s| s.split("\n===VLLM_OMNI_RS_JSON_END===").next())
        .context("stage config introspection output missing JSON markers")?;
    let raw: HashMap<String, StageSamplingDefaults> =
        serde_json::from_str(json_str).context("failed to parse introspected sampling params")?;
    let mut result = HashMap::with_capacity(raw.len());
    for (stage_id, defaults) in raw {
        let stage_id: u32 = stage_id
            .parse()
            .with_context(|| format!("invalid stage_id in introspection output: {stage_id:?}"))?;
        info!("Stage {stage_id} sampling defaults: {defaults:?}");
        result.insert(stage_id, defaults);
    }
    Ok(result)
}
