//! vllm-omni-rs: Pure Rust HTTP frontend for vllm-omni.
//!
//! Architecture:
//!   Rust (this binary):
//!     - HTTP server (axum): POST /v1/audio/speech, POST /v1/chat/completions
//!     - OmniMasterServer (ZMQ registration)
//!     - EngineCoreClient per stage (ZMQ communication)
//!     - Topology-driven routing across however many stages the pipeline
//!       declares (e.g. Qwen3-TTS: talker -> code2wav; Qwen3-Omni:
//!       thinker -> talker -> code2wav)
//!     - Chat template rendering (minijinja) and audio encoding (WAV/PCM)
//!     - Tokenizer (vllm-tokenizer)
//!   Python (headless subprocesses, inference only): one process per stage,
//!   spawned with the stage IDs vllm-omni's own pipeline config declares.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod audio;
mod chat_template;
mod introspect;
mod master;
mod routes;
mod routing;
mod server;
mod stages;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use tokio::net::TcpListener;
use tracing::{info, warn};
use vllm_managed_engine::allocate_handshake_port;
use vllm_tokenizer::HuggingFaceTokenizer;

use crate::chat_template::ChatTemplateRenderer;
use crate::introspect::{extract_chat_template, extract_tokenizer, introspect_pipeline_topology};
use crate::master::start_and_connect_stages;
use crate::routing::PipelineRouter;
use crate::stages::{StageSpawnConfig, shutdown_stages, spawn_stages};

#[derive(Parser)]
#[command(name = "vllm-omni-rs", about = "Rust HTTP frontend for vllm-omni TTS")]
struct Cli {
    /// Model name or path.
    model: String,

    /// Host to bind to.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to bind to.
    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// Master bind host for ZMQ registration.
    #[arg(long, default_value = "127.0.0.1")]
    master_host: String,

    /// Timeout for engine handshake in seconds.
    #[arg(long, default_value_t = 300)]
    handshake_timeout: u64,

    /// Path to tokenizer.json for prompt length estimation.
    #[arg(long)]
    tokenizer_path: Option<String>,

    /// Extra CLI args passed to headless Python stages (e.g.
    /// --stage-args --gpu-memory-utilization 0.5).
    #[arg(long, num_args = 1.., allow_hyphen_values = true)]
    stage_args: Vec<String>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    info!(
        "vllm-omni-rs starting for model {} on {}:{}",
        cli.model, cli.host, cli.port
    );

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(std::cmp::min(num_cpus::get(), 32))
        .enable_all()
        .build()
        .context("Failed to build tokio runtime")?;

    rt.block_on(async move {
        // 1. Introspect the pipeline topology (stage count, roles, sampling
        // defaults) before spawning anything, since that's what determines
        // which stage IDs to spawn -- not a hardcoded list.
        let topology = introspect_pipeline_topology(&cli.model)
            .context("Failed to introspect pipeline topology")?;
        let mut stage_ids: Vec<u32> = topology.keys().copied().collect();
        stage_ids.sort_unstable();
        info!("Pipeline stages for {}: {:?}", cli.model, stage_ids);

        // 2. Allocate master registration port
        let master_port =
            allocate_handshake_port(&cli.master_host).context("Failed to allocate master port")?;
        info!("Master registration port: {master_port}");

        // 3. Spawn headless Python stages
        let spawn_config = StageSpawnConfig {
            model: cli.model.clone(),
            master_host: cli.master_host.clone(),
            master_port,
            stage_ids: stage_ids.clone(),
            extra_args: cli.stage_args.clone(),
        };
        let mut stage_processes = spawn_stages(&spawn_config)
            .await
            .context("Failed to spawn headless stages")?;

        // 4. Accept registrations + perform handshake with each engine core
        let timeout = Duration::from_secs(cli.handshake_timeout);
        let connected_stages = start_and_connect_stages(
            &cli.master_host,
            master_port,
            stage_ids,
            &cli.model,
            timeout,
        )
        .await
        .context("Failed to connect to stage engine cores")?;

        info!(
            "All {} stages connected. Starting HTTP server.",
            connected_stages.len()
        );

        // 5. Load the tokenizer and build the router
        let tokenizer_path = match cli.tokenizer_path {
            Some(path) => path,
            None => extract_tokenizer(&cli.model)
                .context("No --tokenizer-path given and auto-extraction failed")?,
        };
        let tokenizer = Arc::new(
            HuggingFaceTokenizer::new(std::path::Path::new(&tokenizer_path))
                .with_context(|| format!("Failed to load tokenizer from {tokenizer_path}"))?,
        );
        info!("Tokenizer loaded from {tokenizer_path}");
        let router = Arc::new(
            PipelineRouter::new(connected_stages, Arc::clone(&tokenizer), &topology)
                .context("Failed to create pipeline router")?,
        );

        // 6. If the pipeline has a text-output stage, try to load a chat
        // template for it. A missing template isn't a startup failure --
        // /v1/chat/completions just returns 501 for this model.
        let has_text_stage = topology.values().any(|t| t.final_output_type.as_deref() == Some("text"));
        let chat_template = if has_text_stage {
            match extract_chat_template(&cli.model).context("Failed to introspect chat template")? {
                Some(info) => Some(Arc::new(ChatTemplateRenderer::new(info.template, info.bos_token, info.eos_token)?)),
                None => {
                    warn!("Model has a text-output stage but no chat template was found; /v1/chat/completions will 501");
                    None
                }
            }
        } else {
            None
        };

        // 7. Start HTTP server
        let state = server::AppState {
            model_name: cli.model.clone(),
            router: Arc::clone(&router),
            tokenizer,
            chat_template,
        };
        let app = server::build_router(state);

        let addr = format!("{}:{}", cli.host, cli.port);
        let listener = TcpListener::bind(&addr)
            .await
            .context(format!("Failed to bind to {addr}"))?;
        info!("Listening on {addr}");

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .context("Server error")?;

        // 8. Cleanup
        info!("Shutting down...");
        if let Ok(r) = Arc::try_unwrap(router) {
            let _ = r.shutdown();
        }
        shutdown_stages(&mut stage_processes).await;
        info!("Goodbye.");
        Ok(())
    })
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => info!("Received Ctrl+C"),
        () = terminate => info!("Received SIGTERM"),
    }
}
