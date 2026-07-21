//! vllm-omni-rs: Pure Rust HTTP frontend for vllm-omni TTS.
//!
//! Architecture:
//!   Rust (this binary):
//!     - HTTP server (axum)
//!     - OmniMasterServer (ZMQ registration)
//!     - EngineCoreClient per stage (ZMQ communication)
//!     - 2-stage routing (talker -> code2wav)
//!     - Audio encoding (WAV/PCM)
//!     - Tokenizer (vllm-tokenizer)
//!   Python (headless subprocesses, inference only):
//!     - Stage 0: Talker (autoregressive codec generation)
//!     - Stage 1: Code2Wav (vocoder)

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod master;
mod routes;
mod server;
mod stages;

use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use tokio::net::TcpListener;
use tracing::info;
use vllm_managed_engine::allocate_handshake_port;

use crate::master::start_and_connect_stages;
use crate::stages::{StageSpawnConfig, shutdown_stages, spawn_stages};

#[derive(Parser)]
#[command(
    name = "vllm-omni-rs",
    about = "Rust HTTP frontend for vllm-omni TTS"
)]
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

    /// Extra CLI args passed to headless Python stages.
    #[arg(long, num_args = 1..)]
    stage_args: Vec<String>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
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
        // 1. Allocate master registration port
        let master_port = allocate_handshake_port(&cli.master_host)
            .context("Failed to allocate master port")?;
        info!("Master registration port: {master_port}");

        // 2. Spawn headless Python stages
        let stage_ids = vec![0, 1]; // Qwen3 TTS: talker + code2wav
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

        // 3. Accept registrations + perform handshake with each engine core
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

        // TODO: Create a routing layer that uses connected_stages
        // TODO: Wire HTTP routes to send requests via EngineCoreClient

        // 4. Start HTTP server
        let addr = format!("{}:{}", cli.host, cli.port);
        let listener = TcpListener::bind(&addr)
            .await
            .context(format!("Failed to bind to {addr}"))?;
        info!("Listening on {addr}");

        // Wait for shutdown
        shutdown_signal().await;

        // 5. Cleanup
        info!("Shutting down stages...");
        for stage in connected_stages {
            stage.client.shutdown();
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
        tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::terminate(),
        )
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
