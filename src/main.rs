//! vllm-omni-rs: Rust HTTP frontend for vllm-omni.
//!
//! Architecture:
//!   Rust (this binary):
//!     - HTTP server (axum)
//!     - OmniMasterServer (ZMQ registration for headless stages)
//!     - 2-stage routing (talker -> code2wav)
//!     - Audio encoding (WAV/PCM)
//!   Python (headless subprocesses):
//!     - Stage engine cores (model inference only)

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod engine;
mod master;
mod routes;
mod server;
mod stages;

use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tracing::{error, info};

use crate::engine::OmniEngine;
use crate::master::{StageAllocation, run_master_server};
use crate::server::{AppState, build_router};
use crate::stages::{StageSpawnConfig, spawn_stages, shutdown_stages};

#[derive(Parser)]
#[command(name = "vllm-omni-rs", about = "Rust HTTP frontend for vllm-omni")]
struct Cli {
    /// Model name or path.
    model: String,

    /// Host to bind to.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to bind to.
    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// Additional engine kwargs as JSON object.
    #[arg(long)]
    engine_kwargs: Option<String>,

    /// Use native ZMQ mode (Rust master, headless Python stages).
    /// When disabled, falls back to PyO3 FFI mode.
    #[arg(long)]
    native: bool,
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

    if cli.native {
        run_native(cli)
    } else {
        run_pyo3(cli)
    }
}

/// Native mode: Rust is the master, headless Python stages for inference only.
fn run_native(cli: Cli) -> Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(std::cmp::min(num_cpus::get(), 32))
        .enable_all()
        .build()
        .context("Failed to build tokio runtime")?;

    rt.block_on(async move {
        // 1. Allocate master port
        let master_host = "127.0.0.1".to_string();
        let master_port = vllm_managed_engine::allocate_handshake_port(&master_host)
            .context("Failed to allocate master port")?;
        info!("Master port allocated: {master_port}");

        // 2. Start master server (waits for stage registrations)
        let stage_ids = vec![0, 1]; // Qwen3 TTS: stage 0 (talker) + stage 1 (code2wav)
        let (alloc_tx, alloc_rx) = oneshot::channel::<Vec<StageAllocation>>();
        let master_host_clone = master_host.clone();
        tokio::spawn(async move {
            run_master_server(&master_host_clone, master_port, stage_ids, alloc_tx).await;
        });

        // 3. Spawn headless stages
        let extra_args: Vec<String> = if let Some(ref json_str) = cli.engine_kwargs {
            // Parse JSON and convert to CLI args
            let extra: serde_json::Map<String, serde_json::Value> =
                serde_json::from_str(json_str)
                    .context("--engine-kwargs must be valid JSON")?;
            extra.into_iter().flat_map(|(k, v)| {
                let flag = format!("--{}", k.replace('_', "-"));
                vec![flag, v.to_string().trim_matches('"').to_string()]
            }).collect()
        } else {
            vec![]
        };

        let spawn_config = StageSpawnConfig {
            model: cli.model.clone(),
            master_host: master_host.clone(),
            master_port,
            stage_ids: vec![0, 1],
            extra_args,
        };
        let mut stage_processes = spawn_stages(&spawn_config)
            .await
            .context("Failed to spawn headless stages")?;

        // 4. Wait for all stages to register
        info!("Waiting for stages to register...");
        let allocations = alloc_rx.await
            .context("Master server channel closed before all stages registered")?;
        info!("All stages registered: {:?}", allocations.iter().map(|a| a.stage_id).collect::<Vec<_>>());

        // TODO: Connect vllm-engine-core-client to each stage's ZMQ addresses
        // TODO: Implement 2-stage routing
        // TODO: Start HTTP server with ZMQ-based request handling

        // For now, just report success and keep running
        info!("Native mode: stages connected. HTTP server not yet wired to ZMQ.");

        // Start HTTP server (placeholder -- still uses engine for now)
        // This will be replaced with ZMQ-based routing
        let addr = format!("{}:{}", cli.host, cli.port);
        let listener = TcpListener::bind(&addr).await
            .context(format!("Failed to bind to {addr}"))?;
        info!("Listening on {addr}");

        // Wait for shutdown signal
        shutdown_signal().await;

        info!("Shutting down stages...");
        shutdown_stages(&mut stage_processes).await;
        info!("Goodbye.");
        Ok(())
    })
}

/// PyO3 mode: Rust calls into AsyncOmni via FFI (current working implementation).
fn run_pyo3(cli: Cli) -> Result<()> {
    pyo3::prepare_freethreaded_python();
    engine::init_python_event_loop().context("Failed to init Python event loop")?;

    let extra_kwargs: serde_json::Map<String, serde_json::Value> =
        if let Some(ref json_str) = cli.engine_kwargs {
            serde_json::from_str(json_str)
                .context("--engine-kwargs must be a valid JSON object")?
        } else {
            serde_json::Map::new()
        };

    let engine =
        Arc::new(OmniEngine::new(&cli.model, &extra_kwargs).context("Failed to create engine")?);

    let _pg_guard = ProcessGroupGuard;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(std::cmp::min(num_cpus::get(), 32))
        .enable_all()
        .build()
        .context("Failed to build tokio runtime")?;

    rt.block_on(async move {
        let state = AppState {
            engine: Arc::clone(&engine),
            model_name: cli.model.clone(),
        };

        let app = build_router(state);
        let addr = format!("{}:{}", cli.host, cli.port);
        let listener = TcpListener::bind(&addr)
            .await
            .context(format!("Failed to bind to {addr}"))?;

        info!("Listening on {addr}");

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .context("Server error")?;

        info!("Shutting down engine...");
        let engine_ref = engine;
        tokio::task::spawn_blocking(move || {
            if let Err(e) = engine_ref.shutdown() {
                error!("Engine shutdown error: {e:#}");
            }
        })
        .await?;

        info!("Goodbye.");
        Ok(())
    })
}

struct ProcessGroupGuard;

impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            libc::kill(0, libc::SIGTERM);
        }
    }
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
