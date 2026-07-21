//! vllm-omni-rs: Rust HTTP frontend for vllm-omni.
//!
//! Embeds a Python interpreter via PyO3, creates an AsyncOmni instance,
//! and serves /v1/audio/speech using axum. The generate() async generator
//! is consumed directly via pyo3-async-runtimes.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod engine;
mod routes;
mod server;

use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use tokio::net::TcpListener;
use tracing::{error, info};

use crate::engine::OmniEngine;
use crate::server::{AppState, build_router};

struct ProcessGroupGuard;

impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            // Kill our entire process group (us + all Python children).
            // Negative PID = process group.
            libc::kill(0, libc::SIGTERM);
        }
    }
}

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
    /// Example: --engine-kwargs '{"stage_init_timeout": 300, "init_timeout": 600}'
    #[arg(long)]
    engine_kwargs: Option<String>,
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

    // Put this process into its own process group so all Python children
    // (StageEngineCoreProc) share our PGID and can be killed together.
    #[cfg(unix)]
    unsafe {
        libc::setpgid(0, 0);
    }

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

    // Ensure all Python children are killed if this process exits for any reason.
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
