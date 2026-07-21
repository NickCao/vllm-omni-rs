//! Rust implementation of OmniMasterServer.
//!
//! Handles registration from headless Python stage engine cores,
//! allocates ZMQ addresses, and provides them to the Rust-side
//! vllm-engine-core-client for direct communication.

use std::collections::HashMap;
use std::net::TcpListener;

use anyhow::{Context, Result};
use rmp_serde;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;
use tracing::{error, info};

/// Allocated addresses for one stage replica.
#[derive(Debug, Clone)]
pub struct StageAllocation {
    pub stage_id: u32,
    pub replica_id: u32,
    pub handshake_address: String,
    pub input_address: String,
    pub output_address: String,
}

/// Registration request from a headless stage.
#[derive(Debug, Deserialize)]
struct RegistrationRequest {
    stage_id: u32,
    #[serde(default)]
    replica_id: i32,
    #[serde(default)]
    stage_config: Option<rmpv::Value>,
    #[serde(default)]
    replica_bind_address: Option<String>,
}

/// Registration response sent back to the headless stage.
#[derive(Debug, Serialize)]
struct RegistrationResponse {
    handshake_address: String,
    input_address: String,
    output_address: String,
    replica_id: u32,
    coordinator_router_address: Option<String>,
}

/// Allocate N unique open TCP ports on the given host.
fn allocate_ports(host: &str, count: usize, exclude: &mut std::collections::HashSet<u16>) -> Result<Vec<u16>> {
    let mut ports = Vec::with_capacity(count);
    for _ in 0..count {
        loop {
            let listener = TcpListener::bind((host, 0))
                .context("failed to allocate port")?;
            let port = listener.local_addr()?.port();
            if exclude.insert(port) {
                ports.push(port);
                break;
            }
        }
    }
    Ok(ports)
}

/// Run the OmniMasterServer: listen for stage registrations on a ZMQ ROUTER
/// socket, allocate addresses, and reply.
///
/// Returns when all expected stages have registered.
pub async fn run_master_server(
    bind_host: &str,
    bind_port: u16,
    expected_stages: Vec<u32>,
    result_tx: oneshot::Sender<Vec<StageAllocation>>,
) {
    let addr = format!("tcp://{}:{}", bind_host, bind_port);
    info!("OmniMasterServer listening on {addr}");

    let result = tokio::task::spawn_blocking(move || -> Result<Vec<StageAllocation>> {
        let ctx = zmq::Context::new();
        let socket = ctx.socket(zmq::ROUTER)
            .context("failed to create ROUTER socket")?;
        socket.bind(&addr)
            .context(format!("failed to bind to {addr}"))?;

        let mut allocations: HashMap<u32, StageAllocation> = HashMap::new();
        let mut used_ports: std::collections::HashSet<u16> = std::collections::HashSet::new();
        used_ports.insert(bind_port);

        let bind_host_str = addr.split("://").nth(1)
            .and_then(|s| s.split(':').next())
            .unwrap_or("127.0.0.1");

        while allocations.len() < expected_stages.len() {
            let frames = socket.recv_multipart(0)
                .context("recv_multipart failed")?;
            if frames.len() < 2 {
                continue;
            }

            let identity = &frames[0];
            let msg_bytes = &frames[frames.len() - 1];

            let req: RegistrationRequest = match rmp_serde::from_slice(msg_bytes) {
                Ok(r) => r,
                Err(e) => {
                    error!("Failed to decode registration: {e}");
                    continue;
                }
            };

            if !expected_stages.contains(&req.stage_id) {
                error!("Unexpected stage_id: {}", req.stage_id);
                continue;
            }

            let replica_id = if req.replica_id < 0 { 0u32 } else { req.replica_id as u32 };

            // Allocate 3 ports: handshake, input, output
            let ports = allocate_ports(bind_host_str, 3, &mut used_ports)
                .context("failed to allocate ports")?;

            let alloc = StageAllocation {
                stage_id: req.stage_id,
                replica_id,
                handshake_address: format!("tcp://{}:{}", bind_host_str, ports[0]),
                input_address: format!("tcp://{}:{}", bind_host_str, ports[1]),
                output_address: format!("tcp://{}:{}", bind_host_str, ports[2]),
            };

            let response = RegistrationResponse {
                handshake_address: alloc.handshake_address.clone(),
                input_address: alloc.input_address.clone(),
                output_address: alloc.output_address.clone(),
                replica_id,
                coordinator_router_address: None,
            };

            let response_bytes = rmp_serde::to_vec_named(&response)
                .context("failed to encode response")?;
            socket.send_multipart(&[identity.as_slice(), &response_bytes], 0)
                .context("send_multipart failed")?;

            info!(
                "Stage {} replica {} registered: handshake={}, input={}, output={}",
                req.stage_id, replica_id,
                alloc.handshake_address, alloc.input_address, alloc.output_address,
            );

            allocations.insert(req.stage_id, alloc);
        }

        drop(socket);
        drop(ctx);

        let mut result: Vec<StageAllocation> = allocations.into_values().collect();
        result.sort_by_key(|a| a.stage_id);
        Ok(result)
    })
    .await;

    match result {
        Ok(Ok(allocs)) => {
            info!("All {} stages registered", allocs.len());
            let _ = result_tx.send(allocs);
        }
        Ok(Err(e)) => error!("Master server error: {e:#}"),
        Err(e) => error!("Master server task panicked: {e}"),
    }
}
