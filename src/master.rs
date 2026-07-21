//! Rust OmniMasterServer: registration, handshake, and ZMQ connection.

use std::collections::HashMap;
use std::net::TcpListener;
use std::time::Duration;

use anyhow::{Context, Result};
use rmp_serde;
use serde::{Deserialize, Serialize};
use tracing::{error, info};
use vllm_engine_core_client::{
    EngineCoreClient, EngineCoreClientConfig, TransportMode,
};

/// Allocated addresses for one stage.
#[derive(Debug, Clone)]
pub struct StageAllocation {
    pub stage_id: u32,
    pub replica_id: u32,
    pub handshake_address: String,
    pub input_address: String,
    pub output_address: String,
}

/// A stage with its connected EngineCoreClient.
pub struct ConnectedStage {
    pub stage_id: u32,
    pub client: EngineCoreClient,
}

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

#[derive(Debug, Serialize)]
struct RegistrationResponse {
    handshake_address: String,
    input_address: String,
    output_address: String,
    replica_id: u32,
    coordinator_router_address: Option<String>,
}

fn allocate_ports(
    host: &str,
    count: usize,
    exclude: &mut std::collections::HashSet<u16>,
) -> Result<Vec<u16>> {
    let mut ports = Vec::with_capacity(count);
    for _ in 0..count {
        loop {
            let listener =
                TcpListener::bind((host, 0)).context("failed to allocate port")?;
            let port = listener.local_addr()?.port();
            if exclude.insert(port) {
                ports.push(port);
                break;
            }
        }
    }
    Ok(ports)
}

/// Run the full startup:
/// 1. Accept registrations from headless stages
/// 2. For each stage, spawn an EngineCoreClient::connect() that handles handshake
/// 3. Return connected clients for each stage
pub async fn start_and_connect_stages(
    bind_host: &str,
    bind_port: u16,
    expected_stages: Vec<u32>,
    model_name: &str,
    handshake_timeout: Duration,
) -> Result<Vec<ConnectedStage>> {
    // Phase 1: Registration (blocking ZMQ)
    let allocations =
        run_registration(bind_host, bind_port, &expected_stages).await?;

    // Phase 2: Connect EngineCoreClient to each stage
    // The engine cores connect to the handshake address after registration.
    // We use HandshakeOwner mode which binds and waits for HELLO/INIT/READY.
    let mut connected = Vec::new();
    for alloc in &allocations {
        info!(
            "Connecting to stage {} engine core (handshake={})",
            alloc.stage_id, alloc.handshake_address
        );

        let config = EngineCoreClientConfig {
            transport_mode: TransportMode::HandshakeOwner {
                handshake_address: alloc.handshake_address.clone(),
                advertised_host: bind_host.to_string(),
                engine_count: 1,
                ready_timeout: handshake_timeout,
                local_input_address: Some(alloc.input_address.clone()),
                local_output_address: Some(alloc.output_address.clone()),
            },
            coordinator_mode: None,
            model_name: model_name.to_string(),
            client_index: 0,
        };

        let client = EngineCoreClient::connect(config)
            .await
            .context(format!(
                "Failed to connect to stage {} engine core",
                alloc.stage_id
            ))?;

        info!("Stage {} engine core connected", alloc.stage_id);
        connected.push(ConnectedStage {
            stage_id: alloc.stage_id,
            client,
        });
    }

    Ok(connected)
}

async fn run_registration(
    bind_host: &str,
    bind_port: u16,
    expected_stages: &[u32],
) -> Result<Vec<StageAllocation>> {
    let addr = format!("tcp://{}:{}", bind_host, bind_port);
    info!("OmniMasterServer listening on {addr}");

    let expected = expected_stages.to_vec();
    let host = bind_host.to_string();

    let allocations = tokio::task::spawn_blocking(move || -> Result<Vec<StageAllocation>> {
        let ctx = zmq::Context::new();
        let socket = ctx
            .socket(zmq::ROUTER)
            .context("failed to create ROUTER socket")?;
        socket
            .bind(&addr)
            .context(format!("failed to bind to {addr}"))?;

        let mut allocs: HashMap<u32, StageAllocation> = HashMap::new();
        let mut used_ports: std::collections::HashSet<u16> =
            std::collections::HashSet::new();
        used_ports.insert(bind_port);

        while allocs.len() < expected.len() {
            let frames = socket
                .recv_multipart(0)
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

            if !expected.contains(&req.stage_id) {
                error!("Unexpected stage_id: {}", req.stage_id);
                continue;
            }

            let replica_id =
                if req.replica_id < 0 { 0u32 } else { req.replica_id as u32 };

            let ports = allocate_ports(&host, 3, &mut used_ports)
                .context("failed to allocate ports")?;

            let alloc = StageAllocation {
                stage_id: req.stage_id,
                replica_id,
                handshake_address: format!("tcp://{}:{}", host, ports[0]),
                input_address: format!("tcp://{}:{}", host, ports[1]),
                output_address: format!("tcp://{}:{}", host, ports[2]),
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
            socket
                .send_multipart(
                    &[identity.as_slice(), &response_bytes],
                    0,
                )
                .context("send_multipart failed")?;

            info!(
                "Stage {} replica {} registered (handshake={})",
                req.stage_id, replica_id, alloc.handshake_address,
            );

            allocs.insert(req.stage_id, alloc);
        }

        drop(socket);
        drop(ctx);

        let mut result: Vec<StageAllocation> = allocs.into_values().collect();
        result.sort_by_key(|a| a.stage_id);
        Ok(result)
    })
    .await
    .context("Registration task panicked")??;

    info!("All {} stages registered", allocations.len());
    Ok(allocations)
}
