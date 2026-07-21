//! Spawn and manage headless Python stage engine cores.
//!
//! Each stage runs as a separate Python process:
//!   vllm serve <model> --omni --headless --stage-id N \
//!     --omni-master-address <host> --omni-master-port <port>

use std::process::Stdio;

use anyhow::{Context, Result};
use tokio::process::{Child, Command};
use tracing::info;

/// Configuration for spawning headless stage processes.
pub struct StageSpawnConfig {
    pub model: String,
    pub master_host: String,
    pub master_port: u16,
    pub stage_ids: Vec<u32>,
    pub extra_args: Vec<String>,
}

/// Handle to a running headless stage process.
pub struct StageProcess {
    pub stage_id: u32,
    child: Child,
}

impl StageProcess {
    pub async fn kill(&mut self) {
        let _ = self.child.kill().await;
    }

    pub fn id(&self) -> Option<u32> {
        self.child.id()
    }
}

/// Spawn headless Python processes for each stage.
pub async fn spawn_stages(config: &StageSpawnConfig) -> Result<Vec<StageProcess>> {
    let mut processes = Vec::new();

    for &stage_id in &config.stage_ids {
        let mut cmd = Command::new("python3");
        cmd.arg("-m")
            .arg("vllm_omni.entrypoints.cli.main")
            .arg("serve")
            .arg(&config.model)
            .arg("--omni")
            .arg("--headless")
            .arg("--stage-id")
            .arg(stage_id.to_string())
            .arg("--omni-master-address")
            .arg(&config.master_host)
            .arg("--omni-master-port")
            .arg(config.master_port.to_string());

        cmd.arg("--async-chunk");

        for arg in &config.extra_args {
            cmd.arg(arg);
        }

        cmd.stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());

        // Put each stage in its own process group for clean shutdown.
        unsafe {
            cmd.pre_exec(|| {
                if libc::setpgid(0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }

        let child = cmd
            .spawn()
            .context(format!("failed to spawn headless stage {stage_id}"))?;

        info!(
            "Spawned headless stage {} (pid={})",
            stage_id,
            child.id().unwrap_or(0)
        );

        processes.push(StageProcess { stage_id, child });
    }

    Ok(processes)
}

/// Kill all stage processes.
pub async fn shutdown_stages(stages: &mut [StageProcess]) {
    for stage in stages.iter_mut() {
        if let Some(pid) = stage.id() {
            info!("Killing stage {} (pid={})", stage.stage_id, pid);
            // Kill the process group
            unsafe {
                libc::kill(-(pid as i32), libc::SIGTERM);
            }
        }
    }
    // Wait briefly for graceful shutdown
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    for stage in stages.iter_mut() {
        stage.kill().await;
    }
}
