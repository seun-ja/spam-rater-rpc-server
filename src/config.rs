use serde::Deserialize;

#[derive(Deserialize)]
pub struct Config {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_port: Option<u16>,
    #[serde(default)]
    pub rust_log: Option<String>,
    pub environment: Option<String>,
    pub model: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server_port: Some(5500),
            rust_log: Some("info".to_string()),
            environment: Some("production".to_string()),
            model: Some("huggingface-pytorch-inference-2026-04-21-23-03-14-915".to_string()),
        }
    }
}

impl Config {
    pub fn init() -> Self {
        let config = envy::from_env::<Self>()
            .map_err(|e| eprint!("Failed to load config: {}", e))
            .unwrap_or_default();

        if config.environment.as_deref().unwrap_or_default() != "production" {
            eprint!("Environment not set to production, defaulting to production");
        }

        config
    }
}
