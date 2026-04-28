use rpc_agent::{AgentServer, Providers};
use tracing_subscriber::{EnvFilter, fmt::format::FmtSpan};

mod config;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();

    let config = config::Config::init();
    let env = config.environment.as_deref().unwrap_or_default();
    let model = config.model.as_deref().unwrap_or_default();
    let port = config.rpc_port.unwrap_or_default();
    let rust_log = config.rust_log.as_deref().unwrap_or_default();

    let (model, provider) = if env == "production" {
        (model, Providers::CustomSageMakerAI)
    } else {
        ("./merged-model-new", Providers::LocalInference)
    };

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(rust_log))
        .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
        .pretty()
        .init();

    let builder = rpc_agent::AgentServerBuilder::new(
        port,
        provider,
        "You are a email spam classifier, you would be given an email, sender and its subject and you would give a spam score from 0 to 1.
        You would respond in json format.

        here is an example response:
        {
            \"score\": 0.2,
            \"reason\": \"looks legitimate\"
        }",
        model,
    );

    let server = if env == "production" {
        builder.build().await?
    } else {
        builder
            .function_handler("predict".to_string())
            .script_name("local_inference".to_string())
            .perm_file("spam-rater-private_key.pem")
            .build()
            .await?
    };

    call_agent(server).await
}

#[tracing::instrument(name = "rpc.caller", skip(server))]
async fn call_agent(server: AgentServer) -> Result<(), Box<dyn std::error::Error>> {
    server.run().await?;

    Ok(())
}
