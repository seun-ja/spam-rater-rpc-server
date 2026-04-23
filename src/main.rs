use rpc_agent::{AgentServer, Providers};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();

    let env = std::env::var("ENVIRONMENT").unwrap_or_else(|_| "production".to_string());

    let model = std::env::var("MODEL")
        .unwrap_or_else(|_| "huggingface-pytorch-inference-2026-04-21-23-03-14-915".to_string());

    let port: u16 = std::env::var("RPC_CLIENT_PORT")
        .unwrap_or_else(|_| "5500".to_string())
        .parse()
        .expect("PORT must be a number");

    let rust_log = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());

    let (model, provider) = if env == "production" {
        (model.as_str(), Providers::CustomSageMakerAI)
    } else {
        ("./merged-model-new", Providers::LocalInference)
    };

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(rust_log))
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
