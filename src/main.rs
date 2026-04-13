use rpc_agent::Providers;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv::dotenv().ok();

    let model = std::env::var("MODEL").unwrap_or_else(|_| "llama3.2:latest".to_string());
    let port: u16 = std::env::var("RPC_CLIENT_PORT")
        .unwrap_or_else(|_| "5500".to_string())
        .parse()
        .expect("PORT must be a number");

    let rust_log = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(rust_log))
        .init();

    let builder = rpc_agent::AgentServerBuilder::new(
        port,
        Providers::Ollama,
        "You are a email spam classifier, you would be given an email, sender and its subject and you would give a spam score from 0 to 1.
        You would respond in json format.

        here is an example response:
        {
            \"score\": 0.2,
            \"reason\": \"looks legitimate\"
        }",
        &model, //
    );

    let server = builder.build()?;

    server.run().await?;

    Ok(())
}
