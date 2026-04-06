use rpc_agent::Providers;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let builder = rpc_agent::AgentServerBuilder::new(
        5500,
        Providers::Ollama,
        "You are a email spam classifier, you would be given an email, sender and its subject and you would give a spam score from 0 to 1.
        You would respond in json format.

        here is an example response:
        {
            \"score\": 0.2,
            \"reason\": \"looks legitimate\"
        }",
        "gpt-oss:20b",
    );

    let server = builder.build()?;

    server.run().await?;

    Ok(())
}
