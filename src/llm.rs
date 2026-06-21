use crate::error::AppError;
use serde::Deserialize;

#[derive(Deserialize)]
struct LlmApiResponse {
    choices: Vec<LlmChoice>,
}

#[derive(Deserialize)]
struct LlmChoice {
    message: LlmMessage,
}

#[derive(Deserialize)]
struct LlmMessage {
    content: String,
}

pub async fn llm_call(prompt: &str, model: &str, api_key: &str) -> Result<String, AppError> {
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
    });

    let resp = client
        .post("https://api.groq.com/openai/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    let response_text = resp.text().await?;

    if !status.is_success() {
        return Err(AppError::internal(format!(
            "Groq API returned HTTP {status}: {response_text}"
        )));
    }

    let parsed: LlmApiResponse = serde_json::from_str(&response_text).map_err(|e| {
        AppError::internal(format!(
            "Failed to parse Groq response as chat completion JSON: {e}; body: {response_text}"
        ))
    })?;

    Ok(parsed
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .unwrap_or_else(|| "No response generated.".to_string()))
}
