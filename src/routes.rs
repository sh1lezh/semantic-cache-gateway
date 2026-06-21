use crate::constants::{COLLECTION_NAME, GROQ_MODEL};
use crate::dto::{ChatRequest, ChatResponse, Choice, ResponseMessage};
use crate::embedding::embed_prompt;
use crate::error::AppError;
use crate::llm::llm_call;
use crate::metrics::{update_hit_metrics, update_miss_metrics};
use crate::state::AppState;
use axum::{Json, extract::State};
use qdrant_client::{
    Payload,
    qdrant::{PointStruct, SearchPointsBuilder, UpsertPointsBuilder},
};
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

pub async fn handle_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

pub async fn handle_chat(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, AppError> {
    let requested_model = payload.model.trim();

    if requested_model.is_empty() {
        return Err(AppError::bad_request(
            "model is required and must not be empty",
        ));
    }

    if requested_model != GROQ_MODEL {
        return Err(AppError::bad_request(format!(
            "unsupported model '{requested_model}'. This gateway is configured for '{GROQ_MODEL}'"
        )));
    }

    let last_message = payload
        .messages
        .last()
        .ok_or_else(|| AppError::bad_request("messages must contain at least one item"))?;

    if last_message.role.trim() != "user" {
        return Err(AppError::bad_request("last message role must be 'user'"));
    }

    let request_start = Instant::now();
    let user_prompt = last_message.content.trim().to_string();

    if user_prompt.is_empty() {
        return Err(AppError::bad_request(
            "last message content must not be empty",
        ));
    }

    println!("--- Incoming request: '{user_prompt}' ---");

    let mean_vector = embed_prompt(&state, &user_prompt)?;

    let search_result = state
        .db_client
        .search_points(
            SearchPointsBuilder::new(COLLECTION_NAME, mean_vector.clone(), 1)
                .with_payload(true)
                .score_threshold(state.cache_score_threshold),
        )
        .await
        .map_err(|e| AppError::internal(format!("Qdrant search failed: {e}")))?;

    let final_response_text = if let Some(hit) = search_result.result.first() {
        let cached_resp = hit
            .payload
            .get("response")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                AppError::internal("Qdrant hit did not contain a string response payload")
            })?;

        let latency = request_start.elapsed().as_secs_f64() * 1000.0;
        update_hit_metrics(&state, latency);
        println!(
            ">> CACHE HIT! Similarity score: {:.4}, latency: {:.2} ms",
            hit.score, latency
        );

        format!("[CACHED] {cached_resp}")
    } else {
        println!(">> CACHE MISS! Calling LLM...");
        let llm_response = llm_call(&user_prompt, requested_model, &state.llm_api_key).await?;

        let mut payload = Payload::new();
        payload.insert("prompt", user_prompt.clone());
        payload.insert("response", llm_response.clone());

        let point = PointStruct::new(Uuid::new_v4().to_string(), mean_vector, payload);
        state
            .db_client
            .upsert_points(UpsertPointsBuilder::new(COLLECTION_NAME, vec![point]))
            .await
            .map_err(|e| AppError::internal(format!("Qdrant upsert failed: {e}")))?;

        let latency = request_start.elapsed().as_secs_f64() * 1000.0;
        update_miss_metrics(&state, latency);
        println!(">> CACHE MISS completed. End-to-end latency: {latency:.2} ms");

        llm_response
    };

    Ok(Json(ChatResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".to_string(),
        choices: vec![Choice {
            message: ResponseMessage {
                role: "assistant".to_string(),
                content: final_response_text,
            },
        }],
    }))
}

pub async fn handle_metrics(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let m = state.metrics.lock().unwrap_or_else(|e| e.into_inner());

    let hit_rate = if m.total_requests > 0 {
        (m.cache_hits as f64 / m.total_requests as f64) * 100.0
    } else {
        0.0
    };

    let avg_hit_ms = if m.cache_hits > 0 {
        m.total_hit_latency_ms / m.cache_hits as f64
    } else {
        0.0
    };

    let avg_miss_ms = if m.cache_misses > 0 {
        m.total_miss_latency_ms / m.cache_misses as f64
    } else {
        0.0
    };

    let speedup_factor = if avg_hit_ms > 0.0 && avg_miss_ms > 0.0 {
        format!("{:.2}x", avg_miss_ms / avg_hit_ms)
    } else {
        "N/A".to_string()
    };

    let estimated_tokens_saved = m.cache_hits * 500;
    let estimated_cost_saved_usd = estimated_tokens_saved as f64 * 0.000002;

    Json(serde_json::json!({
        "total_requests": m.total_requests,
        "cache_hits": m.cache_hits,
        "cache_misses": m.cache_misses,
        "hit_rate_percent": format!("{hit_rate:.1}"),
        "avg_hit_latency_ms": format!("{avg_hit_ms:.2}"),
        "avg_miss_latency_ms": format!("{avg_miss_ms:.2}"),
        "speedup_factor": speedup_factor,
        "cache_score_threshold": state.cache_score_threshold,
        "estimated_tokens_saved": estimated_tokens_saved,
        "estimated_cost_saved_usd": format!("{estimated_cost_saved_usd:.6}")
    }))
}
