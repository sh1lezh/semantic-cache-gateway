use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use qdrant_client::{
    qdrant::{
        CreateCollectionBuilder, Distance, PointStruct, SearchPointsBuilder, UpsertPointsBuilder,
        VectorParamsBuilder,
    },
    Payload, Qdrant,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::net::TcpListener;
use uuid::Uuid;

const COLLECTION_NAME: &str = "semantic-cache";
const EMBEDDING_DIM: usize = 384;
const GROQ_MODEL: &str = "llama-3.1-8b-instant";

#[derive(Deserialize, Debug)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
}

#[derive(Deserialize, Debug)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: String,
    choices: Vec<Choice>,
}

#[derive(Serialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Serialize)]
struct ResponseMessage {
    role: String,
    content: String,
}

struct AppState {
    model_session: Mutex<Session>,
    tokenizer: Tokenizer,
    db_client: Qdrant,
    llm_api_key: String,
    cache_score_threshold: f32,
    metrics: Mutex<MetricsData>,
}

#[derive(Default)]
struct MetricsData {
    total_requests: u64,
    cache_hits: u64,
    cache_misses: u64,
    total_hit_latency_ms: f64,
    total_miss_latency_ms: f64,
}

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

#[derive(Debug)]
struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = Json(serde_json::json!({
            "error": {
                "message": self.message,
                "type": "gateway_error",
                "code": self.status.as_u16()
            }
        }));
        (self.status, body).into_response()
    }
}

impl From<reqwest::Error> for AppError {
    fn from(e: reqwest::Error) -> Self {
        AppError::internal(format!("HTTP client error: {e}"))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    let llm_api_key = std::env::var("API_KEY")
        .expect("API_KEY must be set in .env or environment.");

    // qdrant-client uses the gRPC endpoint. Expose 6334 from Docker.
    let qdrant_url = std::env::var("QDRANT_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:6334".to_string());

    let cache_score_threshold = std::env::var("CACHE_SCORE_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.90);

    println!("1. Initializing Machine Learning Engine...");

    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("model/model.onnx")
        .expect("Failed to load model/model.onnx from disk.");

    let tokenizer = Tokenizer::from_file("model/tokenizer.json")
        .expect("Failed to load model/tokenizer.json from disk.");

    println!("2. Connecting to Qdrant Vector Database at {qdrant_url}...");
    let db_client = Qdrant::from_url(&qdrant_url).build()?;

    if !db_client.collection_exists(COLLECTION_NAME).await? {
        db_client
            .create_collection(
                CreateCollectionBuilder::new(COLLECTION_NAME).vectors_config(
                    VectorParamsBuilder::new(EMBEDDING_DIM as u64, Distance::Cosine),
                ),
            )
            .await?;
    }

    let shared_state = Arc::new(AppState {
        model_session: Mutex::new(session),
        tokenizer,
        db_client,
        llm_api_key,
        cache_score_threshold,
        metrics: Mutex::new(MetricsData::default()),
    });

    let app = Router::new()
        .route("/health", get(handle_health))
        .route("/v1/chat/completions", post(handle_chat))
        .route("/metrics", get(handle_metrics))
        .with_state(shared_state);

    let app_host = std::env::var("APP_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let app_port = std::env::var("APP_PORT").unwrap_or_else(|_| "3000".to_string());
    let bind_addr = format!("{app_host}:{app_port}");

    let listener = TcpListener::bind(&bind_addr).await?;
    println!("Semantic Gateway listening on http://{bind_addr}");
    println!("Cache score threshold: {cache_score_threshold:.2}\n");

    axum::serve(listener, app).await?;

    Ok(())
}

async fn handle_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn handle_chat(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, AppError> {
    let requested_model = payload.model.trim();

    if requested_model.is_empty() {
        return Err(AppError::bad_request("model is required and must not be empty"));
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
        return Err(AppError::bad_request("last message content must not be empty"));
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
            .ok_or_else(|| AppError::internal("Qdrant hit did not contain a string response payload"))?;

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

fn embed_prompt(state: &AppState, user_prompt: &str) -> Result<Vec<f32>, AppError> {
    let encoding = state
        .tokenizer
        .encode(user_prompt.to_string(), true)
        .map_err(|e| AppError::internal(format!("Tokenizer failed: {e}")))?;

    let input_ids = encoding.get_ids();
    let attention_mask = encoding.get_attention_mask();
    let token_type_ids = encoding.get_type_ids();
    let seq_len = input_ids.len();

    if seq_len == 0 {
        return Err(AppError::bad_request("prompt produced zero tokens"));
    }

    let input_ids_2d = Array2::from_shape_vec(
        (1, seq_len),
        input_ids.iter().map(|&x| x as i64).collect::<Vec<_>>(),
    )
    .map_err(|e| AppError::internal(format!("input_ids tensor shape error: {e}")))?;

    let attention_mask_2d = Array2::from_shape_vec(
        (1, seq_len),
        attention_mask.iter().map(|&x| x as i64).collect::<Vec<_>>(),
    )
    .map_err(|e| AppError::internal(format!("attention_mask tensor shape error: {e}")))?;

    let token_type_ids_2d = Array2::from_shape_vec(
        (1, seq_len),
        token_type_ids.iter().map(|&x| x as i64).collect::<Vec<_>>(),
    )
    .map_err(|e| AppError::internal(format!("token_type_ids tensor shape error: {e}")))?;

    let input_ids_val = Value::from_array(input_ids_2d)
        .map_err(|e| AppError::internal(format!("input_ids tensor creation failed: {e}")))?;
    let attention_mask_val = Value::from_array(attention_mask_2d)
        .map_err(|e| AppError::internal(format!("attention_mask tensor creation failed: {e}")))?;
    let token_type_ids_val = Value::from_array(token_type_ids_2d)
        .map_err(|e| AppError::internal(format!("token_type_ids tensor creation failed: {e}")))?;

    let mut session_guard = state
        .model_session
        .lock()
        .map_err(|e| AppError::internal(format!("Session lock poisoned: {e}")))?;

    let outputs = session_guard
        .run(ort::inputs![
            "input_ids" => input_ids_val,
            "attention_mask" => attention_mask_val,
            "token_type_ids" => token_type_ids_val,
        ])
        .map_err(|e| AppError::internal(format!("ONNX inference failed: {e}")))?;

    let (shape, data) = outputs["last_hidden_state"]
        .try_extract_tensor::<f32>()
        .map_err(|e| AppError::internal(format!("Failed to extract last_hidden_state tensor: {e}")))?;

    let hidden_dim = shape
        .last()
        .copied()
        .map(|v| v as usize)
        .unwrap_or(EMBEDDING_DIM);

    if hidden_dim != EMBEDDING_DIM {
        return Err(AppError::internal(format!(
            "Embedding dimension mismatch: model produced {hidden_dim}, Qdrant collection expects {EMBEDDING_DIM}"
        )));
    }

    let expected_len = seq_len * EMBEDDING_DIM;
    if data.len() < expected_len {
        return Err(AppError::internal(format!(
            "ONNX output too small: got {} values, expected at least {expected_len}",
            data.len()
        )));
    }

    let mut mean_vector = vec![0.0f32; EMBEDDING_DIM];
    let mut real_token_count = 0f32;

    for token_idx in 0..seq_len {
        if attention_mask[token_idx] == 1 {
            real_token_count += 1.0;
            let start = token_idx * EMBEDDING_DIM;
            for dim_idx in 0..EMBEDDING_DIM {
                mean_vector[dim_idx] += data[start + dim_idx];
            }
        }
    }

    if real_token_count == 0.0 {
        return Err(AppError::bad_request("prompt contained no real tokens after masking"));
    }

    for value in &mut mean_vector {
        *value /= real_token_count;
    }

    l2_normalize(&mut mean_vector);
    Ok(mean_vector)
}

fn l2_normalize(vector: &mut [f32]) {
    let norm = vector
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();

    if norm > 0.0 {
        for value in vector {
            *value = (*value as f64 / norm) as f32;
        }
    }
}

async fn llm_call(prompt: &str, model: &str, api_key: &str) -> Result<String, AppError> {
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

fn update_hit_metrics(state: &AppState, latency_ms: f64) {
    if let Ok(mut m) = state.metrics.lock() {
        m.total_requests += 1;
        m.cache_hits += 1;
        m.total_hit_latency_ms += latency_ms;
    }
}

fn update_miss_metrics(state: &AppState, latency_ms: f64) {
    if let Ok(mut m) = state.metrics.lock() {
        m.total_requests += 1;
        m.cache_misses += 1;
        m.total_miss_latency_ms += latency_ms;
    }
}

async fn handle_metrics(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
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
