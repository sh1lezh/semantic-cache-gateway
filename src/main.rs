use axum::{
    Json, Router, extract::State, http::StatusCode, response::{IntoResponse, Response}, routing::{get, post}
};
use ort::value::Value;
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use ort::session::{builder::GraphOptimizationLevel, Session};
use tokenizers::Tokenizer;
use ndarray::Array2;
use uuid::Uuid;
use qdrant_client::{
    Qdrant,
    Payload,
    qdrant::{CreateCollectionBuilder, Distance, PointStruct, SearchPointsBuilder, VectorParamsBuilder, UpsertPointsBuilder},
};
use std::time::Instant;

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
    metrics: Mutex<MetricsData>,
}

#[derive(Default)]
struct MetricsData {
    total_request: u64,
    cache_hits: u64,
    cache_misses: u64,
    total_hit_latency_ms: f64,
    total_miss_latency_ms: f64,
}

#[derive(Deserialize)]
struct LlmApiRespone {
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

struct AppError(String);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = Json(serde_json::json!({
            "error": {
                "message": self.0,
                "type": "gateway_error",
                "code": 500
            }
        }));
        (StatusCode::INTERNAL_SERVER_ERROR, body).into_response()
    }
}

impl From<reqwest::Error> for AppError {
    fn from(e: reqwest::Error) -> Self {
        AppError(format!("HTTP Client error: {}", e))
    }
}

#[tokio::main]
async fn main() -> Result<(),  Box<dyn std::error::Error>>{
    dotenvy::dotenv().ok();

    let llm_api_key = std::env::var("API_KEY")
        .expect("API_KEY must be set in .env or enviornment.");

    let qdrant_url = std::env::var("QDRANT_URL")
        .unwrap_or_else(|_| "http://localhost:6334".to_string());

    println!("Initializing Machine Learning Engine...");

    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("model/model.onnx")
        .expect("Failed to Load the Model ONNX from the hard drive...");

    let tokenizer = Tokenizer::from_file("model/tokenizer.json")
        .expect("Failed to load the Tokenizer from the hard drive...");
    
    println!("2. Connecting to Qdrant Vector Database...");
    let db_client = Qdrant::from_url(&qdrant_url).build()?;

    if !db_client.collection_exists("semantic-cache").await? {
        db_client
            .create_collection(
                CreateCollectionBuilder::new("semantic-cache")
                    .vectors_config(
                        VectorParamsBuilder::new(384, Distance::Cosine)
                    )
            ).await?;
    }

    let shared_state = Arc::new(AppState {
        model_session: Mutex::new(session),
        tokenizer,
        db_client,  
        llm_api_key,
        metrics: Mutex::new(MetricsData::default()),
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat))
        .route("/metrics", post(handle_metrics))
        .with_state(shared_state);

    let listener = TcpListener::bind("127.0.0.1:3000").await?;
    println!("Semantic Gateway Listening on Port 3000..... \n");

    axum::serve(listener, app).await?;

    Ok(())
}


async fn handle_chat(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, AppError> {
    
    let mut final_response_text = String::new();

    if let Some(last_message) = payload.messages.last() {

        let request_start = Instant::now();

        let user_prompt = last_message.content.clone();
        println!("--- Incoming Request: '{}' ---", user_prompt);

        let encoding = state.tokenizer.encode(user_prompt.clone(), true)
            .map_err(|e| AppError(format!("Tokenizer failed: {}", e)))?;
        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        let token_type_ids = encoding.get_type_ids();
        let seq_len = input_ids.len();

        let input_ids_2d = Array2::from_shape_vec((1, seq_len), input_ids.iter().map(|&x| x as i64).collect::<Vec<_>>()).unwrap();
        let attention_mask_2d = Array2::from_shape_vec((1, seq_len), attention_mask.iter().map(|&x| x as i64).collect::<Vec<_>>()).unwrap();
        let token_type_ids_2d = Array2::from_shape_vec((1, seq_len), token_type_ids.iter().map(|&x| x as i64).collect::<Vec<_>>()).unwrap();

        let input_ids_val = Value::from_array(input_ids_2d)
            .map_err(|e| AppError(format!("Input Tensor failed to load: {}", e)))?;
        let attention_mask_val = Value::from_array(attention_mask_2d)
            .map_err(|e| AppError(format!("Attention Tensor failed to load: {}", e)))?;
        let token_type_ids_val = Value::from_array(token_type_ids_2d)
            .map_err(|e| AppError(format!("Token Tensor failed to load: {}", e)))?;

        let mean_vector = {
            let mut session_guard = state.model_session.lock()
                .map_err(|e| AppError(format!("Session lock poisned: {}", e)))?;

            let outputs = session_guard.run(ort::inputs![
                "input_ids" => input_ids_val,
                "attention_mask" => attention_mask_val,
                "token_type_ids" => token_type_ids_val,
            ]).map_err(|e| AppError(format!("ONNX Inference failed: {}", e)))?;

            let embedding_tuple = outputs["last_hidden_state"].try_extract_tensor::<f32>().unwrap();
            let data = embedding_tuple.1;

            let hidden_dim = 384;
            let mut mean_vector = vec![0.0f32; hidden_dim];
            let mut real_token_count = 0f32;

            for token_idx in 0..seq_len {
                if attention_mask[token_idx] == 1 {
                    real_token_count += 1.0;
                    let start = token_idx * hidden_dim;
                    for dim_idx in 0..hidden_dim {
                        mean_vector[dim_idx] += data[start + dim_idx];
                    }
                }
            }
            if real_token_count > 0.0 {
                for dim_idx in 0..hidden_dim { 
                    mean_vector[dim_idx] /= real_token_count;
                }
            }   

            mean_vector
        };

        let search_result = state.db_client.search_points(SearchPointsBuilder::new(
                "semantic-cache",
                mean_vector.clone(),
                1,
            ).with_payload(true).score_threshold(0.95),
        ).await
            .map_err(|e| AppError(format!("Qdrant search failed: {}", e)))?;

        if let Some(hit) = search_result.result.first() {
            
            let latency = request_start.elapsed().as_secs_f64() * 1000.0;
            if let Ok(mut m) = state.metrics.lock() {
                m.total_request += 1;
                m.cache_hits += 1;
                m.total_hit_latency_ms += latency;
            }
            println!(">> CACHE HIT! Similarity Score: {:.4}, Latency score: {:.2}", hit.score, latency);
            
            if let Some(cached_resp) = hit.payload.get("response").and_then(|v| v.as_str()) {
                final_response_text = format!("[CACHED] {}", cached_resp);
            }
        } else {
            
            let latency = request_start.elapsed().as_secs_f64() * 1000.0;
            if let Ok(mut m) = state.metrics.lock() {
                m.total_request += 1;
                m.cache_misses += 1;
                m.total_miss_latency_ms += latency;
            }
            println!(">> CACHE MISS! Latency score: {:.2} (includes llm call)", latency);

            // final_response_text = format!("This is dynamically generated response for: {}", user_prompt);

            match llm_call(&user_prompt, &state.llm_api_key).await {
                Ok(llm_response) => {
                    final_response_text = llm_response;
                }
                Err(e) => {
                    return Err(AppError(format!("LLM Call Failed: {}", e)));
                }
            }

            let mut payload = Payload::new();
            payload.insert("prompt", user_prompt.clone());
            payload.insert("response", final_response_text.clone());

            let point = PointStruct::new(Uuid::new_v4().to_string(), mean_vector, payload);
            state.db_client.upsert_points(UpsertPointsBuilder::new("semantic-cache", vec![point],)).await
                .map_err(|e| AppError(format!("Qdrant upsert failed: {}", e)))?;
        }
    }

    Ok(Json(ChatResponse { 
        id: format!("chatcmpl-{}", Uuid::new_v4()), 
        object: "chat.completions".to_string(), 
        choices: vec![Choice {
            message: ResponseMessage {
                role: "assistant".to_string(),
                content: final_response_text,
            }
        }]  
    }))
}

async fn llm_call(prompt: &str, api_key: &str) -> Result<String, reqwest::Error> {

    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
    });

    let resp = client
        .post("https://api.groq.com/openai/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await?
        .json::<LlmApiRespone>()
        .await?;

    Ok(resp 
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .unwrap_or_else(|| "No response generated.".to_string()))
}

async fn handle_metrics(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {

    let m = state.metrics.lock().unwrap();

    let hit_rate = if m.total_request > 0 {
        (m.cache_hits as f64 / m.total_request as f64) * 100.0
    } else { 0.0 };

    let avg_hit_ms = if m.cache_hits > 0 {
        (m.total_hit_latency_ms as f64 / m.cache_hits as f64) / 100.0
    } else { 0.0 };

    let avg_miss_ms = if m.cache_misses > 0 {
        (m.total_miss_latency_ms as f64 / m.cache_misses as f64) / 100.0 
    } else { 0.0 };

    //rough cost estimate of gemini flash pricing for average 500 tokens per request at $0.000002/token

    let estimated_tokens_saved = m.cache_hits * 500;
    let estimated_cost_saved_usd = estimated_tokens_saved as f64 * 0.000002;

    Json(serde_json::json!({
        "total_request": m.total_request,
        "cache_hits": m.cache_hits,
        "cache_misses": m.cache_misses,
        "hit_rate_percent": format!("{:.1}", hit_rate),
        "avg_hit_latency_ms": format!("{:.2}", avg_hit_ms),
        "avg_miss_latency_ms": format!("{:.2}", avg_miss_ms),
        "speedup_factor": if avg_hit_ms > 0.0 { format!("{:.2}x", avg_hit_ms / avg_miss_ms) } else { "N/A".to_string() },
        "estimated_tokens_saved": estimated_tokens_saved,
        "estimated_cost_saved_usd": format!("{:.6}", estimated_cost_saved_usd)

    }))
}