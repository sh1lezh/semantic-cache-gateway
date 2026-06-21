use axum::{
    Router,
    routing::{get, post},
};
use ort::session::{Session, builder::GraphOptimizationLevel};
use qdrant_client::{
    Qdrant,
    qdrant::{CreateCollectionBuilder, Distance, VectorParamsBuilder},
};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;
use tokio::net::TcpListener;
mod constants;
mod dto;
mod embedding;
mod error;
mod llm;
mod metrics;
mod routes;
mod state;
use crate::constants::{COLLECTION_NAME, EMBEDDING_DIM};
use crate::metrics::MetricsData;
use crate::routes::{handle_chat, handle_health, handle_metrics};
use crate::state::AppState;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok();

    let llm_api_key =
        std::env::var("API_KEY").expect("API_KEY must be set in .env or environment.");

    // qdrant-client uses the gRPC endpoint. Expose 6334 from Docker.
    let qdrant_url =
        std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://127.0.0.1:6334".to_string());

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
