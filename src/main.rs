use axum::{
    routing::post,
    Router,
    Json,
    extract::State
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

#[derive(Deserialize, Debug)]
struct ChatRequest {
    // model: String,
    messages: Vec<Message>,
}

#[derive(Deserialize, Debug)]
struct Message {
    // role: String,
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
}

#[tokio::main]
async fn main() -> Result<(),  Box<dyn std::error::Error>>{
    println!("Initializing Machine Learning Engine...");

    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("model/model.onnx")
        .expect("Failed to Load the Model ONNX from the hard drive...");

    let tokenizer = Tokenizer::from_file("model/tokenizer.json")
        .expect("Failed to load the Tokenizer from the hard drive...");
    
    println!("2. Connecting to Qdrant Vector Database...");
    let db_client = Qdrant::from_url("http://localhost:6334").build()?;

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
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat))
        .with_state(shared_state);

    let listener = TcpListener::bind("127.0.0.1:3000").await?;
    println!("Semantic Gateway Listening on Port 3000..... \n");

    axum::serve(listener, app).await?;

    Ok(())
}


async fn handle_chat(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Json<ChatResponse> {
    
    let mut final_response_text = String::new();

    if let Some(last_message) = payload.messages.last() {
        let user_prompt = last_message.content.clone();
        println!("--- Incoming Request: '{}' ---", user_prompt);

        let encoding = state.tokenizer.encode(user_prompt.clone(), true).unwrap();
        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        let token_type_ids = encoding.get_type_ids();
        let seq_len = input_ids.len();

        let input_ids_2d = Array2::from_shape_vec((1, seq_len), input_ids.iter().map(|&x| x as i64).collect::<Vec<_>>()).unwrap();
        let attention_mask_2d = Array2::from_shape_vec((1, seq_len), attention_mask.iter().map(|&x| x as i64).collect::<Vec<_>>()).unwrap();
        let token_type_ids_2d = Array2::from_shape_vec((1, seq_len), token_type_ids.iter().map(|&x| x as i64).collect::<Vec<_>>()).unwrap();

        let input_ids_val = Value::from_array(input_ids_2d).unwrap();
        let attention_mask_val = Value::from_array(attention_mask_2d).unwrap();
        let token_type_ids_val = Value::from_array(token_type_ids_2d).unwrap();

        let mean_vector = {
            let mut session_guard = state.model_session.lock().unwrap();

            let outputs = session_guard.run(ort::inputs![
                "input_ids" => input_ids_val,
                "attention_mask" => attention_mask_val,
                "token_type_ids" => token_type_ids_val,
            ]).unwrap(); 

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
        ).await.unwrap();

        if let Some(hit) = search_result.result.first() {
            println!(">> CACHE HIT! Similarity Score: {:.4}", hit.score);
            if let Some(cached_resp) = hit.payload.get("response").and_then(|v| v.as_str()) {
                final_response_text = format!("[CACHED] {}", cached_resp);
            }
        } else {
            println!(">> CACHE MISS! Forwarding to LLM...");

            final_response_text = format!("This is dynamically generated response for: {}", user_prompt);

            let mut payload = Payload::new();
            payload.insert("prompt", user_prompt.clone());
            payload.insert("response", final_response_text.clone());

            let point = PointStruct::new(Uuid::new_v4().to_string(), mean_vector, payload);
            state.db_client.upsert_points(UpsertPointsBuilder::new("semantic-cache", vec![point],)).await.unwrap();
        }
    }

    let response = ChatResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat-completions".to_string(),
        choices: vec![Choice {
            message: ResponseMessage {
                role: "assistant".to_string(),
                content: final_response_text,
            }
        }]
    };
    
    Json(response)
}