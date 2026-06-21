use crate::metrics::MetricsData;
use ort::session::Session;
use qdrant_client::Qdrant;
use std::sync::Mutex;
use tokenizers::Tokenizer;

pub struct AppState {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) db_client: Qdrant,
    pub(crate) llm_api_key: String,
    pub(crate) model_session: Mutex<Session>,
    pub(crate) cache_score_threshold: f32,
    pub(crate) metrics: Mutex<MetricsData>,
}
