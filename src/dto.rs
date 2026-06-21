use serde::{Deserialize, Serialize};

#[derive(Deserialize, Debug)]
pub struct ChatRequest {
    pub(crate) model: String,
    pub(crate) messages: Vec<Message>,
}

#[derive(Deserialize, Debug)]
pub struct Message {
    pub(crate) role: String,
    pub(crate) content: String,
}

#[derive(Serialize)]
pub struct ChatResponse {
    pub(crate) id: String,
    pub(crate) object: String,
    pub(crate) choices: Vec<Choice>,
}

#[derive(Serialize)]
pub struct Choice {
    pub(crate) message: ResponseMessage,
}

#[derive(Serialize)]
pub struct ResponseMessage {
    pub(crate) role: String,
    pub(crate) content: String,
}
