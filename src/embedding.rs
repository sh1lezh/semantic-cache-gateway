use crate::constants::EMBEDDING_DIM;
use crate::error::AppError;
use crate::state::AppState;
use ndarray::Array2;
use ort::value::Value;

pub fn embed_prompt(state: &AppState, user_prompt: &str) -> Result<Vec<f32>, AppError> {
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
        .map_err(|e| {
            AppError::internal(format!("Failed to extract last_hidden_state tensor: {e}"))
        })?;

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
        return Err(AppError::bad_request(
            "prompt contained no real tokens after masking",
        ));
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
