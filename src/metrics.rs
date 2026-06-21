use crate::state::AppState;

#[derive(Default)]
pub struct MetricsData {
    pub(crate) total_requests: u64,
    pub(crate) cache_hits: u64,
    pub(crate) cache_misses: u64,
    pub(crate) total_hit_latency_ms: f64,
    pub(crate) total_miss_latency_ms: f64,
}

pub fn update_hit_metrics(state: &AppState, latency_ms: f64) {
    if let Ok(mut m) = state.metrics.lock() {
        m.total_requests += 1;
        m.cache_hits += 1;
        m.total_hit_latency_ms += latency_ms;
    }
}

pub fn update_miss_metrics(state: &AppState, latency_ms: f64) {
    if let Ok(mut m) = state.metrics.lock() {
        m.total_requests += 1;
        m.cache_misses += 1;
        m.total_miss_latency_ms += latency_ms;
    }
}
