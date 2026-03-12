pub mod audio;
pub mod download;
pub mod inference;
pub mod model;
pub mod queue;
pub mod text;

use model::ModelHandle;

#[derive(Clone)]
pub enum ModelStatus {
    Idle,
    Downloading { progress: f32 },
    Loading,
    Ready(std::sync::Arc<ModelHandle>),
    Failed(String),
}
