//! Runtime subsystem coordination.
//!
//! [`RuntimeReconfigurator`] applies committed
//! configuration changes to live subsystems with
//! the smallest possible blast radius.

pub mod model;
pub mod reconfigure;
pub mod subsystems;

pub use model::ModelService;
pub use reconfigure::RuntimeReconfigurator;
