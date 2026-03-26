//! Specifies different properties as extension points on Dimensions type.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use vrp_core::construction::features::BreakPolicy;
use vrp_core::custom_dimension;
use vrp_core::models::common::Dimensions;
use vrp_core::utils::Float;

custom_dimension!(pub VehicleType typeof String);

custom_dimension!(pub ShiftIndex typeof usize);

custom_dimension!(pub TourSize typeof usize);

custom_dimension!(pub PlaceTags typeof Vec<(usize, String)>);

custom_dimension!(pub JobOrder typeof i32);

custom_dimension!(pub JobValue typeof Float);

custom_dimension!(pub JobType typeof String);

custom_dimension!(pub BreakPolicy typeof BreakPolicy);

custom_dimension!(pub VehicleKey typeof u64);

custom_dimension!(pub ConditionalJobKind typeof JobKind);

/// Describes a special conditional job kind used in hot-path feature checks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JobKind {
    /// Optional or required break job.
    Break,
    /// Reload marker job.
    Reload,
    /// Recharge marker job.
    Recharge,
}

pub(crate) fn get_vehicle_key(vehicle_id: &str, shift_index: usize) -> u64 {
    let mut hasher = DefaultHasher::new();
    vehicle_id.hash(&mut hasher);
    shift_index.hash(&mut hasher);
    hasher.finish()
}

pub(crate) fn get_conditional_job_kind(job_type: &str) -> Option<JobKind> {
    match job_type {
        "break" => Some(JobKind::Break),
        "reload" => Some(JobKind::Reload),
        "recharge" => Some(JobKind::Recharge),
        _ => None,
    }
}
