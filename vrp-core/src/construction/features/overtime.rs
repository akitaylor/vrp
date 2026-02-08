use super::*;

/// Specifies a soft work time limit with overtime penalty.
#[derive(Clone, Copy, Debug)]
pub struct Overtime {
    /// Work time limit in seconds.
    pub limit: Duration,
    /// Overtime cost per time unit.
    pub cost: Cost,
}

impl Overtime {
    /// Calculates overtime penalty for given duration.
    pub fn penalty(&self, duration: Duration) -> Cost {
        (duration - self.limit).max(0.) * self.cost
    }
}

custom_dimension!(pub VehicleOvertime typeof Overtime);
