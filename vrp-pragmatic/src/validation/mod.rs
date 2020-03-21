//! This module provides functionality to validate problem definition for logical correctness.

use crate::json::problem::*;

pub struct ValidationContext<'a> {
    pub problem: &'a Problem,
    pub matrices: Option<&'a Vec<Matrix>>,
}

mod common;
use self::common::*;

mod jobs;
use self::jobs::validate_jobs;

mod objectives;
use self::objectives::validate_objectives;

mod vehicles;
use self::vehicles::validate_vehicles;

const VALIDATION_MESSAGE_PREFIX: &str = "Problem has the following validation errors:\n";

impl<'a> ValidationContext<'a> {
    /// Creates an instance of `ValidationContext`.
    pub fn new(problem: &'a Problem, matrices: Option<&'a Vec<Matrix>>) -> Self {
        Self { problem, matrices }
    }

    /// Validates problem on set of rules.
    pub fn validate(&self) -> Result<(), String> {
        let errors = validate_jobs(&self)
            .err()
            .into_iter()
            .chain(validate_vehicles(&self).err().into_iter())
            .chain(validate_objectives(&self).err().into_iter())
            .flatten()
            .collect::<Vec<_>>();

        if errors.is_empty() {
            Ok(())
        } else {
            Err(format!("{}{}", VALIDATION_MESSAGE_PREFIX, errors.join("\n")))
        }
    }

    /// Get list of jobs from the problem.
    fn jobs(&self) -> impl Iterator<Item = &Job> {
        self.problem.plan.jobs.iter()
    }

    /// Get list of vehicles from the problem.
    fn vehicles(&self) -> impl Iterator<Item = &VehicleType> {
        self.problem.fleet.vehicles.iter()
    }
}
