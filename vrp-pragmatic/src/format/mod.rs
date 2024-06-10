//! This module defines logic to serialize/deserialize problem and routing matrix in pragmatic
//! format from json input and create and write pragmatic solution.
//!

extern crate serde_json;

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use vrp_core::construction::enablers::ReservedTimesIndex;
use vrp_core::models::common::{Distance, Duration, ValueDimension};
use vrp_core::models::problem::Job as CoreJob;
use vrp_core::models::{Extras as CoreExtras, Problem as CoreProblem};
use vrp_core::prelude::GenericError;

mod coord_index;
pub use self::coord_index::CoordIndex;

pub mod problem;
pub mod solution;

pub(crate) mod enablers;

/// Represents a location type.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Location {
    /// A location type represented by geocoordinate with latitude and longitude.
    Coordinate {
        /// Latitude.
        lat: f64,
        /// Longitude.
        lng: f64,
    },

    /// A location type represented by index reference in routing matrix.
    Reference {
        /// An index in routing matrix.
        index: usize,
    },

    /// A custom location type with no reference in matrix.
    Custom {
        /// Specifies a custom location type.
        r#type: CustomLocationType,
    },
}

impl Location {
    /// Creates a new [`Location`] as coordinate.
    pub fn new_coordinate(lat: f64, lng: f64) -> Self {
        Self::Coordinate { lat, lng }
    }

    /// Creates a new [`Location`] as index reference.
    pub fn new_reference(index: usize) -> Self {
        Self::Reference { index }
    }

    /// Creates a new [`Location`] as custom unknown type.
    pub fn new_unknown() -> Self {
        Self::Custom { r#type: CustomLocationType::Unknown }
    }

    /// Returns lat lng if location is coordinate, panics otherwise.
    pub fn to_lat_lng(&self) -> (f64, f64) {
        match self {
            Self::Coordinate { lat, lng } => (*lat, *lng),
            _ => unreachable!("expect coordinate"),
        }
    }
}

impl std::fmt::Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            Location::Coordinate { lat, lng } => write!(f, "lat={lat}, lng={lng}"),
            Location::Reference { index } => write!(f, "index={index}"),
            Location::Custom { r#type } => {
                let value = match r#type {
                    CustomLocationType::Unknown => "unknown",
                };
                write!(f, "custom={value}")
            }
        }
    }
}

/// A custom location type which has no reference to matrix.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum CustomLocationType {
    /// Unknown location type which has a zero distance/duration to any other location.
    #[serde(rename(deserialize = "unknown", serialize = "unknown"))]
    Unknown,
}

/// A format error.
#[derive(Clone, Debug, Serialize)]
pub struct FormatError {
    /// An error code in registry.
    pub code: String,
    /// A possible error cause.
    pub cause: String,
    /// An action to take in order to recover from error.
    pub action: String,
    /// A details about exception.
    pub details: Option<String>,
}

impl FormatError {
    /// Creates a new instance of `FormatError` action without details.
    pub fn new(code: String, cause: String, action: String) -> Self {
        Self { code, cause, action, details: None }
    }

    /// Creates a new instance of `FormatError` action.
    pub fn new_with_details(code: String, cause: String, action: String, details: String) -> Self {
        Self { code, cause, action, details: Some(details) }
    }

    /// Serializes error into json string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self).unwrap()
    }
}

impl std::error::Error for FormatError {}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}, cause: '{}', action: '{}'.", self.code, self.cause, self.action)
    }
}

/// Keeps track of multiple `FormatError`.
#[derive(Debug)]
pub struct MultiFormatError {
    /// Inner errors.
    pub errors: Vec<FormatError>,
}

impl MultiFormatError {
    /// Formats multiple format errors into json string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.errors).unwrap()
    }
}

impl std::error::Error for MultiFormatError {}

impl std::fmt::Display for MultiFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.errors.iter().map(|err| err.to_string()).collect::<Vec<_>>().join("\n"))
    }
}

impl From<Vec<FormatError>> for MultiFormatError {
    fn from(errors: Vec<FormatError>) -> Self {
        MultiFormatError { errors }
    }
}

impl From<MultiFormatError> for GenericError {
    fn from(value: MultiFormatError) -> Self {
        value.to_string().into()
    }
}

impl IntoIterator for MultiFormatError {
    type Item = FormatError;
    type IntoIter = <Vec<FormatError> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.errors.into_iter()
    }
}

const TIME_CONSTRAINT_CODE: i32 = 1;
const DISTANCE_LIMIT_CONSTRAINT_CODE: i32 = 2;
const DURATION_LIMIT_CONSTRAINT_CODE: i32 = 3;
const CAPACITY_CONSTRAINT_CODE: i32 = 4;
const BREAK_CONSTRAINT_CODE: i32 = 5;
const SKILL_CONSTRAINT_CODE: i32 = 6;
const LOCKING_CONSTRAINT_CODE: i32 = 7;
const REACHABLE_CONSTRAINT_CODE: i32 = 8;
const AREA_CONSTRAINT_CODE: i32 = 9;
const TOUR_SIZE_CONSTRAINT_CODE: i32 = 10;
const TOUR_ORDER_CONSTRAINT_CODE: i32 = 11;
const GROUP_CONSTRAINT_CODE: i32 = 12;
const COMPATIBILITY_CONSTRAINT_CODE: i32 = 13;
const RELOAD_RESOURCE_CONSTRAINT_CODE: i32 = 14;
const RECHARGE_CONSTRAINT_CODE: i32 = 15;

/// An job id to job index.
pub type JobIndex = HashMap<String, CoreJob>;

/// Provides way to get/set job index.
pub trait JobIndexAccessor {
    /// Sets job index.
    fn set_job_index(&mut self, coord_index: JobIndex);

    /// Gets job index.
    fn get_job_index(&self) -> Option<Arc<JobIndex>>;
}

impl JobIndexAccessor for CoreExtras {
    fn set_job_index(&mut self, coord_index: JobIndex) {
        self.set_value("job_index", coord_index);
    }

    fn get_job_index(&self) -> Option<Arc<JobIndex>> {
        self.get_value_raw("job_index")
    }
}

/// Provides way to get/set coord index.
pub trait CoordIndexAccessor {
    /// Sets coord index.
    fn set_coord_index(&mut self, coord_index: CoordIndex);

    /// Gets coord index as shared reference.
    fn get_coord_index(&self) -> Option<Arc<CoordIndex>>;
}

impl CoordIndexAccessor for CoreExtras {
    fn set_coord_index(&mut self, coord_index: CoordIndex) {
        self.set_value("coord_index", coord_index);
    }

    fn get_coord_index(&self) -> Option<Arc<CoordIndex>> {
        self.get_value_raw("coord_index")
    }
}

/// Get job and coord indices from extras
pub fn get_indices(extras: &CoreExtras) -> Result<(Arc<JobIndex>, Arc<CoordIndex>), GenericError> {
    let job_index = extras.get_job_index().ok_or_else(|| GenericError::from("cannot get job index"))?;
    let coord_index = extras.get_coord_index().ok_or_else(|| GenericError::from("cannot get coord index"))?;

    Ok((job_index, coord_index))
}
