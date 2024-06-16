//! A job-vehicle skills feature.

#[cfg(test)]
#[path = "../../../tests/unit/construction/features/skills_test.rs"]
mod skills_test;

use super::*;
use std::collections::HashSet;

/// Provides a way to work with the job-vehicle skills feature.
pub trait JobSkillsAspects: Clone + Send + Sync {
    /// Returns job skills if defined.
    fn get_job_skills<'a>(&self, job: &'a Job) -> Option<&'a JobSkills>;

    /// Returns job skills for vehicle if defined.
    fn get_vehicle_skills<'a>(&self, vehicle: &'a Vehicle) -> Option<&'a HashSet<String>>;

    /// Returns a violation code.
    fn get_violation_code(&self) -> ViolationCode;
}

/// A job skills limitation for a vehicle.
pub struct JobSkills {
    /// Vehicle should have all of these skills defined.
    pub all_of: Option<HashSet<String>>,
    /// Vehicle should have at least one of these skills defined.
    pub one_of: Option<HashSet<String>>,
    /// Vehicle should have none of these skills defined.
    pub none_of: Option<HashSet<String>>,
}

impl JobSkills {
    /// Creates a new instance of [`JobSkills`].
    pub fn new(all_of: Option<Vec<String>>, one_of: Option<Vec<String>>, none_of: Option<Vec<String>>) -> Self {
        let map: fn(Option<Vec<_>>) -> Option<HashSet<_>> =
            |skills| skills.and_then(|v| if v.is_empty() { None } else { Some(v.into_iter().collect()) });

        Self { all_of: map(all_of), one_of: map(one_of), none_of: map(none_of) }
    }
}

/// Creates a skills feature as hard constraint.
pub fn create_skills_feature<A>(name: &str, aspects: A) -> Result<Feature, GenericError>
where
    A: JobSkillsAspects + 'static,
{
    FeatureBuilder::default().with_name(name).with_constraint(SkillsConstraint { aspects }).build()
}

struct SkillsConstraint<A: JobSkillsAspects> {
    aspects: A,
}

impl<A: JobSkillsAspects> FeatureConstraint for SkillsConstraint<A> {
    fn evaluate(&self, move_ctx: &MoveContext<'_>) -> Option<ConstraintViolation> {
        match move_ctx {
            MoveContext::Route { route_ctx, job, .. } => {
                if let Some(job_skills) = self.aspects.get_job_skills(job) {
                    let vehicle_skills = self.aspects.get_vehicle_skills(&route_ctx.route().actor.vehicle);
                    let is_ok = check_all_of(job_skills, &vehicle_skills)
                        && check_one_of(job_skills, &vehicle_skills)
                        && check_none_of(job_skills, &vehicle_skills);
                    if !is_ok {
                        return ConstraintViolation::fail(self.aspects.get_violation_code());
                    }
                }

                None
            }
            MoveContext::Activity { .. } => None,
        }
    }

    fn merge(&self, source: Job, candidate: Job) -> Result<Job, ViolationCode> {
        let source_skills = self.aspects.get_job_skills(&source);
        let candidate_skills = self.aspects.get_job_skills(&candidate);

        let check_skill_sets = |source_set: Option<&HashSet<String>>, candidate_set: Option<&HashSet<String>>| match (
            source_set,
            candidate_set,
        ) {
            (Some(_), None) | (None, None) => true,
            (None, Some(_)) => false,
            (Some(source_skills), Some(candidate_skills)) => candidate_skills.is_subset(source_skills),
        };

        let has_comparable_skills = match (source_skills, candidate_skills) {
            (Some(_), None) | (None, None) => true,
            (None, Some(_)) => false,
            (Some(source_skills), Some(candidate_skills)) => {
                check_skill_sets(source_skills.all_of.as_ref(), candidate_skills.all_of.as_ref())
                    && check_skill_sets(source_skills.one_of.as_ref(), candidate_skills.one_of.as_ref())
                    && check_skill_sets(source_skills.none_of.as_ref(), candidate_skills.none_of.as_ref())
            }
        };

        if has_comparable_skills {
            Ok(source)
        } else {
            Err(self.aspects.get_violation_code())
        }
    }
}

fn check_all_of(job_skills: &JobSkills, vehicle_skills: &Option<&HashSet<String>>) -> bool {
    match (job_skills.all_of.as_ref(), vehicle_skills) {
        (Some(job_skills), Some(vehicle_skills)) => job_skills.is_subset(vehicle_skills),
        (Some(skills), None) if skills.is_empty() => true,
        (Some(_), None) => false,
        _ => true,
    }
}

fn check_one_of(job_skills: &JobSkills, vehicle_skills: &Option<&HashSet<String>>) -> bool {
    match (job_skills.one_of.as_ref(), vehicle_skills) {
        (Some(job_skills), Some(vehicle_skills)) => job_skills.iter().any(|skill| vehicle_skills.contains(skill)),
        (Some(skills), None) if skills.is_empty() => true,
        (Some(_), None) => false,
        _ => true,
    }
}

fn check_none_of(job_skills: &JobSkills, vehicle_skills: &Option<&HashSet<String>>) -> bool {
    match (job_skills.none_of.as_ref(), vehicle_skills) {
        (Some(job_skills), Some(vehicle_skills)) => job_skills.is_disjoint(vehicle_skills),
        _ => true,
    }
}
