#[cfg(test)]
#[path = "../../../../tests/unit/solver/search/recreate_with_scarce_jobs_test.rs"]
mod recreate_with_scarce_jobs_test;

use super::*;
use crate::construction::features::{JobSkills, JobSkillsDimension, VehicleSkillsDimension};
use crate::construction::heuristics::{
    BestResultSelector, EvaluationContext, InsertionPosition, InsertionResult, LegSelection, apply_insertion_success,
    eval_job_insertion_in_route,
};
use crate::models::Extras;
use crate::models::problem::{Actor, Job, JobIdDimension, VehicleIdDimension};
use std::cmp::Ordering;
use std::collections::HashSet;

custom_extra_property!(pub ScarceJobsSettings typeof ScarceJobsSettings);

/// Controls scarce job handling during recreate.
#[derive(Clone, Debug)]
pub struct ScarceJobsSettings {
    /// Jobs with compatible vehicle count up to this threshold are inserted first.
    pub max_compatible_vehicles: usize,
    /// Jobs with compatible vehicle count up to this threshold are locked after assignment.
    pub lock_compatible_vehicles: usize,
    /// Enables verbose logging.
    pub log: bool,
}

/// Recreate wrapper which prioritizes jobs compatible with only a few vehicles.
pub struct RecreateWithScarceJobs {
    inner: Arc<dyn Recreate>,
    settings: Arc<ScarceJobsSettings>,
}

impl RecreateWithScarceJobs {
    /// Creates a new instance of `RecreateWithScarceJobs`.
    pub fn new(inner: Arc<dyn Recreate>, settings: Arc<ScarceJobsSettings>) -> Self {
        Self { inner, settings }
    }
}

impl Recreate for RecreateWithScarceJobs {
    fn run(&self, refinement_ctx: &RefinementContext, insertion_ctx: InsertionContext) -> InsertionContext {
        let insertion_ctx = assign_scarce_jobs(insertion_ctx, self.settings.as_ref());
        self.inner.run(refinement_ctx, insertion_ctx)
    }
}

fn assign_scarce_jobs(mut insertion_ctx: InsertionContext, settings: &ScarceJobsSettings) -> InsertionContext {
    if settings.max_compatible_vehicles == 0 {
        return insertion_ctx;
    }

    let mut job_compatibility = collect_scarce_jobs(&insertion_ctx, settings.max_compatible_vehicles);
    if job_compatibility.is_empty() {
        return insertion_ctx;
    }

    job_compatibility.sort_by(|left, right| compare_jobs(left, right));

    let leg_selection = LegSelection::Exhaustive;
    let result_selector = BestResultSelector::default();
    let goal = insertion_ctx.problem.goal.clone();
    let mut inserted = 0;
    let mut locked = 0;

    for (job, compatible_vehicle_ids) in job_compatibility {
        let compatible_count = compatible_vehicle_ids.len();
        let eval_ctx = EvaluationContext {
            goal: goal.as_ref(),
            job: &job,
            leg_selection: &leg_selection,
            result_selector: &result_selector,
        };

        let result = find_best_result(&insertion_ctx, &eval_ctx, &compatible_vehicle_ids);
        if let InsertionResult::Success(success) = result {
            apply_insertion_success(&mut insertion_ctx, success);
            insertion_ctx.restore();
            inserted += 1;

            if compatible_count <= settings.lock_compatible_vehicles {
                insertion_ctx.solution.locked.insert(job.clone());
                locked += 1;
            }
        }
    }

    if settings.log && inserted > 0 {
        (insertion_ctx.environment.logger)(
            format!(
                "scarce jobs: inserted={inserted}, locked={locked}, threshold={}, lock_threshold={}",
                settings.max_compatible_vehicles, settings.lock_compatible_vehicles
            )
            .as_str(),
        );
    }

    insertion_ctx
}

fn collect_scarce_jobs(
    insertion_ctx: &InsertionContext,
    max_compatible_vehicles: usize,
) -> Vec<(Job, HashSet<String>)> {
    let mut visited = HashSet::<Job>::default();

    insertion_ctx
        .solution
        .required
        .iter()
        .chain(insertion_ctx.solution.unassigned.keys())
        .filter_map(|job| {
            if !visited.insert(job.clone()) {
                return None;
            }

            let compatible_vehicle_ids = get_compatible_vehicle_ids(insertion_ctx, job);
            let compatible_count = compatible_vehicle_ids.len();

            (compatible_count > 0 && compatible_count <= max_compatible_vehicles)
                .then_some((job.clone(), compatible_vehicle_ids))
        })
        .collect()
}

fn get_compatible_vehicle_ids(insertion_ctx: &InsertionContext, job: &Job) -> HashSet<String> {
    insertion_ctx
        .problem
        .fleet
        .vehicles
        .iter()
        .filter(|vehicle| is_vehicle_compatible(job, vehicle.dimens.get_vehicle_skills()))
        .filter_map(|vehicle| vehicle.dimens.get_vehicle_id().cloned())
        .collect()
}

fn find_best_result(
    insertion_ctx: &InsertionContext,
    eval_ctx: &EvaluationContext,
    compatible_vehicle_ids: &HashSet<String>,
) -> InsertionResult {
    let is_compatible_actor = |actor: &Actor| {
        actor.vehicle.dimens.get_vehicle_id().is_some_and(|vehicle_id| compatible_vehicle_ids.contains(vehicle_id))
    };

    let available_routes = insertion_ctx
        .solution
        .registry
        .resources()
        .available()
        .filter(|actor| is_compatible_actor(actor.as_ref()))
        .map(|actor| {
            let mut route_ctx = crate::construction::heuristics::RouteContext::new(actor);
            insertion_ctx.problem.goal.accept_route_state(&mut route_ctx);
            route_ctx
        })
        .collect::<Vec<_>>();

    insertion_ctx
        .solution
        .routes
        .iter()
        .filter(|route_ctx| is_compatible_actor(route_ctx.route().actor.as_ref()))
        .chain(available_routes.iter())
        .fold(InsertionResult::make_failure(), |acc, route_ctx| {
            eval_job_insertion_in_route(insertion_ctx, eval_ctx, route_ctx, InsertionPosition::Any, acc)
        })
}

fn compare_jobs(left: &(Job, HashSet<String>), right: &(Job, HashSet<String>)) -> Ordering {
    left.1.len().cmp(&right.1.len()).then_with(|| get_job_id(&left.0).cmp(&get_job_id(&right.0)))
}

fn get_job_id(job: &Job) -> String {
    job.dimens().get_job_id().cloned().unwrap_or_default()
}

fn is_vehicle_compatible(job: &Job, vehicle_skills: Option<&HashSet<String>>) -> bool {
    let Some(job_skills) = job.dimens().get_job_skills() else {
        return true;
    };

    check_all_of(job_skills, &vehicle_skills)
        && check_one_of(job_skills, &vehicle_skills)
        && check_none_of(job_skills, &vehicle_skills)
}

fn check_all_of(job_skills: &JobSkills, vehicle_skills: &Option<&HashSet<String>>) -> bool {
    match (job_skills.all_of.as_ref(), vehicle_skills) {
        (Some(job_skills), Some(vehicle_skills)) => job_skills.is_subset(vehicle_skills),
        (Some(skills), None) if skills.is_empty() => true,
        (None, _) => true,
        _ => false,
    }
}

fn check_one_of(job_skills: &JobSkills, vehicle_skills: &Option<&HashSet<String>>) -> bool {
    match (job_skills.one_of.as_ref(), vehicle_skills) {
        (Some(job_skills), Some(vehicle_skills)) => job_skills.iter().any(|skill| vehicle_skills.contains(skill)),
        (Some(skills), None) if skills.is_empty() => true,
        (None, _) => true,
        _ => false,
    }
}

fn check_none_of(job_skills: &JobSkills, vehicle_skills: &Option<&HashSet<String>>) -> bool {
    match (job_skills.none_of.as_ref(), vehicle_skills) {
        (Some(job_skills), Some(vehicle_skills)) => job_skills.is_disjoint(vehicle_skills),
        (Some(_), None) => true,
        (None, _) => true,
    }
}
