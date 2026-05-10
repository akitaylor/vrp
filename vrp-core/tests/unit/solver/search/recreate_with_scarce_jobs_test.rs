use super::*;

use crate::construction::features::{JobSkills, VehicleSkillsDimension, create_skills_feature};
use crate::construction::heuristics::InsertionContext;
use crate::helpers::models::domain::{
    ProblemBuilder, TestGoalContextBuilder, get_customer_ids_from_routes_sorted, get_customer_ids_from_unassigned,
};
use crate::helpers::models::problem::{FleetBuilder, TestSingleBuilder, TestVehicleBuilder, test_driver};
use crate::models::ViolationCode;
use crate::solver::search::DummyRecreate;
use crate::solver::{RefinementContext, create_elitism_population};
use rosomaxa::evolution::TelemetryMode;
use rosomaxa::prelude::Environment;
use std::collections::HashSet;
use std::sync::Arc;

const SKILL_CONSTRAINT_CODE: ViolationCode = ViolationCode(7);

#[test]
fn can_insert_and_lock_job_with_single_compatible_vehicle() {
    let job = create_job("scarce", Some(vec!["permit_a"]), None, None);
    let other = create_job("other", None, None, None);
    let problem = Arc::new(create_problem(vec![job.clone(), other.clone()]));
    let environment = Arc::new(Environment::default());
    let refinement_ctx = RefinementContext::new(
        problem.clone(),
        Box::new(create_elitism_population(problem.goal.clone(), environment.clone())),
        TelemetryMode::None,
        environment.clone(),
    );
    let insertion_ctx = InsertionContext::new(problem, environment);
    let recreate = RecreateWithScarceJobs::new(
        Arc::new(DummyRecreate),
        Arc::new(ScarceJobsSettings { max_compatible_vehicles: 1, lock_compatible_vehicles: 1, log: false }),
    );

    let result = recreate.run(&refinement_ctx, insertion_ctx);

    assert_eq!(get_customer_ids_from_routes_sorted(&result), vec![vec!["scarce".to_string()]]);
    assert_eq!(get_customer_ids_from_unassigned(&result), vec!["other".to_string()]);
    assert!(result.solution.locked.contains(&job));
    assert!(!result.solution.locked.contains(&other));
}

#[test]
fn can_leave_two_vehicle_job_unlocked_when_lock_threshold_is_lower() {
    let scarce = create_job("scarce", None, Some(vec!["permit_a", "permit_b"]), None);
    let other = create_job("other", None, None, None);
    let problem = Arc::new(create_problem(vec![scarce.clone(), other.clone()]));
    let environment = Arc::new(Environment::default());
    let refinement_ctx = RefinementContext::new(
        problem.clone(),
        Box::new(create_elitism_population(problem.goal.clone(), environment.clone())),
        TelemetryMode::None,
        environment.clone(),
    );
    let insertion_ctx = InsertionContext::new(problem, environment);
    let recreate = RecreateWithScarceJobs::new(
        Arc::new(DummyRecreate),
        Arc::new(ScarceJobsSettings { max_compatible_vehicles: 2, lock_compatible_vehicles: 1, log: false }),
    );

    let result = recreate.run(&refinement_ctx, insertion_ctx);

    assert_eq!(get_customer_ids_from_routes_sorted(&result), vec![vec!["scarce".to_string()]]);
    assert_eq!(get_customer_ids_from_unassigned(&result), vec!["other".to_string()]);
    assert!(!result.solution.locked.contains(&scarce));
}

fn create_problem(jobs: Vec<Job>) -> Problem {
    let mut permit_a = TestVehicleBuilder::default();
    permit_a.id("permit_a");
    permit_a.dimens_mut().set_vehicle_skills(HashSet::from_iter(["permit_a".to_string()]));

    let mut permit_b = TestVehicleBuilder::default();
    permit_b.id("permit_b");
    permit_b.dimens_mut().set_vehicle_skills(HashSet::from_iter(["permit_b".to_string()]));

    let mut generic = TestVehicleBuilder::default();
    generic.id("generic");

    let fleet = FleetBuilder::default()
        .add_driver(test_driver())
        .add_vehicle(permit_a.build())
        .add_vehicle(permit_b.build())
        .add_vehicle(generic.build())
        .build();

    let goal = TestGoalContextBuilder::default()
        .add_feature(create_skills_feature("skills", SKILL_CONSTRAINT_CODE).unwrap())
        .build();

    let mut problem = ProblemBuilder::default();
    problem.with_fleet(fleet).with_goal(goal).with_jobs(jobs);
    problem.build()
}

fn create_job(id: &str, all_of: Option<Vec<&str>>, one_of: Option<Vec<&str>>, none_of: Option<Vec<&str>>) -> Job {
    let mut job = TestSingleBuilder::default();
    job.id(id);
    job.dimens_mut().set_job_skills(JobSkills {
        all_of: all_of.map(|skills| skills.into_iter().map(|skill| skill.to_string()).collect()),
        one_of: one_of.map(|skills| skills.into_iter().map(|skill| skill.to_string()).collect()),
        none_of: none_of.map(|skills| skills.into_iter().map(|skill| skill.to_string()).collect()),
    });
    job.build_as_job_ref()
}
