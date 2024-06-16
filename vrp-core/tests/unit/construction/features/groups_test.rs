use super::*;
use crate::construction::enablers::create_typed_actor_groups;
use crate::helpers::models::domain::{test_random, TestGoalContextBuilder};
use crate::helpers::models::problem::{test_driver, test_vehicle_with_id, FleetBuilder, SingleBuilder};
use crate::helpers::models::solution::{ActivityBuilder, RouteBuilder, RouteContextBuilder, RouteStateBuilder};
use crate::models::problem::Actor;
use crate::models::problem::{Fleet, Single};
use crate::models::solution::Registry;
use std::collections::HashMap;
use std::sync::Arc;

const VIOLATION_CODE: ViolationCode = 1;

#[derive(Clone)]
struct TestGroupAspects {
    state_key: StateKey,
}

struct GroupDimenKey;

impl GroupAspects for TestGroupAspects {
    fn get_job_group<'a>(&self, job: &'a Job) -> Option<&'a String> {
        job.dimens().get_value::<GroupDimenKey, _>()
    }

    fn get_state_key(&self) -> StateKey {
        self.state_key
    }

    fn get_violation_code(&self) -> ViolationCode {
        VIOLATION_CODE
    }
}

fn create_test_group_feature(total_jobs: usize, state_key: StateKey) -> Feature {
    create_group_feature("group", total_jobs, TestGroupAspects { state_key }).unwrap()
}

fn get_total_jobs(routes: &[(&str, Vec<Option<&str>>)]) -> usize {
    routes.iter().map(|(_, jobs)| jobs.len()).sum::<usize>() + 1
}

fn create_test_fleet() -> Fleet {
    FleetBuilder::default()
        .add_driver(test_driver())
        .add_vehicle(test_vehicle_with_id("v1"))
        .add_vehicle(test_vehicle_with_id("v2"))
        .with_group_key_fn(Box::new(|actors| {
            Box::new(create_typed_actor_groups(actors, |a| a.vehicle.dimens.get_vehicle_id().cloned().unwrap()))
        }))
        .build()
}

fn create_test_single(group: Option<&str>) -> Arc<Single> {
    let mut builder = SingleBuilder::default();

    if let Some(group) = group {
        builder.property::<GroupDimenKey, _>(group.to_string());
    }

    builder.build_shared()
}

fn create_test_solution_context(
    total_jobs: usize,
    fleet: &Fleet,
    routes: Vec<(&str, Vec<Option<&str>>)>,
    state_key: StateKey,
) -> SolutionContext {
    SolutionContext {
        required: (0..total_jobs).map(|_| Job::Single(create_test_single(None))).collect(),
        ignored: vec![],
        unassigned: Default::default(),
        locked: Default::default(),
        routes: routes
            .into_iter()
            .map(|(vehicle, groups)| {
                RouteContextBuilder::default()
                    .with_state(
                        RouteStateBuilder::default()
                            .add_route_state(
                                state_key,
                                (
                                    groups.iter().filter_map(|g| *g).map(|g| g.to_string()).collect::<HashSet<_>>(),
                                    groups.len(),
                                ),
                            )
                            .build(),
                    )
                    .with_route(
                        RouteBuilder::default()
                            .with_vehicle(fleet, vehicle)
                            .add_activities(groups.into_iter().map(|group| {
                                ActivityBuilder::with_location(1).job(Some(create_test_single(group))).build()
                            }))
                            .build(),
                    )
                    .build()
            })
            .collect(),
        registry: RegistryContext::new(&TestGoalContextBuilder::default().build(), Registry::new(fleet, test_random())),
        state: Default::default(),
    }
}

fn get_actor(fleet: &Fleet, vehicle: &str) -> Arc<Actor> {
    fleet.actors.iter().find(|actor| actor.vehicle.dimens.get_vehicle_id().unwrap() == vehicle).unwrap().clone()
}

fn get_actor_groups(solution_ctx: &mut SolutionContext, state_key: StateKey) -> HashMap<String, Arc<Actor>> {
    solution_ctx
        .routes
        .iter()
        .filter_map(|route_ctx| {
            route_ctx
                .state()
                .get_route_state::<HashSet<String>>(state_key)
                .map(|groups| (route_ctx.route().actor.clone(), groups.clone()))
        })
        .fold(HashMap::default(), |mut acc, (actor, groups)| {
            groups.into_iter().for_each(|group| {
                acc.insert(group, actor.clone());
            });
            acc
        })
}

fn compare_actor_groups(fleet: &Fleet, original: HashMap<String, Arc<Actor>>, expected: Vec<(&str, &str)>) {
    let test = expected
        .iter()
        .map(|(group, vehicle)| (group.to_string(), get_actor(fleet, vehicle)))
        .collect::<HashMap<_, _>>();

    assert_eq!(original.len(), test.len());
    assert!(original.keys().all(|k| test[k] == original[k]));
}

#[test]
fn can_build_expected_state() {
    let state_key = StateKeyRegistry::default().next_key();
    let total_jobs = 1;
    let state = create_test_group_feature(total_jobs, state_key).state.unwrap();

    assert_eq!(state.state_keys().cloned().collect::<Vec<_>>(), vec![state_key]);
}

parameterized_test! {can_accept_insertion, (routes, job_group, expected), {
    can_accept_insertion_impl(routes, job_group, expected);
}}

can_accept_insertion! {
    case_01: (vec![("v1", vec![None])], Some("g1"), vec![("g1", "v1")]),
    case_02: (vec![("v1", vec![None]), ("v2", vec![Some("g2")])], Some("g1"), vec![("g1", "v1"), ("g2", "v2")]),
}

fn can_accept_insertion_impl(
    routes: Vec<(&str, Vec<Option<&str>>)>,
    job_group: Option<&str>,
    expected: Vec<(&str, &str)>,
) {
    let state_key = StateKeyRegistry::default().next_key();
    let total_jobs = get_total_jobs(&routes) + 1;
    let fleet = create_test_fleet();
    let state = create_test_group_feature(total_jobs, state_key).state.unwrap();
    let mut solution_ctx = create_test_solution_context(total_jobs, &fleet, routes, state_key);
    state.accept_solution_state(&mut solution_ctx);

    state.accept_insertion(&mut solution_ctx, 0, &Job::Single(create_test_single(job_group)));

    compare_actor_groups(&fleet, get_actor_groups(&mut solution_ctx, state_key), expected);
}

parameterized_test! {can_accept_solution_state, (routes, expected), {
    can_accept_solution_state_impl(routes, expected);
}}

can_accept_solution_state! {
    case_01: (vec![("v1", vec![Some("g1")])], vec![("g1", "v1")]),
    case_02: (vec![("v1", vec![Some("g1")]), ("v2", vec![Some("g2")])], vec![("g1", "v1"), ("g2", "v2")]),
    case_03: (vec![("v1", vec![Some("g1")]), ("v1", vec![Some("g2")])], vec![("g1", "v1"), ("g2", "v1")]),
    case_04: (vec![("v1", vec![None])], vec![]),
}

fn can_accept_solution_state_impl(routes: Vec<(&str, Vec<Option<&str>>)>, expected: Vec<(&str, &str)>) {
    let state_key = StateKeyRegistry::default().next_key();
    let total_jobs = get_total_jobs(&routes) + 1;
    let fleet = create_test_fleet();
    let state = create_test_group_feature(total_jobs, state_key).state.unwrap();
    let mut solution_ctx = create_test_solution_context(total_jobs, &fleet, routes, state_key);

    state.accept_solution_state(&mut solution_ctx);

    compare_actor_groups(&fleet, get_actor_groups(&mut solution_ctx, state_key), expected);
}

parameterized_test! {can_evaluate_job, (routes, route_idx, job_group, expected), {
    can_evaluate_job_impl(routes, route_idx, job_group, expected);
}}

can_evaluate_job! {
    case_01: (vec![("v1", vec![]), ("v2", vec![Some("g1")])], 0, Some("g1"), Some(VIOLATION_CODE)),
    case_02: (vec![("v1", vec![]), ("v2", vec![])], 0, Some("g1"), None),
}

fn can_evaluate_job_impl(
    routes: Vec<(&str, Vec<Option<&str>>)>,
    route_idx: usize,
    job_group: Option<&str>,
    expected: Option<i32>,
) {
    let state_key = StateKeyRegistry::default().next_key();
    let total_jobs = get_total_jobs(&routes) + 1;
    let fleet = create_test_fleet();
    let solution_ctx = create_test_solution_context(total_jobs, &fleet, routes, state_key);
    let route_ctx = solution_ctx.routes.get(route_idx).unwrap();
    let job = Job::Single(create_test_single(job_group));
    let constraint = create_test_group_feature(total_jobs, state_key).constraint.unwrap();

    let result = constraint.evaluate(&MoveContext::route(&solution_ctx, route_ctx, &job));

    assert_eq!(result, expected.map(|code| ConstraintViolation { code, stopped: true }));
}

parameterized_test! {can_merge_groups, (source, candidate, expected), {
    can_merge_groups_impl(Job::Single(source), Job::Single(candidate), expected);
}}

can_merge_groups! {
    case_01: (create_test_single(Some("group1")), create_test_single(Some("group2")), Err(VIOLATION_CODE)),
    case_02: (create_test_single(Some("group1")), create_test_single(Some("group1")), Ok(())),
    case_03: (create_test_single(None), create_test_single(Some("group1")), Err(VIOLATION_CODE)),
    case_04: (create_test_single(Some("group1")), create_test_single(None), Err(VIOLATION_CODE)),
    case_05: (create_test_single(None), create_test_single(None), Ok(())),
}

fn can_merge_groups_impl(source: Job, candidate: Job, expected: Result<(), i32>) {
    let state_key = StateKeyRegistry::default().next_key();
    let total_jobs = 1;
    let constraint = create_test_group_feature(total_jobs, state_key).constraint.unwrap();

    let result = constraint.merge(source, candidate).map(|_| ());

    assert_eq!(result, expected);
}
