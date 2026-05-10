use super::*;
use crate::construction::features::{CapacityFeatureBuilder, TransportFeatureBuilder};
use crate::helpers::construction::features::create_simple_demand;
use crate::helpers::models::domain::{TestGoalContextBuilder, get_customer_ids_from_routes_sorted, test_logger};
use crate::helpers::models::problem::*;
use crate::helpers::models::solution::*;
use crate::helpers::solver::create_default_refinement_ctx;
use crate::models::common::SingleDimLoad;
use crate::models::problem::{Fleet, Job, MatrixData, SimpleActivityCost, Single, create_matrix_transport_cost};
use crate::models::solution::{Registry, Route};
use crate::models::{Extras, Problem, Solution, ViolationCode};
use crate::prelude::Jobs;
use rosomaxa::prelude::{DefaultRandom, Environment, Float};
use std::sync::Arc;

#[test]
fn can_relocate_near_job_and_eject_far_job() {
    let (problem, solution) = create_cluster_relocate_problem();
    let insertion_ctx =
        InsertionContext::new_from_solution(Arc::new(problem), (solution, None), Arc::new(Environment::default()));

    let new_insertion_ctx = ClusterRelocate::new(ClusterRelocateConfig {
        route_neighbors: 1,
        job_candidates: 2,
        max_evictions: 1,
        neighbor_radius: 3,
        min_shared_neighbors: 1,
        allow_unassigned: false,
    })
    .explore(&create_default_refinement_ctx(insertion_ctx.problem.clone()), &insertion_ctx)
    .expect("cannot find cluster relocate move");

    let mut actual = get_customer_ids_from_routes_sorted(&new_insertion_ctx);
    actual.iter_mut().for_each(|route| route.sort());
    actual.sort();

    assert_eq!(
        actual,
        vec![vec!["anchor".to_string(), "near".to_string()], vec!["far".to_string(), "remote".to_string()]]
    );
    assert!(new_insertion_ctx.solution.unassigned.is_empty());
}

#[test]
fn cannot_relocate_when_evictions_are_disabled_and_target_is_full() {
    let (problem, solution) = create_cluster_relocate_problem();
    let insertion_ctx =
        InsertionContext::new_from_solution(Arc::new(problem), (solution, None), Arc::new(Environment::default()));

    let new_insertion_ctx = ClusterRelocate::new(ClusterRelocateConfig {
        route_neighbors: 1,
        job_candidates: 2,
        max_evictions: 0,
        neighbor_radius: 3,
        min_shared_neighbors: 1,
        allow_unassigned: false,
    })
    .explore(&create_default_refinement_ctx(insertion_ctx.problem.clone()), &insertion_ctx);

    assert!(new_insertion_ctx.is_none());
}

fn create_cluster_relocate_problem() -> (Problem, Solution) {
    let transport =
        create_matrix_transport_cost(vec![MatrixData::new(0, None, create_test_matrix(), create_test_matrix())])
            .unwrap();
    let activity = Arc::new(SimpleActivityCost::default());

    let fleet = Arc::new(
        FleetBuilder::default()
            .add_driver(test_driver())
            .add_vehicle(TestVehicleBuilder::default().id("v1").capacity(2).build())
            .add_vehicle(TestVehicleBuilder::default().id("v2").capacity(2).build())
            .build(),
    );

    let anchor = create_job("anchor", 0);
    let far = create_job("far", 1);
    let near = create_job("near", 2);
    let remote = create_job("remote", 3);
    let jobs = vec![
        Job::Single(anchor.clone()),
        Job::Single(far.clone()),
        Job::Single(near.clone()),
        Job::Single(remote.clone()),
    ];

    let goal = TestGoalContextBuilder::empty()
        .add_feature(
            CapacityFeatureBuilder::<SingleDimLoad>::new("capacity")
                .set_violation_code(ViolationCode(1))
                .build()
                .unwrap(),
        )
        .add_feature(
            TransportFeatureBuilder::new("transport")
                .set_violation_code(ViolationCode(2))
                .set_transport_cost(transport.clone())
                .set_activity_cost(activity.clone())
                .build_minimize_cost()
                .unwrap(),
        )
        .build();

    let problem = Problem {
        fleet: fleet.clone(),
        jobs: Arc::new(Jobs::new(fleet.as_ref(), jobs, transport.as_ref(), &test_logger()).unwrap()),
        locks: vec![],
        goal: Arc::new(goal),
        activity,
        transport,
        extras: Arc::new(Extras::default()),
    };

    let routes = vec![
        create_route(fleet.as_ref(), "v1", vec![anchor, far]),
        create_route(fleet.as_ref(), "v2", vec![near, remote]),
    ];

    let solution = Solution {
        cost: 0.,
        registry: Registry::new(fleet.as_ref(), Arc::new(DefaultRandom::default())),
        routes,
        unassigned: Default::default(),
        telemetry: None,
    };

    (problem, solution)
}

fn create_job(id: &str, location: usize) -> Arc<Single> {
    TestSingleBuilder::default().id(id).location(Some(location)).demand(create_simple_demand(1)).build_shared()
}

fn create_route(fleet: &Fleet, vehicle_id: &str, jobs: Vec<Arc<Single>>) -> Route {
    let mut builder = RouteBuilder::default();
    builder.with_vehicle(fleet, vehicle_id);
    jobs.into_iter().for_each(|job| {
        let location = job.places.first().and_then(|place| place.location).unwrap();
        builder.add_activity(ActivityBuilder::with_location(location).job(Some(job)).build());
    });

    builder.build()
}

fn create_test_matrix() -> Vec<Float> {
    // locations: anchor=0, far=1, near=2, remote=3
    // anchor/near and far/remote are close pairs; mixed pairs are expensive.
    vec![0., 100., 1., 101., 100., 0., 99., 1., 1., 99., 0., 100., 101., 1., 100., 0.]
}
