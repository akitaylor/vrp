use crate::construction::features::{MinimizeUnassignedBuilder, TransportFeatureBuilder};
use crate::helpers::construction::heuristics::TestInsertionContextBuilder;
use crate::helpers::models::domain::{ProblemBuilder, TestGoalContextBuilder, get_customer_ids_from_routes_sorted, test_random};
use crate::helpers::models::problem::*;
use crate::helpers::models::solution::{ActivityBuilder, RouteBuilder, RouteContextBuilder};
use crate::models::ViolationCode;
use crate::models::problem::{Job, VehicleIdDimension};
use crate::models::solution::Registry;

#[test]
fn can_reassign_route_to_cheaper_unused_vehicle_using_repair() {
    let expensive_vehicle = {
        let mut vehicle = TestVehicleBuilder::default();
        vehicle.id("expensive");
        let mut vehicle = vehicle.build();
        vehicle.costs.fixed = 100.;
        vehicle.costs.per_distance = 10.;
        vehicle
    };
    let cheap_vehicle = {
        let mut vehicle = TestVehicleBuilder::default();
        vehicle.id("cheap");
        let mut vehicle = vehicle.build();
        vehicle.costs.fixed = 0.;
        vehicle.costs.per_distance = 1.;
        vehicle
    };
    let fleet = FleetBuilder::default()
        .add_driver(test_driver())
        .add_vehicle(expensive_vehicle)
        .add_vehicle(cheap_vehicle)
        .build();

    let customer = TestSingleBuilder::default().id("job1").location(Some(10)).build_shared();
    let route = RouteBuilder::default()
        .with_vehicle(&fleet, "expensive")
        .add_activity(ActivityBuilder::with_location(10).job(Some(customer.clone())).build())
        .build();
    let routes = vec![RouteContextBuilder::default().with_route(route).build()];
    let jobs = vec![Job::Single(customer)];
    let transport = TestTransportCost::new_shared();
    let activity = TestActivityCost::new_shared();

    let problem = ProblemBuilder::default()
        .with_goal(
            TestGoalContextBuilder::default()
                .add_feature(MinimizeUnassignedBuilder::new("min_unassigned").build().unwrap())
                .add_feature(
                    TransportFeatureBuilder::new("transport")
                        .set_violation_code(ViolationCode(1))
                        .set_transport_cost(transport)
                        .set_activity_cost(activity)
                        .build_minimize_cost()
                        .unwrap(),
                )
                .build(),
        )
        .with_fleet(fleet)
        .with_jobs(jobs)
        .build();
    let registry = Registry::new(problem.fleet.as_ref(), test_random());

    let mut insertion_ctx = TestInsertionContextBuilder::default()
        .with_problem(problem)
        .with_registry(registry)
        .with_routes(routes)
        .build();
    for route_ctx in insertion_ctx.solution.routes.iter() {
        assert!(insertion_ctx.solution.registry.use_route(route_ctx));
    }
    insertion_ctx.problem.goal.accept_solution_state(&mut insertion_ctx.solution);

    let expensive_actor = insertion_ctx.solution.routes[0].route().actor.clone();
    let cheap_actor = insertion_ctx
        .problem
        .fleet
        .actors
        .iter()
        .find(|actor| actor.vehicle.dimens.get_vehicle_id() == Some(&"cheap".to_string()))
        .unwrap()
        .clone();
    let move_ctx = super::RepairAllocationMove::Reassign {
        source_actor: expensive_actor,
        candidate_actor: cheap_actor,
        force_actor: true,
        improvement: 1.,
    };

    let result = super::try_apply_exact_swap(&insertion_ctx, &move_ctx).unwrap();

    assert!(result.solution.unassigned.is_empty());
    assert_eq!(get_customer_ids_from_routes_sorted(&result), vec![vec!["job1".to_string()]]);
    assert_eq!(result.solution.routes[0].route().actor.vehicle.dimens.get_vehicle_id(), Some(&"cheap".to_string()));
}
