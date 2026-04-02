use crate::construction::heuristics::{MoveContext, SolutionContext};
use crate::helpers::construction::heuristics::TestInsertionContextBuilder;
use crate::helpers::models::solution::{ActivityBuilder, RouteBuilder, RouteContextBuilder};
use crate::models::common::Cost;
use crate::models::problem::Job;
use crate::models::{FeatureBuilder, FeatureObjective, FeatureState, GoalContextBuilder};
use std::sync::{Arc, Mutex};

struct RecomputeState {
    visited: Arc<Mutex<Vec<usize>>>,
}

impl FeatureState for RecomputeState {
    fn accept_insertion(&self, _: &mut SolutionContext, _: usize, _: &Job) {}

    fn accept_route_state(&self, _: &mut crate::construction::heuristics::RouteContext) {}

    fn accept_solution_state(&self, solution_ctx: &mut SolutionContext) {
        solution_ctx.routes.iter_mut().enumerate().filter(|(_, route_ctx)| route_ctx.is_stale()).for_each(
            |(idx, route_ctx)| {
                self.visited.lock().unwrap().push(idx);
                route_ctx.mark_stale(false);
            },
        );
    }
}

struct MarkOtherRouteStaleState;

impl FeatureState for MarkOtherRouteStaleState {
    fn accept_insertion(&self, _: &mut SolutionContext, _: usize, _: &Job) {}

    fn accept_route_state(&self, _: &mut crate::construction::heuristics::RouteContext) {}

    fn accept_solution_state(&self, solution_ctx: &mut SolutionContext) {
        solution_ctx.routes[1].mark_stale(true);
    }
}

struct MutateSameRouteOnceState {
    mutated: Arc<Mutex<bool>>,
}

impl FeatureState for MutateSameRouteOnceState {
    fn accept_insertion(&self, _: &mut SolutionContext, _: usize, _: &Job) {}

    fn accept_route_state(&self, _: &mut crate::construction::heuristics::RouteContext) {}

    fn accept_solution_state(&self, solution_ctx: &mut SolutionContext) {
        let mut mutated = self.mutated.lock().unwrap();
        if !*mutated {
            solution_ctx.routes[0].route_mut().tour.remove_activity_at(1);
            *mutated = true;
        }
    }
}

struct NoopObjective;

impl FeatureObjective for NoopObjective {
    fn fitness(&self, _: &crate::construction::heuristics::InsertionContext) -> Cost {
        0.
    }

    fn estimate(&self, _: &MoveContext<'_>) -> Cost {
        0.
    }
}

#[test]
fn can_detect_stale_route_set_changes_between_solution_state_passes() {
    let visited = Arc::new(Mutex::new(Vec::new()));
    let recompute_feature = FeatureBuilder::default()
        .with_name("recompute")
        .with_state(RecomputeState { visited: visited.clone() })
        .build()
        .unwrap();
    let mark_stale_feature =
        FeatureBuilder::default().with_name("mark_stale").with_state(MarkOtherRouteStaleState).build().unwrap();
    let objective_feature =
        FeatureBuilder::default().with_name("objective").with_objective(NoopObjective).build().unwrap();
    let goal = GoalContextBuilder::with_features(&[objective_feature, recompute_feature, mark_stale_feature])
        .unwrap()
        .build()
        .unwrap();

    let route1 = RouteContextBuilder::default().with_route(RouteBuilder::default().build()).build();
    let mut route2 = RouteContextBuilder::default().with_route(RouteBuilder::default().build()).build();
    route2.mark_stale(false);

    let mut solution_ctx = TestInsertionContextBuilder::default().with_routes(vec![route1, route2]).build().solution;

    goal.accept_solution_state(&mut solution_ctx);

    assert_eq!(*visited.lock().unwrap(), vec![0, 1]);
    assert!(solution_ctx.routes.iter().all(|route_ctx| !route_ctx.is_stale()));
}

#[test]
fn can_detect_same_stale_route_when_it_is_mutated_between_solution_state_passes() {
    let visited = Arc::new(Mutex::new(Vec::new()));
    let mutated = Arc::new(Mutex::new(false));

    let recompute_feature = FeatureBuilder::default()
        .with_name("recompute")
        .with_state(RecomputeState { visited: visited.clone() })
        .build()
        .unwrap();
    let mutate_feature = FeatureBuilder::default()
        .with_name("mutate_same_route")
        .with_state(MutateSameRouteOnceState { mutated })
        .build()
        .unwrap();
    let objective_feature =
        FeatureBuilder::default().with_name("objective").with_objective(NoopObjective).build().unwrap();
    let goal = GoalContextBuilder::with_features(&[objective_feature, recompute_feature, mutate_feature])
        .unwrap()
        .build()
        .unwrap();

    let route = RouteContextBuilder::default()
        .with_route(RouteBuilder::default().add_activity(ActivityBuilder::default().build()).build())
        .build();
    let mut solution_ctx = TestInsertionContextBuilder::default().with_routes(vec![route]).build().solution;

    goal.accept_solution_state(&mut solution_ctx);

    assert_eq!(*visited.lock().unwrap(), vec![0, 0]);
    assert!(solution_ctx.routes.iter().all(|route_ctx| !route_ctx.is_stale()));
}
