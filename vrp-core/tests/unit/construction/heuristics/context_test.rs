use crate::construction::heuristics::{MoveContext, RouteState, SolutionContext, UnassignmentInfo};
use crate::helpers::construction::heuristics::TestInsertionContextBuilder;
use crate::helpers::models::domain::TestGoalContextBuilder;
use crate::helpers::models::problem::{TestSingleBuilder, test_fleet};
use crate::helpers::models::solution::*;
use crate::models::common::Schedule;
use crate::models::{FeatureBuilder, FeatureObjective, FeatureState};
use crate::models::problem::{Job, JobIdDimension};
use std::sync::{Arc, Mutex};

struct NoopObjective;

impl FeatureObjective for NoopObjective {
    fn fitness(&self, _: &crate::construction::heuristics::InsertionContext) -> f64 {
        0.
    }

    fn estimate(&self, _: &MoveContext<'_>) -> f64 {
        0.
    }
}

struct CountSolutionAcceptState {
    calls: Arc<Mutex<usize>>,
}

impl FeatureState for CountSolutionAcceptState {
    fn accept_insertion(&self, _: &mut SolutionContext, _: usize, _: &Job) {}

    fn accept_route_state(&self, _: &mut crate::construction::heuristics::RouteContext) {}

    fn accept_solution_state(&self, solution_ctx: &mut SolutionContext) {
        let _ = solution_ctx.routes.len();
        *self.calls.lock().unwrap() += 1;
    }
}

#[test]
fn can_set_and_get_activity_states_with_different_type_keys() {
    let mut route_state = RouteState::default();

    route_state.set_activity_states::<i8, _>(vec!["key1".to_string()]);
    route_state.set_activity_states::<i16, _>(vec!["key2".to_string()]);
    route_state.set_activity_states::<i32, _>(vec!["key3".to_string()]);
    let result3 = route_state.get_activity_state::<i32, String>(0);
    let result1 = route_state.get_activity_state::<i8, String>(0);
    let result2 = route_state.get_activity_state::<i16, String>(0);
    let result4 = route_state.get_activity_state::<i64, String>(0);

    assert_eq!(result1.unwrap(), "key1");
    assert_eq!(result2.unwrap(), "key2");
    assert_eq!(result3.unwrap(), "key3");
    assert!(result4.is_none());
}

#[test]
fn can_set_and_get_route_state() {
    let mut route_state = RouteState::default();

    route_state.set_tour_state::<(), _>("my_value".to_string());
    let result = route_state.get_tour_state::<(), String>();

    assert_eq!(result.unwrap(), "my_value");
}

#[test]
fn can_set_and_get_empty_route_state() {
    let mut route_state = RouteState::default();

    route_state.set_tour_state::<i8, _>("my_value".to_string());
    let result = route_state.get_tour_state::<i16, String>();

    assert!(result.is_none());
}

#[test]
fn can_use_stale_flag() {
    let mut route_ctx = RouteContextBuilder::default().build();

    assert!(route_ctx.is_stale());
    route_ctx.mark_stale(false);
    assert!(!route_ctx.is_stale());

    let mut route_ctx = RouteContextBuilder::default().build();
    route_ctx.mark_stale(false);
    let _ = route_ctx.as_mut();
    assert!(route_ctx.is_stale());
}

#[test]
fn can_use_debug_fmt_for_insertion_ctx() {
    let insertion_ctx = TestInsertionContextBuilder::default()
        .with_goal(TestGoalContextBuilder::with_transport_feature().build())
        .with_routes(vec![
            RouteContextBuilder::default()
                .with_route(
                    RouteBuilder::default()
                        .add_activity(ActivityBuilder::default().build())
                        .with_vehicle(&test_fleet(), "v1")
                        .build(),
                )
                .build(),
        ])
        .with_unassigned(vec![(TestSingleBuilder::default().build_as_job_ref(), UnassignmentInfo::Unknown)])
        .build();

    let result = format!("{insertion_ctx:#?}");

    println!("{result}");
    assert!(!result.contains("::"));
    assert!(result.contains("tour"));
    assert!(result.contains("vehicle: \"v1\""));
    assert!(result.contains("departure"));
    assert!(result.contains("arrival"));

    assert!(result.contains("unassigned"));
    assert!(result.contains("id: \"single\""));
}

#[test]
fn solution_conversion_restores_insertion_context_first() {
    let calls = Arc::new(Mutex::new(0));
    let objective_feature = FeatureBuilder::default().with_name("objective").with_objective(NoopObjective).build().unwrap();
    let mutate_feature = FeatureBuilder::default()
        .with_name("mutate")
        .with_state(CountSolutionAcceptState { calls: calls.clone() })
        .build()
        .unwrap();
    let goal = TestGoalContextBuilder::empty().add_feature(objective_feature).add_feature(mutate_feature).build();

    let insertion_ctx = TestInsertionContextBuilder::default()
        .with_goal(goal)
        .with_routes(vec![
            RouteContextBuilder::default()
                .with_route(RouteBuilder::default().add_activity(ActivityBuilder::default().build()).build())
                .build(),
        ])
        .build();

    let _: crate::models::Solution = (insertion_ctx, None).into();

    assert_eq!(*calls.lock().unwrap(), 1);
}

#[test]
fn restore_removes_terminal_reload_activity() {
    let reload = TestSingleBuilder::default().id("route_reload_1").build_shared();
    let delivery = TestSingleBuilder::default().id("customer").build_shared();
    let goal = TestGoalContextBuilder::with_transport_feature().build();

    let mut insertion_ctx = TestInsertionContextBuilder::default()
        .with_goal(goal)
        .with_routes(vec![
            RouteContextBuilder::default()
                .with_route(
                    RouteBuilder::default()
                        .with_start(ActivityBuilder::default().job(None).schedule(Schedule::new(0., 0.)).build())
                        .add_activity(
                            ActivityBuilder::with_location(10)
                                .job(Some(delivery))
                                .schedule(Schedule::new(10., 10.))
                                .build(),
                        )
                        .add_activity(
                            ActivityBuilder::with_location(20)
                                .job(Some(reload))
                                .schedule(Schedule::new(20., 20.))
                                .build(),
                        )
                        .with_end(ActivityBuilder::default().job(None).schedule(Schedule::new(30., 30.)).build())
                        .build(),
                )
                .build(),
        ])
        .build();

    insertion_ctx.restore();

    let activities = insertion_ctx.solution.routes[0]
        .route()
        .tour
        .all_activities()
        .filter_map(|activity| activity.retrieve_job())
        .map(|job| job.dimens().get_job_id().cloned().unwrap())
        .collect::<Vec<_>>();

    assert_eq!(activities, vec!["customer".to_string()]);
}
