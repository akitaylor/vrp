use crate::format::problem::*;
use crate::format::solution::*;
use crate::format_time;
use crate::helpers::*;
use std::io::{BufReader, BufWriter};
use std::sync::Arc;
use vrp_core::construction::heuristics::InsertionContext;
use vrp_core::utils::Environment;

fn create_basic_problem(breaks: Option<Vec<VehicleBreak>>) -> Problem {
    Problem {
        plan: Plan {
            jobs: vec![
                create_delivery_job("job1", (1., 0.)),
                create_multi_job(
                    "job2",
                    vec![((2., 0.), 1., vec![1]), ((3., 0.), 1., vec![1])],
                    vec![((4., 0.), 1., vec![2])],
                ),
                create_pickup_job("job3", (5., 0.)),
            ],
            ..create_empty_plan()
        },
        fleet: Fleet {
            vehicles: vec![VehicleType {
                costs: create_default_vehicle_costs(),
                shifts: vec![VehicleShift { breaks, ..create_default_vehicle_shift() }],
                ..create_default_vehicle_type()
            }],
            ..create_default_fleet()
        },
        ..create_empty_problem()
    }
}

fn create_default_breaks() -> Option<Vec<VehicleBreak>> {
    Some(vec![VehicleBreak::Optional {
        time: VehicleOptionalBreakTime::TimeWindow(vec![format_time(5.), format_time(10.)]),
        places: vec![VehicleOptionalBreakPlace { duration: 2.0, location: None, tag: None }],
        policy: None,
    }])
}

fn create_required_breaks() -> Option<Vec<VehicleBreak>> {
    Some(vec![VehicleBreak::Required {
        time: VehicleRequiredBreakTime::ExactTime { earliest: format_time(5.), latest: format_time(10.) },
        duration: 2.0,
    }])
}

fn create_unassigned_jobs(job_ids: &[&str]) -> Option<Vec<UnassignedJob>> {
    Some(
        job_ids
            .iter()
            .map(|job_id| UnassignedJob {
                job_id: job_id.to_string(),
                reasons: vec![UnassignedJobReason {
                    code: "NO_REASON_FOUND".to_string(),
                    description: "unknown".to_string(),
                    details: None,
                }],
            })
            .collect(),
    )
}

fn get_init_solution(problem: Problem, solution: &Solution) -> Result<Solution, GenericError> {
    let environment = Arc::new(Environment::default());
    let matrix = create_matrix_from_problem(&problem);
    let core_problem = Arc::new(
        (problem, vec![matrix]).read_pragmatic().unwrap_or_else(|err| panic!("cannot read core problem: {err:?}")),
    );

    let core_solution = to_core_solution(solution, core_problem.clone(), create_random())?;

    // NOTE: get statistic/tours updated
    let core_solution =
        InsertionContext::new_from_solution(core_problem.clone(), (core_solution, None), environment).into();

    let mut writer = BufWriter::new(Vec::new());
    write_pragmatic(&core_problem, &core_solution, Default::default(), &mut writer)
        .expect("cannot serialize result solution");

    let bytes = writer.into_inner().expect("cannot get bytes from writer");

    deserialize_solution(BufReader::new(bytes.as_slice())).map_err(|err| format!("cannot read solution: {err}").into())
}

fn assert_solution_with_sorted_unassigned(mut result: Solution, mut expected: Solution) {
    result.unassigned.iter_mut().for_each(|jobs| jobs.sort_by(|left, right| left.job_id.cmp(&right.job_id)));
    expected.unassigned.iter_mut().for_each(|jobs| jobs.sort_by(|left, right| left.job_id.cmp(&right.job_id)));

    assert_eq!(result, expected);
}

#[test]
fn can_read_basic_init_solution() {
    let problem = create_basic_problem(create_default_breaks());

    let solution = SolutionBuilder::default()
        .tour(
            TourBuilder::default()
                .stops(vec![
                    StopBuilder::default().coordinate((0., 0.)).schedule_stamp(0., 0.).load(vec![1]).build_departure(),
                    StopBuilder::default()
                        .coordinate((1., 0.))
                        .schedule_stamp(1., 2.)
                        .load(vec![0])
                        .distance(1)
                        .build_single("job1", "delivery"),
                    StopBuilder::default()
                        .coordinate((2., 0.))
                        .schedule_stamp(3., 4.)
                        .load(vec![1])
                        .distance(2)
                        .build_single_tag("job2", "pickup", "p1"),
                    StopBuilder::default()
                        .coordinate((3., 0.))
                        .schedule_stamp(5., 8.)
                        .load(vec![2])
                        .distance(3)
                        .activity(
                            ActivityBuilder::pickup()
                                .job_id("job2")
                                .coordinate((3., 0.))
                                .time_stamp(5., 6.)
                                .tag("p2")
                                .build(),
                        )
                        .activity(ActivityBuilder::break_type().coordinate((3., 0.)).time_stamp(6., 8.).build())
                        .build(),
                    StopBuilder::default()
                        .coordinate((4., 0.))
                        .schedule_stamp(9., 10.)
                        .load(vec![0])
                        .distance(4)
                        .build_single_tag("job2", "delivery", "d1"),
                    StopBuilder::default()
                        .coordinate((0., 0.))
                        .schedule_stamp(14., 14.)
                        .load(vec![0])
                        .distance(8)
                        .build_arrival(),
                ])
                .statistic(StatisticBuilder::default().driving(8).serving(4).break_time(2).build())
                .build(),
        )
        .unassigned(create_unassigned_jobs(&["job3"]))
        .build();

    let result_solution =
        get_init_solution(problem, &solution).unwrap_or_else(|err| panic!("cannot get solution: {err}"));

    assert_eq!(result_solution, solution);
}

#[test]
fn can_ignore_required_break_which_starts_at_latest_offset_boundary() {
    let problem = create_basic_problem(create_required_breaks());

    let solution = SolutionBuilder::default()
        .tour(
            TourBuilder::default()
                .stops(vec![
                    StopBuilder::default().coordinate((0., 0.)).schedule_stamp(0., 0.).load(vec![1]).build_departure(),
                    StopBuilder::default()
                        .coordinate((1., 0.))
                        .schedule_stamp(1., 2.)
                        .load(vec![0])
                        .distance(1)
                        .build_single("job1", "delivery"),
                    StopBuilder::default()
                        .coordinate((2., 0.))
                        .schedule_stamp(10., 12.)
                        .load(vec![0])
                        .distance(2)
                        .activity(ActivityBuilder::break_type().time_stamp(10., 12.).build())
                        .build(),
                    StopBuilder::default()
                        .coordinate((0., 0.))
                        .schedule_stamp(14., 14.)
                        .load(vec![0])
                        .distance(4)
                        .build_arrival(),
                ])
                .statistic(StatisticBuilder::default().driving(4).serving(1).break_time(2).build())
                .build(),
        )
        .unassigned(create_unassigned_jobs(&["job2", "job3"]))
        .build();

    let result_solution =
        get_init_solution(problem, &solution).unwrap_or_else(|err| panic!("cannot get solution: {err}"));

    let expected = SolutionBuilder::default()
        .tour(
            TourBuilder::default()
                .stops(vec![
                    StopBuilder::default().coordinate((0., 0.)).schedule_stamp(0., 0.).load(vec![1]).build_departure(),
                    StopBuilder::default()
                        .coordinate((1., 0.))
                        .schedule_stamp(1., 2.)
                        .load(vec![0])
                        .distance(1)
                        .build_single("job1", "delivery"),
                    StopBuilder::default()
                        .coordinate((0., 0.))
                        .schedule_stamp(3., 3.)
                        .load(vec![0])
                        .distance(2)
                        .build_arrival(),
                ])
                .statistic(StatisticBuilder::default().driving(2).serving(1).build())
                .build(),
        )
        .unassigned(create_unassigned_jobs(&["job2", "job3"]))
        .build();

    assert_solution_with_sorted_unassigned(result_solution, expected);
}

#[test]
fn can_ignore_required_break_on_transit_stop() {
    let problem = create_basic_problem(create_required_breaks());

    let solution = SolutionBuilder::default()
        .tour(
            TourBuilder::default()
                .stops(vec![
                    StopBuilder::default().coordinate((0., 0.)).schedule_stamp(0., 0.).load(vec![1]).build_departure(),
                    StopBuilder::default()
                        .coordinate((1., 0.))
                        .schedule_stamp(1., 2.)
                        .load(vec![0])
                        .distance(1)
                        .build_single("job1", "delivery"),
                    StopBuilder::new_transit().schedule_stamp(10., 12.).load(vec![0]).build_single("break", "break"),
                    StopBuilder::default()
                        .coordinate((0., 0.))
                        .schedule_stamp(14., 14.)
                        .load(vec![0])
                        .distance(2)
                        .build_arrival(),
                ])
                .statistic(StatisticBuilder::default().driving(2).serving(1).break_time(2).build())
                .build(),
        )
        .unassigned(create_unassigned_jobs(&["job2", "job3"]))
        .build();

    let result_solution =
        get_init_solution(problem, &solution).unwrap_or_else(|err| panic!("cannot get solution: {err}"));

    let expected = SolutionBuilder::default()
        .tour(
            TourBuilder::default()
                .stops(vec![
                    StopBuilder::default().coordinate((0., 0.)).schedule_stamp(0., 0.).load(vec![1]).build_departure(),
                    StopBuilder::default()
                        .coordinate((1., 0.))
                        .schedule_stamp(1., 2.)
                        .load(vec![0])
                        .distance(1)
                        .build_single("job1", "delivery"),
                    StopBuilder::default()
                        .coordinate((0., 0.))
                        .schedule_stamp(3., 3.)
                        .load(vec![0])
                        .distance(2)
                        .build_arrival(),
                ])
                .statistic(StatisticBuilder::default().driving(2).serving(1).build())
                .build(),
        )
        .unassigned(create_unassigned_jobs(&["job3", "job2"]))
        .build();

    assert_solution_with_sorted_unassigned(result_solution, expected);
}

#[test]
fn can_ignore_unmatched_activity_and_keep_job_unassigned() {
    let problem = create_basic_problem(None);

    let solution = SolutionBuilder::default()
        .tour(
            TourBuilder::default()
                .stops(vec![
                    StopBuilder::default().coordinate((0., 0.)).schedule_stamp(0., 0.).load(vec![1]).build_departure(),
                    StopBuilder::default()
                        .coordinate((99., 99.))
                        .schedule_stamp(1., 2.)
                        .load(vec![0])
                        .distance(1)
                        .build_single("job1", "delivery"),
                    StopBuilder::default()
                        .coordinate((0., 0.))
                        .schedule_stamp(3., 3.)
                        .load(vec![0])
                        .distance(2)
                        .build_arrival(),
                ])
                .build(),
        )
        .build();

    let result_solution = get_init_solution(problem, &solution).unwrap();

    assert!(
        result_solution
            .tours
            .iter()
            .flat_map(|tour| tour.stops.iter())
            .flat_map(|stop| stop.activities())
            .all(|a| a.job_id != "job1")
    );
    assert!(result_solution.unassigned.unwrap_or_default().iter().any(|job| job.job_id == "job1"));
}

#[test]
fn can_ignore_unknown_vehicle_tour_and_keep_jobs_unassigned() {
    let problem = create_basic_problem(None);

    let solution = SolutionBuilder::default()
        .tour(
            TourBuilder::default()
                .vehicle_id("unknown_vehicle")
                .stops(vec![
                    StopBuilder::default().coordinate((0., 0.)).schedule_stamp(0., 0.).load(vec![1]).build_departure(),
                    StopBuilder::default()
                        .coordinate((1., 0.))
                        .schedule_stamp(1., 2.)
                        .load(vec![0])
                        .distance(1)
                        .build_single("job1", "delivery"),
                    StopBuilder::default()
                        .coordinate((0., 0.))
                        .schedule_stamp(3., 3.)
                        .load(vec![0])
                        .distance(2)
                        .build_arrival(),
                ])
                .build(),
        )
        .build();

    let result_solution = get_init_solution(problem, &solution).unwrap();

    assert!(result_solution.tours.is_empty());
    assert!(result_solution.unassigned.unwrap_or_default().iter().any(|job| job.job_id == "job1"));
}

#[test]
fn can_move_overloaded_init_route_to_unassigned() {
    let mut problem = create_basic_problem(None);
    problem.fleet.vehicles[0].capacity = vec![0];

    let solution = SolutionBuilder::default()
        .tour(
            TourBuilder::default()
                .stops(vec![
                    StopBuilder::default().coordinate((0., 0.)).schedule_stamp(0., 0.).load(vec![1]).build_departure(),
                    StopBuilder::default()
                        .coordinate((1., 0.))
                        .schedule_stamp(1., 2.)
                        .load(vec![0])
                        .distance(1)
                        .build_single("job1", "delivery"),
                    StopBuilder::default()
                        .coordinate((0., 0.))
                        .schedule_stamp(3., 3.)
                        .load(vec![0])
                        .distance(2)
                        .build_arrival(),
                ])
                .build(),
        )
        .unassigned(create_unassigned_jobs(&["job2", "job3"]))
        .build();

    let result_solution = get_init_solution(problem, &solution).unwrap();
    assert!(result_solution.tours.is_empty());
    let mut unassigned =
        result_solution.unassigned.unwrap_or_default().into_iter().map(|job| job.job_id).collect::<Vec<_>>();
    unassigned.sort();

    assert_eq!(unassigned, vec!["job1".to_string(), "job2".to_string(), "job3".to_string()]);
}

#[test]
fn can_handle_empty_tour_error_in_init_solution() {
    let problem = create_basic_problem(create_default_breaks());
    let solution = SolutionBuilder::default()
        .tour(Tour {
            vehicle_id: "my_vehicle_1".to_string(),
            type_id: "my_vehicle".to_string(),
            shift_index: 0,
            stops: Default::default(),
            statistic: Default::default(),
        })
        .build();

    let result_solution = get_init_solution(problem, &solution);

    assert_eq!(result_solution, Err("empty tour in init solution".into()));
}

#[test]
fn can_handle_commute_error_in_init_solution() {
    let problem = create_basic_problem(None);
    let solution = SolutionBuilder::default()
        .tour(
            TourBuilder::default()
                .stops(vec![
                    StopBuilder::default().coordinate((0., 0.)).schedule_stamp(0., 0.).load(vec![1]).build_departure(),
                    StopBuilder::default()
                        .coordinate((1., 0.))
                        .schedule_stamp(1., 2.)
                        .load(vec![0])
                        .distance(1)
                        .activity(
                            ActivityBuilder::delivery()
                                .job_id("job1")
                                .coordinate((1., 0.))
                                .time_stamp(1., 2.)
                                .commute(Commute { forward: None, backward: None })
                                .build(),
                        )
                        .build(),
                ])
                .build(),
        )
        .build();

    let result_solution = get_init_solution(problem, &solution);

    assert_eq!(result_solution, Err("commute property in initial solution is not supported".into()));
}
