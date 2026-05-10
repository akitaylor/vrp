use crate::format::problem::*;
use crate::helpers::*;
#[test]
fn can_skip_constraints_check() {
    let problem = Problem {
        plan: Plan {
            jobs: vec![create_delivery_job("job1", (1., 0.)), create_delivery_job("job2", (2., 0.))],
            relations: Some(vec![Relation {
                type_field: RelationType::Any,
                jobs: to_strings(vec!["departure", "job1", "job2"]),
                vehicle_id: "my_vehicle_1".to_string(),
                shift_index: None,
            }]),
            ..create_empty_plan()
        },
        fleet: Fleet {
            vehicles: vec![VehicleType { capacity: vec![1], ..create_default_vehicle_type() }],
            ..create_default_fleet()
        },
        ..create_empty_problem()
    };
    let matrix = create_matrix_from_problem(&problem);

    let solution = solve_with_metaheuristic(problem, Some(vec![matrix]));

    assert!(solution.tours.is_empty());
    assert_eq!(solution.unassigned.unwrap().len(), 2);
}
