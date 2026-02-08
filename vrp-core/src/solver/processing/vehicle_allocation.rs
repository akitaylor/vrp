use super::*;
use crate::construction::enablers::{TotalDistanceTourState, TotalDurationTourState};
use crate::construction::heuristics::*;
use crate::models::Extras;
use crate::models::common::{Cost, TimeSpan};
use crate::models::problem::{Actor, Costs, Job, JobIdDimension, Single, VehicleIdDimension};
use crate::models::solution::Activity;
use rosomaxa::prelude::{Float, UnwrapValue};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::ops::ControlFlow;
use std::sync::Arc;

custom_extra_property!(pub VehicleAllocationSettings typeof VehicleAllocationSettings);

const DEFAULT_MAX_ITERATIONS: usize = 1;
const DEFAULT_ALLOW_UNUSED: bool = true;
const DEFAULT_ALLOW_SWAPS: bool = true;
const DEFAULT_LOG: bool = false;

/// Settings used to control vehicle allocation during search.
#[derive(Clone, Debug)]
pub struct VehicleAllocationSettings {
    /// Max iterations of improvement pass.
    pub max_iterations: usize,
    /// Allow reassigning routes to unused vehicles.
    pub allow_unused: bool,
    /// Allow swapping vehicles between routes.
    pub allow_swaps: bool,
    /// Enable logging for allocation runs.
    pub log: bool,
    /// Interval in generations for search-time allocation.
    pub interval: usize,
    /// Applies allocation to top-k population solutions when > 0.
    pub search_top_k: usize,
    /// Applies allocation to random population samples when > 0.
    pub search_samples: usize,
}

/// Reassigns routes to cheaper compatible vehicles (optionally with swaps).
pub struct VehicleAllocation {
    max_iterations: usize,
    allow_unused: bool,
    allow_swaps: bool,
    log: bool,
}

impl VehicleAllocation {
    /// Creates a new instance with configurable behavior.
    pub fn new(max_iterations: usize, allow_unused: bool, allow_swaps: bool, log: bool) -> Self {
        Self {
            max_iterations: max_iterations.max(1),
            allow_unused,
            allow_swaps,
            log,
        }
    }
}

impl Default for VehicleAllocation {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_ITERATIONS, DEFAULT_ALLOW_UNUSED, DEFAULT_ALLOW_SWAPS, DEFAULT_LOG)
    }
}

impl HeuristicSolutionProcessing for VehicleAllocation {
    type Solution = InsertionContext;

    fn post_process(&self, solution: Self::Solution) -> Self::Solution {
        self.apply(solution)
    }
}

impl VehicleAllocation {
    /// Applies vehicle allocation to a solution context.
    pub fn apply(&self, solution: InsertionContext) -> InsertionContext {
        let mut insertion_ctx = solution;

        if insertion_ctx.solution.routes.is_empty() {
            return insertion_ctx;
        }

        insertion_ctx.problem.goal.accept_solution_state(&mut insertion_ctx.solution);
        let before_cost = insertion_ctx.get_total_cost();
        let before_fitness = collect_fitness(&insertion_ctx);

        if self.log {
            let unused_actors = collect_unused_actors(&insertion_ctx);
            let routes = insertion_ctx.solution.routes.len();
            let unused = unused_actors.len();
            let before = format_cost(before_cost);
            let fitness_before = format_fitness(&before_fitness);
            (insertion_ctx.environment.logger)(
                format!(
                    "vehicle allocation enabled: routes={routes}, unused_actors={unused}, cost_before={before}, fitness_before={fitness_before}"
                )
                .as_str(),
            );
        }

        let mut applied_moves = 0_usize;

        for _ in 0..self.max_iterations {
            let route_infos = build_route_infos(&insertion_ctx);
            if route_infos.is_empty() {
                break;
            }
            let unused_actors = collect_unused_actors(&insertion_ctx);
            let mut candidates: Vec<AllocationMove> = Vec::new();

            if self.allow_unused && !unused_actors.is_empty() {
                for (route_idx, info) in route_infos.iter().enumerate() {
                    let Some(old_cost) = info.cost else { continue };
                    if info.order.is_empty() {
                        continue;
                    }
                    for actor in unused_actors.iter() {
                        let Some((route_ctx, cost)) =
                            build_route_for_actor(&insertion_ctx, actor, info.order.as_slice())
                        else {
                            continue;
                        };

                        if cost + f64::EPSILON < old_cost {
                            let improvement = old_cost - cost;
                            candidates.push(AllocationMove::Reassign {
                                route_idx,
                                new_route: route_ctx,
                                improvement,
                            });
                        }
                    }
                }
            }

            if self.allow_swaps {
                for left in 0..route_infos.len() {
                    for right in (left + 1)..route_infos.len() {
                    let left_info = &route_infos[left];
                    let right_info = &route_infos[right];
                    if left_info.order.is_empty() || right_info.order.is_empty() {
                        continue;
                    }

                        let Some((left_route, left_cost)) =
                            build_route_for_actor(&insertion_ctx, &right_info.actor, left_info.order.as_slice())
                        else {
                            continue;
                        };
                        let Some((right_route, right_cost)) =
                            build_route_for_actor(&insertion_ctx, &left_info.actor, right_info.order.as_slice())
                        else {
                            continue;
                        };

                        let (Some(left_old_cost), Some(right_old_cost)) = (left_info.cost, right_info.cost) else {
                            continue;
                        };
                        let old_cost = left_old_cost + right_old_cost;
                        let new_cost = left_cost + right_cost;

                        if new_cost + f64::EPSILON < old_cost {
                            let improvement = old_cost - new_cost;
                            candidates.push(AllocationMove::Swap {
                                left_idx: left,
                                right_idx: right,
                                left_route,
                                right_route,
                                improvement,
                            });
                        }
                    }
                }
            }

            if candidates.is_empty() {
                break;
            }

            let candidate_count = candidates.len();
            candidates.sort_by(|left, right| right.improvement().total_cmp(&left.improvement()));

            let mut applied = false;
            let mut rejected_due_fitness = 0_usize;

            for candidate in candidates {
                let mut candidate_ctx = insertion_ctx.deep_copy();
                apply_move_to_context(&mut candidate_ctx, candidate);
                candidate_ctx.problem.goal.accept_solution_state(&mut candidate_ctx.solution);

                let ordering = candidate_ctx.problem.goal.total_order(&candidate_ctx, &insertion_ctx);
                if ordering == Ordering::Greater {
                    rejected_due_fitness += 1;
                    continue;
                }

                insertion_ctx = candidate_ctx;
                applied_moves += 1;
                applied = true;
                break;
            }

            if !applied {
                if self.log {
                    (insertion_ctx.environment.logger)(
                        format!(
                            "vehicle allocation rejected: candidates={}, rejected_due_fitness={}",
                            candidate_count, rejected_due_fitness
                        )
                        .as_str(),
                    );
                }
                break;
            }
        }

        insertion_ctx.problem.goal.accept_solution_state(&mut insertion_ctx.solution);
        let after_cost = insertion_ctx.get_total_cost();
        let after_fitness = collect_fitness(&insertion_ctx);

        if self.log {
            let improvement = format_improvement(before_cost, after_cost);
            let before = format_cost(before_cost);
            let after = format_cost(after_cost);
            let fitness_before = format_fitness(&before_fitness);
            let fitness_after = format_fitness(&after_fitness);
            (insertion_ctx.environment.logger)(
                format!(
                    "vehicle allocation done: applied_moves={applied_moves}, improvement={improvement}, cost_before={before}, cost_after={after}, fitness_before={fitness_before}, fitness_after={fitness_after}"
                )
                .as_str(),
            );
        }

        insertion_ctx
    }
}

struct RouteInfo {
    actor: Arc<Actor>,
    order: Vec<(Job, Arc<Single>)>,
    cost: Option<Cost>,
}

enum AllocationMove {
    Reassign {
        route_idx: usize,
        new_route: RouteContext,
        improvement: Cost,
    },
    Swap {
        left_idx: usize,
        right_idx: usize,
        left_route: RouteContext,
        right_route: RouteContext,
        improvement: Cost,
    },
}

impl AllocationMove {
    fn improvement(&self) -> Cost {
        match self {
            AllocationMove::Reassign { improvement, .. } => *improvement,
            AllocationMove::Swap { improvement, .. } => *improvement,
        }
    }
}

fn build_route_infos(insertion_ctx: &InsertionContext) -> Vec<RouteInfo> {
    insertion_ctx
        .solution
        .routes
        .iter()
        .map(|route_ctx| {
            let order = collect_route_order(route_ctx);
            let cost = get_route_cost(route_ctx);

            RouteInfo { actor: route_ctx.route().actor.clone(), order, cost }
        })
        .collect()
}

fn collect_unused_actors(insertion_ctx: &InsertionContext) -> Vec<Arc<Actor>> {
    let used = insertion_ctx
        .solution
        .routes
        .iter()
        .map(|route_ctx| route_ctx.route().actor.clone())
        .collect::<HashSet<_>>();

    insertion_ctx.problem.fleet.actors.iter().filter(|actor| !used.contains(*actor)).cloned().collect()
}

fn apply_move_to_context(insertion_ctx: &mut InsertionContext, allocation: AllocationMove) {
    match allocation {
        AllocationMove::Reassign { route_idx, new_route, .. } => {
            let old_route = std::mem::replace(&mut insertion_ctx.solution.routes[route_idx], new_route);
            insertion_ctx.solution.registry.free_route(old_route);
            insertion_ctx.solution.registry.use_route(&insertion_ctx.solution.routes[route_idx]);
        }
        AllocationMove::Swap { left_idx, right_idx, left_route, right_route, .. } => {
            let old_left = std::mem::replace(&mut insertion_ctx.solution.routes[left_idx], left_route);
            let old_right = std::mem::replace(&mut insertion_ctx.solution.routes[right_idx], right_route);

            insertion_ctx.solution.registry.free_route(old_left);
            insertion_ctx.solution.registry.free_route(old_right);
            insertion_ctx.solution.registry.use_route(&insertion_ctx.solution.routes[left_idx]);
            insertion_ctx.solution.registry.use_route(&insertion_ctx.solution.routes[right_idx]);
        }
    }
}

fn collect_route_order(route_ctx: &RouteContext) -> Vec<(Job, Arc<Single>)> {
    route_ctx
        .route()
        .tour
        .all_activities()
        .filter_map(|activity| activity.job.as_ref().map(|single| (single, activity)))
        .filter(|(single, activity)| is_activity_to_single_match(activity, single))
        .filter_map(|(single, activity)| activity.retrieve_job().map(|job| (job, single.clone())))
        .filter(|(job, _)| !is_conditional_job(job))
        .collect()
}

fn is_conditional_job(job: &Job) -> bool {
    let Some(job_id) = job.dimens().get_job_id() else { return false };
    job.dimens().get_vehicle_id().is_some()
        && (job_id.contains("_break_") || job_id.contains("_reload_") || job_id.contains("_recharge_"))
}

fn build_route_for_actor(
    insertion_ctx: &InsertionContext,
    actor: &Arc<Actor>,
    order: &[(Job, Arc<Single>)],
) -> Option<(RouteContext, Cost)> {
    if order.is_empty() {
        return None;
    }

    let mut new_ctx = InsertionContext::new_empty(insertion_ctx.problem.clone(), insertion_ctx.environment.clone());
    let route_ctx = new_ctx.solution.registry.get_route(actor)?;
    new_ctx.solution.routes.push(route_ctx);
    let route_idx = new_ctx.solution.routes.len() - 1;

    let goal = new_ctx.problem.goal.clone();
    let position = InsertionPosition::Last;
    let leg_selection = LegSelection::Exhaustive;
    let result_selector = BestResultSelector::default();

    let mut synchronized_jobs: HashMap<Job, Vec<Arc<Single>>> = HashMap::default();
    let mut invalid_multi_job_ids: HashSet<Job> = HashSet::default();

    for (job, single) in order.iter() {
        let is_already_processed = synchronized_jobs.contains_key(job) && job.as_single().is_some();
        let is_invalid_multi_job = invalid_multi_job_ids.contains(job);

        if is_already_processed || is_invalid_multi_job {
            continue;
        }

        let eval_ctx =
            EvaluationContext { goal: &goal, job, leg_selection: &leg_selection, result_selector: &result_selector };
        let route_ctx = &new_ctx.solution.routes[route_idx];

        let insertion_result = eval_single_constraint_in_route(
            &new_ctx,
            &eval_ctx,
            route_ctx,
            single,
            position,
            Default::default(),
            None,
        );

        match insertion_result {
            InsertionResult::Success(success) => {
                apply_insertion_success(&mut new_ctx, success);
                synchronized_jobs.entry(job.clone()).or_insert_with(Vec::default).push(single.clone());
            }
            InsertionResult::Failure(_) if job.as_multi().is_some() => {
                invalid_multi_job_ids.insert(job.clone());
            }
            InsertionResult::Failure(_) => return None,
        }
    }

    if !invalid_multi_job_ids.is_empty() {
        return None;
    }

    if !validate_multi_jobs(&synchronized_jobs) {
        return None;
    }

    new_ctx.problem.goal.accept_solution_state(&mut new_ctx.solution);

    let route_ctx = new_ctx.solution.routes.into_iter().find(|route_ctx| route_ctx.route().actor == *actor)?;
    let cost = get_route_cost(&route_ctx)?;

    Some((route_ctx, cost))
}

fn validate_multi_jobs(synchronized_jobs: &HashMap<Job, Vec<Arc<Single>>>) -> bool {
    synchronized_jobs.iter().all(|(job, singles)| {
        match job {
            Job::Single(_) => true,
            Job::Multi(multi) => {
                multi.jobs.len() == singles.len() && compare_singles(multi, singles.as_slice())
            }
        }
    })
}

fn compare_singles(multi: &crate::models::problem::Multi, singles: &[Arc<Single>]) -> bool {
    let job_map = multi
        .jobs
        .iter()
        .enumerate()
        .map(|(idx, single)| (Job::Single(single.clone()), idx))
        .collect::<HashMap<_, _>>();

    let permutation =
        singles.iter().filter_map(|single| job_map.get(&Job::Single(single.clone())).cloned()).collect::<Vec<_>>();

    multi.validate(permutation.as_slice())
}

fn get_route_cost(route_ctx: &RouteContext) -> Option<Cost> {
    let actor = &route_ctx.route().actor;
    let distance = route_ctx.state().get_total_distance().copied()?;
    let duration = route_ctx.state().get_total_duration().copied()?;

    Some(get_cost(&actor.vehicle.costs, distance, duration) + get_cost(&actor.driver.costs, distance, duration))
}

fn get_cost(costs: &Costs, distance: Float, duration: Float) -> Cost {
    costs.fixed
        + costs.per_distance * distance
        + costs.per_driving_time.max(costs.per_service_time).max(costs.per_waiting_time) * duration
}

fn format_cost(cost: Option<Cost>) -> String {
    cost.map(|value| format!("{value:.3}")).unwrap_or_else(|| "n/a".to_string())
}

fn format_improvement(before: Option<Cost>, after: Option<Cost>) -> String {
    match (before, after) {
        (Some(before), Some(after)) => format!("{:.3}", before - after),
        _ => "n/a".to_string(),
    }
}

fn collect_fitness(insertion_ctx: &InsertionContext) -> Vec<Float> {
    insertion_ctx.problem.goal.fitness(insertion_ctx).collect()
}

fn format_fitness(values: &[Float]) -> String {
    let mut out = String::from("[");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("{value:.3}"));
    }
    out.push(']');
    out
}

fn is_activity_to_single_match(activity: &Activity, single: &Single) -> bool {
    single
        .places
        .iter()
        .try_fold(false, |_, place| {
            let is_same_duration = activity.place.duration == place.duration;
            let is_same_location = place.location.is_none_or(|location| location == activity.place.location);
            let is_same_time_window = place.times.iter().any(|time| match time {
                TimeSpan::Window(tw) => activity.place.time == *tw,
                TimeSpan::Offset(_) => false,
            });

            if is_same_duration && is_same_location && is_same_time_window {
                ControlFlow::Break(true)
            } else {
                ControlFlow::Continue(false)
            }
        })
        .unwrap_value()
}
