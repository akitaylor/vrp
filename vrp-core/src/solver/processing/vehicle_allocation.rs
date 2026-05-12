use super::*;
#[cfg(test)]
#[path = "../../../tests/unit/solver/processing/vehicle_allocation_test.rs"]
mod vehicle_allocation_test;

use crate::construction::enablers::{
    TotalDurationTourState, advance_departure_time, create_reserved_times_fn, optimize_reserved_times_schedule,
    update_route_schedule,
};
use crate::construction::features::{
    JobSkills, JobSkillsBitset, JobSkillsBitsetDimension, JobSkillsDimension, MaxVehicleLoadTourState,
    VehicleOvertimeDimension, VehicleSkillsBitset, VehicleSkillsBitsetDimension, VehicleSkillsDimension,
};
use crate::construction::heuristics::*;
use crate::models::Extras;
use crate::models::common::{Cost, TimeSpan, TimeWindow};
use crate::models::problem::{
    ActivityCost, Actor, Job, JobIdDimension, Single, TransportCost, TravelTime, VehicleIdDimension,
};
use crate::models::solution::Activity;
use rosomaxa::prelude::{Float, UnwrapValue};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::Instant;

custom_extra_property!(pub VehicleAllocationSettings typeof VehicleAllocationSettings);

const DEFAULT_MAX_ITERATIONS: usize = 1;
const DEFAULT_ALLOW_UNUSED: bool = true;
const DEFAULT_ALLOW_SWAPS: bool = true;
const DEFAULT_LOG: bool = false;
const ESTIMATED_IMPROVEMENT_EPSILON: Cost = 1e-3;
const MAX_EXACT_CANDIDATES_PER_ITERATION: usize = 32;

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
        Self { max_iterations: max_iterations.max(1), allow_unused, allow_swaps, log }
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
        self.apply_with_recreate(solution)
    }

    /// Applies vehicle allocation using exact actor replay plus conditional-job repair.
    pub fn apply_with_recreate(&self, solution: InsertionContext) -> InsertionContext {
        let mut insertion_ctx = solution;

        if insertion_ctx.solution.routes.is_empty() {
            return insertion_ctx;
        }

        repair_conditional_jobs(&mut insertion_ctx);
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
            (insertion_ctx.environment.logger)(
                "vehicle allocation config: insertion_position=any, cost_mode=transport+activity(per-leg)".as_ref(),
            );
        }

        let mut applied_moves = 0_usize;
        let started_at = Instant::now();

        for _ in 0..self.max_iterations {
            let route_infos = build_route_infos(
                &insertion_ctx,
                insertion_ctx.problem.activity.as_ref(),
                insertion_ctx.problem.transport.as_ref(),
            );
            if route_infos.is_empty() {
                break;
            }

            let unused_actors = collect_unused_actors(&insertion_ctx);
            let mut stats = AllocationStats::new(route_infos.len(), unused_actors.len());
            stats.routes_missing_cost = route_infos.iter().filter(|info| info.cost.is_none()).count();
            let best_candidate = find_best_exact_candidate(
                &insertion_ctx,
                route_infos.as_slice(),
                unused_actors.as_slice(),
                self.allow_unused,
                self.allow_swaps,
                &mut stats,
            );

            if self.log {
                (insertion_ctx.environment.logger)(
                    format!(
                        "vehicle allocation stats: routes={}, routes_missing_cost={}, unused_actors={}, reassign_attempts={}, reassign_build_ok={}, reassign_improving={}, reassign_estimate_rejected={}, swap_attempts={}, swap_build_ok={}, swap_improving={}, swap_estimate_rejected={}, estimated_candidates={}, exact_attempts={}, exact_improving={}, exact_applied={}, signature_rejected={}, pending_break_rejected={}, capacity_rejected={}, prefilter_attempts={}, prefilter_vehicle_id={}, prefilter_skills={}, prefilter_time={}, build_failures={}",
                        stats.routes_total,
                        stats.routes_missing_cost,
                        stats.unused_actors,
                        stats.reassign_attempts,
                        stats.reassign_build_ok,
                        stats.reassign_improving,
                        stats.reassign_estimate_rejected,
                        stats.swap_attempts,
                        stats.swap_build_ok,
                        stats.swap_improving,
                        stats.swap_estimate_rejected,
                        stats.estimated_candidates,
                        stats.exact_attempts,
                        stats.exact_improving,
                         stats.exact_applied,
                         stats.signature_rejected,
                         stats.pending_break_rejected,
                         stats.capacity_rejected,
                         stats.prefilter_attempts,
                        stats.prefilter_vehicle_id,
                        stats.prefilter_skills,
                        stats.prefilter_time,
                        stats.build_failures.format()
                    )
                    .as_str(),
                );
            }

            let Some(candidate) = best_candidate else {
                break;
            };

            insertion_ctx = candidate.insertion_ctx;
            applied_moves += 1;
            stats.exact_applied += 1;

            log_capacity_violations(
                &insertion_ctx,
                "vehicle allocation warning: capacity violation detected after applying allocation iteration",
            );
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
            (insertion_ctx.environment.logger)(
                format!("vehicle allocation timing: elapsed_ms={}", started_at.elapsed().as_millis()).as_str(),
            );
        }

        insertion_ctx
    }
}

fn get_capacity_violations(insertion_ctx: &InsertionContext) -> Vec<(String, Float)> {
    insertion_ctx
        .solution
        .routes
        .iter()
        .filter_map(|route_ctx| {
            route_ctx
                .state()
                .get_max_vehicle_load()
                .copied()
                .filter(|load_ratio| *load_ratio > 1. + Float::EPSILON)
                .map(|load_ratio| (get_actor_id(&route_ctx.route().actor), load_ratio))
        })
        .collect()
}

fn log_capacity_violations(insertion_ctx: &InsertionContext, prefix: &str) {
    let violations = get_capacity_violations(insertion_ctx);
    if violations.is_empty() {
        return;
    }

    let details =
        violations.iter().map(|(actor_id, ratio)| format!("{actor_id}:{ratio:.3}")).collect::<Vec<_>>().join(", ");

    (insertion_ctx.environment.logger)(format!("{prefix}: {details}").as_str());
}

struct RouteInfo {
    actor: Arc<Actor>,
    order: Vec<(Job, Arc<Single>)>,
    cost: Option<Cost>,
}

struct ExactAllocationCandidate {
    insertion_ctx: InsertionContext,
    cost: Option<Cost>,
}

#[derive(Copy, Clone, Debug)]
enum PrefilterReason {
    VehicleId,
    Skills,
    TimeWindow,
}

struct PrefilterFailure {
    reason: PrefilterReason,
}

#[derive(Default)]
struct BuildRouteFailureCounters {
    registry_missing: usize,
    insertion_failure: usize,
    invalid_multi_job: usize,
    multi_job_mismatch: usize,
    cost_missing: usize,
}

impl BuildRouteFailureCounters {
    fn format(&self) -> String {
        format!(
            "registry_missing={}, insertion_failure={}, invalid_multi_job={}, multi_job_mismatch={}, cost_missing={}",
            self.registry_missing,
            self.insertion_failure,
            self.invalid_multi_job,
            self.multi_job_mismatch,
            self.cost_missing
        )
    }
}

struct AllocationStats {
    routes_total: usize,
    routes_missing_cost: usize,
    unused_actors: usize,
    reassign_attempts: usize,
    reassign_build_ok: usize,
    reassign_improving: usize,
    reassign_estimate_rejected: usize,
    swap_attempts: usize,
    swap_build_ok: usize,
    swap_improving: usize,
    swap_estimate_rejected: usize,
    estimated_candidates: usize,
    exact_attempts: usize,
    exact_improving: usize,
    exact_applied: usize,
    signature_rejected: usize,
    pending_break_rejected: usize,
    capacity_rejected: usize,
    prefilter_attempts: usize,
    prefilter_vehicle_id: usize,
    prefilter_skills: usize,
    prefilter_time: usize,
    build_failures: BuildRouteFailureCounters,
}

impl AllocationStats {
    fn new(routes_total: usize, unused_actors: usize) -> Self {
        Self {
            routes_total,
            routes_missing_cost: 0,
            unused_actors,
            reassign_attempts: 0,
            reassign_build_ok: 0,
            reassign_improving: 0,
            reassign_estimate_rejected: 0,
            swap_attempts: 0,
            swap_build_ok: 0,
            swap_improving: 0,
            swap_estimate_rejected: 0,
            estimated_candidates: 0,
            exact_attempts: 0,
            exact_improving: 0,
            exact_applied: 0,
            signature_rejected: 0,
            pending_break_rejected: 0,
            capacity_rejected: 0,
            prefilter_attempts: 0,
            prefilter_vehicle_id: 0,
            prefilter_skills: 0,
            prefilter_time: 0,
            build_failures: BuildRouteFailureCounters::default(),
        }
    }
}

#[derive(Clone)]
enum RepairAllocationMove {
    Reassign { source_actor: Arc<Actor>, candidate_actor: Arc<Actor>, force_actor: bool, improvement: Cost },
    Swap { left_actor: Arc<Actor>, right_actor: Arc<Actor>, improvement: Cost },
}

impl RepairAllocationMove {
    fn improvement(&self) -> Cost {
        match self {
            Self::Reassign { improvement, .. } => *improvement,
            Self::Swap { improvement, .. } => *improvement,
        }
    }
}

enum AllocationMove {
    Reassign { route_idx: usize, new_route: RouteContext },
    Swap { left_idx: usize, right_idx: usize, left_route: RouteContext, right_route: RouteContext },
}

fn build_route_infos(
    insertion_ctx: &InsertionContext,
    activity: &dyn ActivityCost,
    transport: &dyn TransportCost,
) -> Vec<RouteInfo> {
    insertion_ctx
        .solution
        .routes
        .iter()
        .map(|route_ctx| {
            let order = collect_route_order(route_ctx);
            let cost = get_route_cost(route_ctx, activity, transport);

            RouteInfo { actor: route_ctx.route().actor.clone(), order, cost }
        })
        .collect()
}

fn collect_unused_actors(insertion_ctx: &InsertionContext) -> Vec<Arc<Actor>> {
    let used =
        insertion_ctx.solution.routes.iter().map(|route_ctx| route_ctx.route().actor.clone()).collect::<HashSet<_>>();

    insertion_ctx.problem.fleet.actors.iter().filter(|actor| !used.contains(*actor)).cloned().collect()
}

fn find_best_exact_candidate(
    insertion_ctx: &InsertionContext,
    route_infos: &[RouteInfo],
    unused_actors: &[Arc<Actor>],
    allow_unused: bool,
    allow_swaps: bool,
    stats: &mut AllocationStats,
) -> Option<ExactAllocationCandidate> {
    let baseline_signature = collect_job_signature(insertion_ctx);
    let mut best_candidate = None;
    let mut estimated_candidates = Vec::new();

    for (route_idx, route_ctx) in insertion_ctx.solution.routes.iter().enumerate() {
        let info = &route_infos[route_idx];
        let Some(old_cost) = info.cost else { continue };
        if info.order.is_empty() {
            continue;
        }

        if !allow_unused {
            continue;
        }

        for actor in unused_actors.iter() {
            if *actor == info.actor {
                continue;
            }

            stats.reassign_attempts += 1;
            stats.prefilter_attempts += 1;
            if let Some(failure) = prefilter_actor_for_order(actor, info.order.as_slice()) {
                match failure.reason {
                    PrefilterReason::VehicleId => stats.prefilter_vehicle_id += 1,
                    PrefilterReason::Skills => stats.prefilter_skills += 1,
                    PrefilterReason::TimeWindow => stats.prefilter_time += 1,
                }
                continue;
            }

            let estimated_improvement = estimate_reassign_improvement(route_ctx, actor, old_cost, insertion_ctx);
            if estimated_improvement <= ESTIMATED_IMPROVEMENT_EPSILON {
                stats.reassign_estimate_rejected += 1;
                continue;
            }

            estimated_candidates.push(RepairAllocationMove::Reassign {
                source_actor: info.actor.clone(),
                candidate_actor: actor.clone(),
                force_actor: true,
                improvement: estimated_improvement,
            });
        }
    }

    if allow_swaps {
        for left in 0..route_infos.len() {
            for right in (left + 1)..route_infos.len() {
                let left_info = &route_infos[left];
                let right_info = &route_infos[right];
                if left_info.order.is_empty() || right_info.order.is_empty() {
                    continue;
                }

                stats.swap_attempts += 1;

                stats.prefilter_attempts += 1;
                if let Some(failure) = prefilter_actor_for_order(&right_info.actor, left_info.order.as_slice()) {
                    match failure.reason {
                        PrefilterReason::VehicleId => stats.prefilter_vehicle_id += 1,
                        PrefilterReason::Skills => stats.prefilter_skills += 1,
                        PrefilterReason::TimeWindow => stats.prefilter_time += 1,
                    }
                    continue;
                }

                stats.prefilter_attempts += 1;
                if let Some(failure) = prefilter_actor_for_order(&left_info.actor, right_info.order.as_slice()) {
                    match failure.reason {
                        PrefilterReason::VehicleId => stats.prefilter_vehicle_id += 1,
                        PrefilterReason::Skills => stats.prefilter_skills += 1,
                        PrefilterReason::TimeWindow => stats.prefilter_time += 1,
                    }
                    continue;
                }

                let estimated_improvement = estimate_swap_improvement(left_info, right_info, insertion_ctx);
                if estimated_improvement <= ESTIMATED_IMPROVEMENT_EPSILON {
                    stats.swap_estimate_rejected += 1;
                    continue;
                }

                estimated_candidates.push(RepairAllocationMove::Swap {
                    left_actor: left_info.actor.clone(),
                    right_actor: right_info.actor.clone(),
                    improvement: estimated_improvement,
                });
            }
        }
    }

    estimated_candidates.sort_by(|left, right| {
        right.improvement().total_cmp(&left.improvement()).then_with(|| match (left, right) {
            (RepairAllocationMove::Reassign { .. }, RepairAllocationMove::Swap { .. }) => Ordering::Less,
            (RepairAllocationMove::Swap { .. }, RepairAllocationMove::Reassign { .. }) => Ordering::Greater,
            _ => Ordering::Equal,
        })
    });
    stats.estimated_candidates = estimated_candidates.len();

    for allocation in estimated_candidates.into_iter().take(MAX_EXACT_CANDIDATES_PER_ITERATION) {
        stats.exact_attempts += 1;
        if let Some(candidate) =
            evaluate_exact_candidate(insertion_ctx, &allocation, baseline_signature.as_slice(), stats)
        {
            match allocation {
                RepairAllocationMove::Reassign { .. } => {
                    stats.reassign_build_ok += 1;
                    stats.reassign_improving += 1;
                }
                RepairAllocationMove::Swap { .. } => {
                    stats.swap_build_ok += 1;
                    stats.swap_improving += 1;
                }
            }
            update_best_candidate(&mut best_candidate, candidate);
        }
    }

    best_candidate
}

fn update_best_candidate(best_candidate: &mut Option<ExactAllocationCandidate>, candidate: ExactAllocationCandidate) {
    match best_candidate {
        Some(best) => {
            let candidate_cost = candidate.cost.unwrap_or(Float::MAX);
            let best_cost = best.cost.unwrap_or(Float::MAX);

            if candidate_cost + Float::EPSILON < best_cost
                || ((candidate_cost - best_cost).abs() <= Float::EPSILON
                    && candidate.insertion_ctx.problem.goal.total_order(&candidate.insertion_ctx, &best.insertion_ctx)
                        == Ordering::Less)
            {
                *best_candidate = Some(candidate);
            }
        }
        None => *best_candidate = Some(candidate),
    }
}

fn estimate_reassign_improvement(
    route_ctx: &RouteContext,
    actor: &Arc<Actor>,
    old_cost: Cost,
    insertion_ctx: &InsertionContext,
) -> Cost {
    estimate_route_cost_for_actor(
        route_ctx,
        actor,
        insertion_ctx.problem.activity.as_ref(),
        insertion_ctx.problem.transport.as_ref(),
    )
    .map(|new_cost| old_cost - new_cost)
    .unwrap_or_default()
}

fn estimate_swap_improvement(left: &RouteInfo, right: &RouteInfo, insertion_ctx: &InsertionContext) -> Cost {
    let (Some(left_old), Some(right_old)) = (left.cost, right.cost) else { return 0. };
    let activity = insertion_ctx.problem.activity.as_ref();
    let transport = insertion_ctx.problem.transport.as_ref();

    let left_route_ctx = insertion_ctx.solution.routes.iter().find(|route_ctx| route_ctx.route().actor == left.actor);
    let right_route_ctx = insertion_ctx.solution.routes.iter().find(|route_ctx| route_ctx.route().actor == right.actor);
    let (Some(left_route_ctx), Some(right_route_ctx)) = (left_route_ctx, right_route_ctx) else { return 0. };

    let left_new = estimate_route_cost_for_actor(left_route_ctx, &right.actor, activity, transport);
    let right_new = estimate_route_cost_for_actor(right_route_ctx, &left.actor, activity, transport);

    left_new
        .zip(right_new)
        .map(|(left_new, right_new)| (left_old + right_old) - (left_new + right_new))
        .unwrap_or_default()
}

fn evaluate_exact_candidate(
    insertion_ctx: &InsertionContext,
    allocation: &RepairAllocationMove,
    baseline_signature: &[(String, usize)],
    stats: &mut AllocationStats,
) -> Option<ExactAllocationCandidate> {
    let baseline_pending_signature = collect_pending_job_signature(insertion_ctx);
    let mut candidate_ctx = try_apply_exact_swap(insertion_ctx, allocation)?;
    candidate_ctx.problem.goal.accept_solution_state(&mut candidate_ctx.solution);
    let baseline_cost = insertion_ctx.get_total_cost()?;
    let candidate_cost = candidate_ctx.get_total_cost()?;

    if collect_job_signature(&candidate_ctx) != baseline_signature {
        stats.signature_rejected += 1;
        return None;
    }

    if collect_pending_job_signature(&candidate_ctx) != baseline_pending_signature {
        stats.signature_rejected += 1;
        return None;
    }

    if has_pending_breaks_for_used_vehicles(&candidate_ctx) {
        stats.pending_break_rejected += 1;
        return None;
    }

    if !get_capacity_violations(&candidate_ctx).is_empty() {
        stats.capacity_rejected += 1;
        return None;
    }

    if candidate_cost + f64::EPSILON >= baseline_cost {
        return None;
    }

    stats.exact_improving += 1;
    Some(ExactAllocationCandidate { insertion_ctx: candidate_ctx, cost: Some(candidate_cost) })
}

fn try_apply_exact_swap(
    insertion_ctx: &InsertionContext,
    allocation: &RepairAllocationMove,
) -> Option<InsertionContext> {
    match allocation {
        RepairAllocationMove::Reassign { source_actor, candidate_actor, force_actor, .. } if *force_actor => {
            let route_idx =
                insertion_ctx.solution.routes.iter().position(|route_ctx| route_ctx.route().actor == *source_actor)?;
            let new_route =
                replay_route_with_actor(insertion_ctx, &insertion_ctx.solution.routes[route_idx], candidate_actor)?;

            let mut candidate_ctx = insertion_ctx.deep_copy();
            apply_move_to_context(&mut candidate_ctx, AllocationMove::Reassign { route_idx, new_route });
            candidate_ctx.restore();
            try_insert_conditional_jobs(&mut candidate_ctx);
            finalize_insertion_ctx(&mut candidate_ctx);

            Some(candidate_ctx)
        }
        RepairAllocationMove::Swap { left_actor, right_actor, .. } => {
            let left_idx =
                insertion_ctx.solution.routes.iter().position(|route_ctx| route_ctx.route().actor == *left_actor)?;
            let right_idx =
                insertion_ctx.solution.routes.iter().position(|route_ctx| route_ctx.route().actor == *right_actor)?;
            let left_route =
                replay_route_with_actor(insertion_ctx, &insertion_ctx.solution.routes[left_idx], right_actor)?;
            let right_route =
                replay_route_with_actor(insertion_ctx, &insertion_ctx.solution.routes[right_idx], left_actor)?;

            let mut candidate_ctx = insertion_ctx.deep_copy();
            apply_move_to_context(
                &mut candidate_ctx,
                AllocationMove::Swap { left_idx, right_idx, left_route, right_route },
            );
            candidate_ctx.restore();
            try_insert_conditional_jobs(&mut candidate_ctx);
            finalize_insertion_ctx(&mut candidate_ctx);

            Some(candidate_ctx)
        }
        _ => None,
    }
}

fn replay_route_with_actor(
    insertion_ctx: &InsertionContext,
    source_route_ctx: &RouteContext,
    actor: &Arc<Actor>,
) -> Option<RouteContext> {
    let mut route = source_route_ctx.route().deep_copy();
    route.actor = actor.clone();

    let mut route_ctx = RouteContext::new_with_state(route, RouteState::default());
    update_route_schedule(
        &mut route_ctx,
        insertion_ctx.problem.activity.as_ref(),
        insertion_ctx.problem.transport.as_ref(),
    );
    insertion_ctx.problem.goal.accept_route_state(&mut route_ctx);
    advance_departure_time(
        &mut route_ctx,
        insertion_ctx.problem.activity.as_ref(),
        insertion_ctx.problem.transport.as_ref(),
        true,
    );
    insertion_ctx.problem.goal.accept_route_state(&mut route_ctx);

    if let Some(reserved_times) = insertion_ctx.problem.extras.get_reserved_times().map(|times| times.as_ref().clone())
    {
        if let Ok(reserved_times_fn) = create_reserved_times_fn(reserved_times) {
            optimize_reserved_times_schedule(route_ctx.route_mut(), &reserved_times_fn);
            route_ctx.mark_stale(false);
        }
    }

    Some(route_ctx)
}

fn estimate_route_cost_for_actor(
    route_ctx: &RouteContext,
    actor: &Arc<Actor>,
    activity: &dyn ActivityCost,
    transport: &dyn TransportCost,
) -> Option<Cost> {
    let route = route_ctx.route();
    let temp_route = crate::models::solution::Route { actor: actor.clone(), tour: route.tour.deep_copy() };
    let mut total = Cost::default();

    if temp_route.tour.has_jobs() {
        total += actor.vehicle.costs.fixed + actor.driver.costs.fixed;
    }

    let mut activities = temp_route.tour.all_activities();
    let start = activities.next()?;
    let mut prev = start;

    for act in activities {
        let travel_time = TravelTime::Departure(prev.schedule.departure);
        total += transport.cost(&temp_route, prev.place.location, act.place.location, travel_time);

        if act.job.is_some() {
            total += activity.cost(&temp_route, act, act.schedule.arrival);
        }

        prev = act;
    }

    if let Some(overtime) = actor.vehicle.dimens.get_vehicle_overtime() {
        let duration = route_ctx.state().get_total_duration().copied().unwrap_or(0.);
        total += overtime.penalty(duration);
    }

    Some(total)
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

fn prefilter_actor_for_order(actor: &Actor, order: &[(Job, Arc<Single>)]) -> Option<PrefilterFailure> {
    let date =
        actor.detail.start.as_ref().map(|start| start.time.to_time_window().start).unwrap_or(actor.detail.time.start);

    for (job, _) in order.iter() {
        if check_vehicle_id_mismatch(actor, job).is_some() {
            return Some(PrefilterFailure { reason: PrefilterReason::VehicleId });
        }
        if check_skill_mismatch(actor, job).is_some() {
            return Some(PrefilterFailure { reason: PrefilterReason::Skills });
        }
        if check_time_window_mismatch(actor, job, date).is_some() {
            return Some(PrefilterFailure { reason: PrefilterReason::TimeWindow });
        }
    }

    None
}

fn check_vehicle_id_mismatch(actor: &Actor, job: &Job) -> Option<String> {
    job.dimens()
        .get_vehicle_id()
        .filter(|job_vehicle_id| actor.vehicle.dimens.get_vehicle_id().as_ref() != Some(job_vehicle_id))
        .map(|job_vehicle_id| {
            let actor_vehicle_id =
                actor.vehicle.dimens.get_vehicle_id().cloned().unwrap_or_else(|| "unknown".to_string());
            format!("job_vehicle_id={job_vehicle_id}, actor_vehicle_id={actor_vehicle_id}")
        })
}

fn check_skill_mismatch(actor: &Actor, job: &Job) -> Option<String> {
    let job_bits = job.dimens().get_job_skills_bitset();
    let vehicle_bits = actor.vehicle.dimens.get_vehicle_skills_bitset();
    if let (Some(job_bits), Some(vehicle_bits)) = (job_bits, vehicle_bits) {
        let ok_all = check_all_of_bits(job_bits, vehicle_bits);
        let ok_one = check_one_of_bits(job_bits, vehicle_bits);
        let ok_none = check_none_of_bits(job_bits, vehicle_bits);

        if ok_all && ok_one && ok_none {
            return None;
        }

        return Some(format!(
            "bitset_ok_all={ok_all}, bitset_ok_one={ok_one}, bitset_ok_none={ok_none}, bits={}",
            vehicle_bits.bits.len()
        ));
    }

    let Some(job_skills) = job.dimens().get_job_skills() else {
        return None;
    };

    let vehicle_skills = actor.vehicle.dimens.get_vehicle_skills();
    let is_ok = check_all_of(&job_skills, &vehicle_skills)
        && check_one_of(&job_skills, &vehicle_skills)
        && check_none_of(&job_skills, &vehicle_skills);

    if is_ok { None } else { Some(format_job_skills_details(&job_skills, vehicle_skills)) }
}

fn check_time_window_mismatch(actor: &Actor, job: &Job, date: Float) -> Option<String> {
    let has_time_intersection = job_has_time_intersection(job, &actor.detail.time, date);
    if has_time_intersection { None } else { Some(format_job_time_window_details(job, &actor.detail.time, date)) }
}

fn job_has_time_intersection(job: &Job, actor_time: &TimeWindow, date: Float) -> bool {
    let check_single = |single: &Arc<Single>| {
        single.places.iter().flat_map(|place| place.times.iter()).any(|time| time.intersects(date, actor_time))
    };

    match job {
        Job::Single(single) => check_single(single),
        Job::Multi(multi) => multi.jobs.iter().all(check_single),
    }
}

fn format_job_time_window_details(job: &Job, actor_time: &TimeWindow, date: Float) -> String {
    let mut min_start = Float::MAX;
    let mut max_end = Float::MIN;
    let mut tw_count = 0_usize;

    let mut collect = |single: &Arc<Single>| {
        for place in single.places.iter() {
            for time in place.times.iter() {
                let tw = time.to_time_window(date);
                min_start = min_start.min(tw.start);
                max_end = max_end.max(tw.end);
                tw_count += 1;
            }
        }
    };

    match job {
        Job::Single(single) => collect(single),
        Job::Multi(multi) => multi.jobs.iter().for_each(|single| collect(single)),
    }

    format!(
        "actor_time=[{:.3},{:.3}], job_tw_count={tw_count}, job_tw_range=[{:.3},{:.3}], date={:.3}",
        actor_time.start, actor_time.end, min_start, max_end, date
    )
}

fn format_job_skills_details(job_skills: &JobSkills, vehicle_skills: Option<&HashSet<String>>) -> String {
    let empty: HashSet<String> = HashSet::new();
    let vehicle_skills = vehicle_skills.unwrap_or(&empty);

    let missing_all = job_skills
        .all_of
        .as_ref()
        .map(|skills| skills.iter().filter(|skill| !vehicle_skills.contains(*skill)).cloned().collect::<Vec<_>>())
        .unwrap_or_default();

    let has_one_of = job_skills
        .one_of
        .as_ref()
        .map_or(true, |skills| skills.is_empty() || skills.iter().any(|skill| vehicle_skills.contains(skill)));

    let conflicting_none = job_skills
        .none_of
        .as_ref()
        .map(|skills| skills.iter().filter(|skill| vehicle_skills.contains(*skill)).cloned().collect::<Vec<_>>())
        .unwrap_or_default();

    let missing_all_sample = format_skill_sample(&missing_all);
    let conflicting_none_sample = format_skill_sample(&conflicting_none);

    format!(
        "all_of_missing={}, one_of_ok={has_one_of}, none_of_conflict={}, vehicle_skills_count={}",
        missing_all_sample,
        conflicting_none_sample,
        vehicle_skills.len()
    )
}

fn format_skill_sample(skills: &[String]) -> String {
    if skills.is_empty() {
        return "0".to_string();
    }

    let sample = skills.iter().take(3).cloned().collect::<Vec<_>>().join(",");
    if skills.len() > 3 { format!("{}(+{})", sample, skills.len() - 3) } else { sample }
}

fn check_all_of(job_skills: &JobSkills, vehicle_skills: &Option<&HashSet<String>>) -> bool {
    match (job_skills.all_of.as_ref(), vehicle_skills) {
        (Some(job_skills), Some(vehicle_skills)) => job_skills.is_subset(vehicle_skills),
        (Some(skills), None) if skills.is_empty() => true,
        (Some(_), None) => false,
        _ => true,
    }
}

fn check_one_of(job_skills: &JobSkills, vehicle_skills: &Option<&HashSet<String>>) -> bool {
    match (job_skills.one_of.as_ref(), vehicle_skills) {
        (Some(job_skills), Some(vehicle_skills)) => job_skills.iter().any(|skill| vehicle_skills.contains(skill)),
        (Some(skills), None) if skills.is_empty() => true,
        (Some(_), None) => false,
        _ => true,
    }
}

fn check_none_of(job_skills: &JobSkills, vehicle_skills: &Option<&HashSet<String>>) -> bool {
    match (job_skills.none_of.as_ref(), vehicle_skills) {
        (Some(job_skills), Some(vehicle_skills)) => job_skills.is_disjoint(vehicle_skills),
        _ => true,
    }
}

fn check_all_of_bits(job_bits: &JobSkillsBitset, vehicle_bits: &VehicleSkillsBitset) -> bool {
    if job_bits.all_of.is_empty() {
        return true;
    }
    if job_bits.all_of.len() != vehicle_bits.bits.len() {
        return false;
    }

    for (job, vehicle) in job_bits.all_of.iter().zip(vehicle_bits.bits.iter()) {
        if job & !vehicle != 0 {
            return false;
        }
    }
    true
}

fn check_one_of_bits(job_bits: &JobSkillsBitset, vehicle_bits: &VehicleSkillsBitset) -> bool {
    if job_bits.one_of.is_empty() {
        return true;
    }
    if job_bits.one_of.len() != vehicle_bits.bits.len() {
        return false;
    }

    let mut has_any = false;
    for (job, vehicle) in job_bits.one_of.iter().zip(vehicle_bits.bits.iter()) {
        if *job != 0 {
            has_any = true;
            if job & vehicle != 0 {
                return true;
            }
        }
    }

    !has_any
}

fn check_none_of_bits(job_bits: &JobSkillsBitset, vehicle_bits: &VehicleSkillsBitset) -> bool {
    if job_bits.none_of.is_empty() {
        return true;
    }
    if job_bits.none_of.len() != vehicle_bits.bits.len() {
        return false;
    }

    for (job, vehicle) in job_bits.none_of.iter().zip(vehicle_bits.bits.iter()) {
        if job & vehicle != 0 {
            return false;
        }
    }
    true
}

fn try_insert_conditional_jobs(insertion_ctx: &mut InsertionContext) {
    repair_conditional_jobs(insertion_ctx);
}

fn get_route_cost(
    route_ctx: &RouteContext,
    activity: &dyn ActivityCost,
    transport: &dyn TransportCost,
) -> Option<Cost> {
    let route = route_ctx.route();
    let mut total = Cost::default();

    if route.tour.has_jobs() {
        total += route.actor.vehicle.costs.fixed + route.actor.driver.costs.fixed;
    }

    let mut iter = route.tour.all_activities();
    let start = iter.next()?;
    let mut prev = start;

    for act in iter {
        let travel_time = TravelTime::Departure(prev.schedule.departure);
        total += transport.cost(route, prev.place.location, act.place.location, travel_time);

        if act.job.is_some() {
            total += activity.cost(route, act, act.schedule.arrival);
        }

        prev = act;
    }

    if let Some(overtime) = route.actor.vehicle.dimens.get_vehicle_overtime() {
        let duration = route_ctx.state().get_total_duration().copied().unwrap_or(0.);
        total += overtime.penalty(duration);
    }

    Some(total)
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

fn has_pending_breaks_for_used_vehicles(insertion_ctx: &InsertionContext) -> bool {
    let used_vehicle_ids = insertion_ctx
        .solution
        .routes
        .iter()
        .filter_map(|route_ctx| route_ctx.route().actor.vehicle.dimens.get_vehicle_id().cloned())
        .collect::<HashSet<_>>();

    let is_pending_used_break = |job: &Job| {
        job.dimens().get_job_id().is_some_and(|job_id| job_id.contains("_break_"))
            && job.dimens().get_vehicle_id().is_some_and(|vehicle_id| used_vehicle_ids.contains(vehicle_id))
    };

    insertion_ctx.solution.required.iter().any(is_pending_used_break)
        || insertion_ctx.solution.unassigned.keys().any(is_pending_used_break)
        || insertion_ctx.solution.ignored.iter().any(is_pending_used_break)
}

fn collect_job_signature(insertion_ctx: &InsertionContext) -> Vec<(String, usize)> {
    let mut counts = insertion_ctx
        .solution
        .routes
        .iter()
        .flat_map(|route_ctx| route_ctx.route().tour.jobs())
        .filter(|job| !is_conditional_job(job))
        .fold(HashMap::<String, usize>::new(), |mut acc, job| {
            *acc.entry(get_job_id(job)).or_insert(0) += 1;
            acc
        })
        .into_iter()
        .collect::<Vec<_>>();

    counts.sort_by(|left, right| left.0.cmp(&right.0));
    counts
}

fn collect_pending_job_signature(insertion_ctx: &InsertionContext) -> Vec<(String, usize)> {
    let mut counts = insertion_ctx
        .solution
        .required
        .iter()
        .chain(insertion_ctx.solution.unassigned.keys())
        .chain(insertion_ctx.solution.ignored.iter())
        .filter(|job| !is_conditional_job(job))
        .fold(HashMap::<String, usize>::new(), |mut acc, job| {
            *acc.entry(get_job_id(job)).or_insert(0) += 1;
            acc
        })
        .into_iter()
        .collect::<Vec<_>>();

    counts.sort_by(|left, right| left.0.cmp(&right.0));
    counts
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

fn get_actor_id(actor: &Actor) -> String {
    actor.vehicle.dimens.get_vehicle_id().cloned().unwrap_or_else(|| "unknown".to_string())
}

fn get_job_id(job: &Job) -> String {
    job.dimens().get_job_id().cloned().unwrap_or_else(|| "unknown".to_string())
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
