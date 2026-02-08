use super::*;
use crate::construction::enablers::{
    advance_departure_time,
    create_reserved_times_fn,
    optimize_reserved_times_schedule,
    TotalDurationTourState,
};
use crate::construction::features::{
    JobSkills,
    JobSkillsBitset,
    JobSkillsBitsetDimension,
    JobSkillsDimension,
    VehicleOvertimeDimension,
    VehicleSkillsBitset,
    VehicleSkillsBitsetDimension,
    VehicleSkillsDimension,
};
use crate::construction::heuristics::*;
use crate::models::Extras;
use crate::models::common::{Cost, TimeSpan, TimeWindow};
use crate::models::problem::{
    ActivityCost,
    Actor,
    Job,
    JobIdDimension,
    Single,
    TransportCost,
    TravelTime,
    VehicleIdDimension,
};
use crate::models::solution::Activity;
use crate::utils::InfoLogger;
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
const DEBUG_FAILURE_LOG_LIMIT: usize = 5;

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
            (insertion_ctx.environment.logger)(
                "vehicle allocation config: insertion_position=any, cost_mode=transport+activity(per-leg)".as_ref(),
            );
        }

        let mut applied_moves = 0_usize;
        let mut debug_state = DebugLogState::new(DEBUG_FAILURE_LOG_LIMIT);

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
            let mut candidates: Vec<AllocationMove> = Vec::new();
            let mut stats = AllocationStats::new(route_infos.len(), unused_actors.len());
            stats.routes_missing_cost = route_infos.iter().filter(|info| info.cost.is_none()).count();
            let logger = if self.log { Some(&insertion_ctx.environment.logger) } else { None };

            if self.allow_unused && !unused_actors.is_empty() {
                for (route_idx, info) in route_infos.iter().enumerate() {
                    let Some(old_cost) = info.cost else { continue };
                    if info.order.is_empty() {
                        continue;
                    }
                    for actor in unused_actors.iter() {
                        stats.reassign_attempts += 1;
                        stats.prefilter_attempts += 1;
                        if let Some(failure) = prefilter_actor_for_order(actor, info.order.as_slice()) {
                            match failure.reason {
                                PrefilterReason::VehicleId => stats.prefilter_vehicle_id += 1,
                                PrefilterReason::Skills => stats.prefilter_skills += 1,
                                PrefilterReason::TimeWindow => stats.prefilter_time += 1,
                            }
                            debug_state.log(
                                logger,
                                format!(
                                    "vehicle allocation prefilter reject: route_idx={route_idx}, actor={}, job={}, reason={}, details={}",
                                    get_actor_id(actor),
                                    get_job_id(&failure.job),
                                    failure.reason.label(),
                                    failure.details
                                ),
                            );
                            continue;
                        }
                        let Some((route_ctx, cost)) =
                            build_route_for_actor(
                                &insertion_ctx,
                                actor,
                                info.order.as_slice(),
                                &mut stats.build_failures,
                                logger,
                                &mut debug_state,
                                route_idx,
                            )
                        else {
                            continue;
                        };
                        stats.reassign_build_ok += 1;

                        if cost + f64::EPSILON < old_cost {
                            let improvement = old_cost - cost;
                            stats.reassign_improving += 1;
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
                    stats.swap_attempts += 1;

                    stats.prefilter_attempts += 1;
                    if let Some(failure) = prefilter_actor_for_order(&right_info.actor, left_info.order.as_slice()) {
                        match failure.reason {
                            PrefilterReason::VehicleId => stats.prefilter_vehicle_id += 1,
                            PrefilterReason::Skills => stats.prefilter_skills += 1,
                            PrefilterReason::TimeWindow => stats.prefilter_time += 1,
                        }
                        debug_state.log(
                            logger,
                            format!(
                                "vehicle allocation prefilter reject: route_idx={left}, actor={}, job={}, reason={}, details={}",
                                get_actor_id(&right_info.actor),
                                get_job_id(&failure.job),
                                failure.reason.label(),
                                failure.details
                            ),
                        );
                        continue;
                    }

                    stats.prefilter_attempts += 1;
                    if let Some(failure) = prefilter_actor_for_order(&left_info.actor, right_info.order.as_slice()) {
                        match failure.reason {
                            PrefilterReason::VehicleId => stats.prefilter_vehicle_id += 1,
                            PrefilterReason::Skills => stats.prefilter_skills += 1,
                            PrefilterReason::TimeWindow => stats.prefilter_time += 1,
                        }
                        debug_state.log(
                            logger,
                            format!(
                                "vehicle allocation prefilter reject: route_idx={right}, actor={}, job={}, reason={}, details={}",
                                get_actor_id(&left_info.actor),
                                get_job_id(&failure.job),
                                failure.reason.label(),
                                failure.details
                            ),
                        );
                        continue;
                    }

                    let Some((left_route, left_cost)) = build_route_for_actor(
                        &insertion_ctx,
                        &right_info.actor,
                        left_info.order.as_slice(),
                            &mut stats.build_failures,
                            logger,
                            &mut debug_state,
                            left,
                        )
                        else {
                            continue;
                        };
                        let Some((right_route, right_cost)) = build_route_for_actor(
                            &insertion_ctx,
                            &left_info.actor,
                            right_info.order.as_slice(),
                            &mut stats.build_failures,
                            logger,
                            &mut debug_state,
                            right,
                        )
                        else {
                            continue;
                        };
                        stats.swap_build_ok += 1;

                        let (Some(left_old_cost), Some(right_old_cost)) = (left_info.cost, right_info.cost) else {
                            continue;
                        };
                        let old_cost = left_old_cost + right_old_cost;
                        let new_cost = left_cost + right_cost;

                        if new_cost + f64::EPSILON < old_cost {
                            let improvement = old_cost - new_cost;
                            stats.swap_improving += 1;
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

            if self.log {
                (insertion_ctx.environment.logger)(
                    format!(
                        "vehicle allocation stats: routes={}, routes_missing_cost={}, unused_actors={}, reassign_attempts={}, reassign_build_ok={}, reassign_improving={}, swap_attempts={}, swap_build_ok={}, swap_improving={}, prefilter_attempts={}, prefilter_vehicle_id={}, prefilter_skills={}, prefilter_time={}, build_failures={}",
                        stats.routes_total,
                        stats.routes_missing_cost,
                        stats.unused_actors,
                        stats.reassign_attempts,
                        stats.reassign_build_ok,
                        stats.reassign_improving,
                        stats.swap_attempts,
                        stats.swap_build_ok,
                        stats.swap_improving,
                        stats.prefilter_attempts,
                        stats.prefilter_vehicle_id,
                        stats.prefilter_skills,
                        stats.prefilter_time,
                        stats.build_failures.format()
                    )
                    .as_str(),
                );
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

struct DebugLogState {
    remaining: usize,
}

impl DebugLogState {
    fn new(remaining: usize) -> Self {
        Self { remaining }
    }

    fn log(&mut self, logger: Option<&InfoLogger>, message: String) {
        if self.remaining == 0 {
            return;
        }
        if let Some(logger) = logger {
            (logger)(message.as_str());
        }
        self.remaining = self.remaining.saturating_sub(1);
    }
}

#[derive(Copy, Clone, Debug)]
enum PrefilterReason {
    VehicleId,
    Skills,
    TimeWindow,
}

impl PrefilterReason {
    fn label(self) -> &'static str {
        match self {
            PrefilterReason::VehicleId => "vehicle_id",
            PrefilterReason::Skills => "skills",
            PrefilterReason::TimeWindow => "time_window",
        }
    }
}

struct PrefilterFailure {
    reason: PrefilterReason,
    job: Job,
    details: String,
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
    swap_attempts: usize,
    swap_build_ok: usize,
    swap_improving: usize,
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
            swap_attempts: 0,
            swap_build_ok: 0,
            swap_improving: 0,
            prefilter_attempts: 0,
            prefilter_vehicle_id: 0,
            prefilter_skills: 0,
            prefilter_time: 0,
            build_failures: BuildRouteFailureCounters::default(),
        }
    }
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

fn reorder_route_order(order: &[(Job, Arc<Single>)], actor: &Actor) -> Vec<(Job, Arc<Single>)> {
    let date = actor
        .detail
        .start
        .as_ref()
        .map(|start| start.time.to_time_window().start)
        .unwrap_or(actor.detail.time.start);

    let mut reordered = order.to_vec();
    reordered.sort_by(|(_, left), (_, right)| {
        let (left_start, left_end) = get_single_time_bounds(left, date);
        let (right_start, right_end) = get_single_time_bounds(right, date);

        left_start
            .total_cmp(&right_start)
            .then_with(|| left_end.total_cmp(&right_end))
    });

    reordered
}

fn is_conditional_job(job: &Job) -> bool {
    let Some(job_id) = job.dimens().get_job_id() else { return false };
    job.dimens().get_vehicle_id().is_some()
        && (job_id.contains("_break_") || job_id.contains("_reload_") || job_id.contains("_recharge_"))
}

fn prefilter_actor_for_order(actor: &Actor, order: &[(Job, Arc<Single>)]) -> Option<PrefilterFailure> {
    let date = actor
        .detail
        .start
        .as_ref()
        .map(|start| start.time.to_time_window().start)
        .unwrap_or(actor.detail.time.start);

    for (job, _) in order.iter() {
        if let Some(details) = check_vehicle_id_mismatch(actor, job) {
            return Some(PrefilterFailure { reason: PrefilterReason::VehicleId, job: job.clone(), details });
        }
        if let Some(details) = check_skill_mismatch(actor, job) {
            return Some(PrefilterFailure { reason: PrefilterReason::Skills, job: job.clone(), details });
        }
        if let Some(details) = check_time_window_mismatch(actor, job, date) {
            return Some(PrefilterFailure { reason: PrefilterReason::TimeWindow, job: job.clone(), details });
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

    if is_ok {
        None
    } else {
        Some(format_job_skills_details(&job_skills, vehicle_skills))
    }
}

fn check_time_window_mismatch(actor: &Actor, job: &Job, date: Float) -> Option<String> {
    let has_time_intersection = job_has_time_intersection(job, &actor.detail.time, date);
    if has_time_intersection {
        None
    } else {
        Some(format_job_time_window_details(job, &actor.detail.time, date))
    }
}

fn build_job_actor_mismatch_details(actor: &Actor, job: &Job, date: Float) -> String {
    let mut details = Vec::new();

    if let Some(detail) = check_vehicle_id_mismatch(actor, job) {
        details.push(format!("vehicle_id={detail}"));
    }
    if let Some(detail) = check_skill_mismatch(actor, job) {
        details.push(format!("skills={detail}"));
    }
    if let Some(detail) = check_time_window_mismatch(actor, job, date) {
        details.push(format!("time_window={detail}"));
    }

    if details.is_empty() {
        "n/a".to_string()
    } else {
        details.join("; ")
    }
}

fn job_has_time_intersection(job: &Job, actor_time: &TimeWindow, date: Float) -> bool {
    let check_single = |single: &Arc<Single>| {
        single
            .places
            .iter()
            .flat_map(|place| place.times.iter())
            .any(|time| time.intersects(date, actor_time))
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
        actor_time.start,
        actor_time.end,
        min_start,
        max_end,
        date
    )
}

fn format_job_skills_details(job_skills: &JobSkills, vehicle_skills: Option<&HashSet<String>>) -> String {
    let empty: HashSet<String> = HashSet::new();
    let vehicle_skills = vehicle_skills.unwrap_or(&empty);

    let missing_all = job_skills
        .all_of
        .as_ref()
        .map(|skills| {
            skills.iter().filter(|skill| !vehicle_skills.contains(*skill)).cloned().collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let has_one_of = job_skills.one_of.as_ref().map_or(true, |skills| {
        skills.is_empty() || skills.iter().any(|skill| vehicle_skills.contains(skill))
    });

    let conflicting_none = job_skills
        .none_of
        .as_ref()
        .map(|skills| {
            skills.iter().filter(|skill| vehicle_skills.contains(*skill)).cloned().collect::<Vec<_>>()
        })
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
    if skills.len() > 3 {
        format!("{}(+{})", sample, skills.len() - 3)
    } else {
        sample
    }
}

fn get_single_time_bounds(single: &Arc<Single>, date: Float) -> (Float, Float) {
    let mut min_start = Float::MAX;
    let mut min_end = Float::MAX;

    for place in single.places.iter() {
        for time in place.times.iter() {
            let tw = time.to_time_window(date);
            min_start = min_start.min(tw.start);
            min_end = min_end.min(tw.end);
        }
    }

    (min_start, min_end)
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

fn build_route_for_actor(
    insertion_ctx: &InsertionContext,
    actor: &Arc<Actor>,
    order: &[(Job, Arc<Single>)],
    failures: &mut BuildRouteFailureCounters,
    logger: Option<&InfoLogger>,
    debug_state: &mut DebugLogState,
    source_route_idx: usize,
) -> Option<(RouteContext, Cost)> {
    if order.is_empty() {
        return None;
    }

    let date = actor
        .detail
        .start
        .as_ref()
        .map(|start| start.time.to_time_window().start)
        .unwrap_or(actor.detail.time.start);

    let mut new_ctx = InsertionContext::new_empty(insertion_ctx.problem.clone(), insertion_ctx.environment.clone());
    let route_ctx = match new_ctx.solution.registry.get_route(actor) {
        Some(route_ctx) => route_ctx,
        None => {
            failures.registry_missing += 1;
            debug_state.log(
                logger,
                format!(
                    "vehicle allocation build failed: route_idx={source_route_idx}, actor={}, reason=registry_missing",
                    get_actor_id(actor)
                ),
            );
            return None;
        }
    };
    new_ctx.solution.routes.push(route_ctx);
    let route_idx = new_ctx.solution.routes.len() - 1;

    let goal = new_ctx.problem.goal.clone();
    let position = InsertionPosition::Any;
    let leg_selection = LegSelection::Exhaustive;
    let result_selector = BestResultSelector::default();

    let order = reorder_route_order(order, actor);
    let mut synchronized_jobs: HashMap<Job, Vec<Arc<Single>>> = HashMap::default();
    let mut invalid_multi_job_ids: HashSet<Job> = HashSet::default();

    let mut had_invalid_multi = false;
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
            InsertionResult::Failure(failure) if job.as_multi().is_some() => {
                invalid_multi_job_ids.insert(job.clone());
                had_invalid_multi = true;
                debug_state.log(
                    logger,
                    format!(
                        "vehicle allocation build failed: route_idx={source_route_idx}, actor={}, job={}, violation={}, stopped={}, reason=invalid_multi_job",
                        get_actor_id(actor),
                        get_job_id(job),
                        failure.constraint,
                        failure.stopped
                    ),
                );
            }
            InsertionResult::Failure(failure) => {
                failures.insertion_failure += 1;
                let details = build_job_actor_mismatch_details(actor, job, date);
                debug_state.log(
                    logger,
                    format!(
                        "vehicle allocation build failed: route_idx={source_route_idx}, actor={}, job={}, violation={}, stopped={}, is_break={}, details={}",
                        get_actor_id(actor),
                        get_job_id(job),
                        failure.constraint,
                        failure.stopped,
                        is_conditional_job(job),
                        details
                    ),
                );
                return None;
            }
        }
    }

    if !invalid_multi_job_ids.is_empty() {
        if had_invalid_multi {
            failures.invalid_multi_job += 1;
        }
        return None;
    }

    if !validate_multi_jobs(&synchronized_jobs) {
        failures.multi_job_mismatch += 1;
        debug_state.log(
            logger,
            format!(
                "vehicle allocation build failed: route_idx={source_route_idx}, actor={}, reason=multi_job_mismatch",
                get_actor_id(actor)
            ),
        );
        return None;
    }

    new_ctx.problem.goal.accept_solution_state(&mut new_ctx.solution);

    if let Some(route_ctx) = new_ctx.solution.routes.get_mut(route_idx) {
        let activity = new_ctx.problem.activity.as_ref();
        let transport = new_ctx.problem.transport.as_ref();
        let consider_whole_tour = true;
        advance_departure_time(route_ctx, activity, transport, consider_whole_tour);
        new_ctx.problem.goal.accept_route_state(route_ctx);

        if let Some(reserved_times) = new_ctx.problem.extras.get_reserved_times().map(|times| times.as_ref().clone()) {
            if let Ok(reserved_times_fn) = create_reserved_times_fn(reserved_times) {
                optimize_reserved_times_schedule(route_ctx.route_mut(), &reserved_times_fn);
                route_ctx.mark_stale(false);
            }
        }
    }

    let route_ctx = new_ctx.solution.routes.into_iter().find(|route_ctx| route_ctx.route().actor == *actor)?;
    let cost = match get_route_cost(&route_ctx, new_ctx.problem.activity.as_ref(), new_ctx.problem.transport.as_ref()) {
        Some(cost) => cost,
        None => {
            failures.cost_missing += 1;
            debug_state.log(
                logger,
                format!(
                    "vehicle allocation build failed: route_idx={source_route_idx}, actor={}, reason=cost_missing",
                    get_actor_id(actor)
                ),
            );
            return None;
        }
    };

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

fn get_route_cost(route_ctx: &RouteContext, activity: &dyn ActivityCost, transport: &dyn TransportCost) -> Option<Cost> {
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
