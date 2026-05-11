#[cfg(test)]
#[path = "../../../../tests/unit/solver/search/local/cluster_relocate_test.rs"]
mod cluster_relocate_test;

use super::*;
use crate::models::common::{Cost, Location, Timestamp};
use crate::models::problem::{Job, TransportCost, TravelTime};
use crate::models::solution::{Activity, Route};
use crate::solver::search::{Recreate, RecreateWithCheapest};
use std::{
    cmp::Ordering,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering},
    },
    time::Instant,
};

/// Settings for cluster-aware relocate with bounded ejection.
#[derive(Clone)]
pub struct ClusterRelocateConfig {
    /// How many nearby source routes to inspect for each target route.
    pub route_neighbors: usize,
    /// How many inbound jobs to inspect for each route pair.
    pub job_candidates: usize,
    /// How many jobs can be ejected from the target route to make room.
    pub max_evictions: usize,
    /// How many nearest job neighbors are used to estimate cluster affinity.
    pub neighbor_radius: usize,
    /// Minimum amount of shared target-route neighbors required for an inbound job.
    pub min_shared_neighbors: usize,
    /// Whether a move can leave extra unassigned jobs.
    pub allow_unassigned: bool,
    /// Maximum amount of globally preselected inbound candidates. Zero means inspect all candidates.
    pub candidate_pool_size: usize,
    /// Whether the operator should become rarer as the amount of unassigned jobs decreases.
    pub phase_aware: bool,
    /// Whether operator activity statistics should be logged.
    pub log: bool,
    /// How often activity statistics should be logged.
    pub log_interval: usize,
}

impl Default for ClusterRelocateConfig {
    fn default() -> Self {
        Self {
            route_neighbors: 4,
            job_candidates: 16,
            max_evictions: 2,
            neighbor_radius: 8,
            min_shared_neighbors: 1,
            allow_unassigned: false,
            candidate_pool_size: 32,
            phase_aware: true,
            log: false,
            log_interval: 100,
        }
    }
}

/// A local search operator which moves a job into a spatially compatible route and can eject
/// poorly fitting jobs from the target route, repairing them with cheapest insertion.
pub struct ClusterRelocate {
    config: ClusterRelocateConfig,
    stats: Arc<ClusterRelocateStats>,
}

impl ClusterRelocate {
    /// Creates a new instance of `ClusterRelocate`.
    pub fn new(config: ClusterRelocateConfig) -> Self {
        Self { config, stats: Arc::new(ClusterRelocateStats::default()) }
    }

    fn record(&self, logger: &InfoLogger, call: usize, delta: ClusterRelocateStatsDelta, accepted: bool) {
        self.stats.inbound_candidates.fetch_add(delta.inbound_candidates, AtomicOrdering::Relaxed);
        self.stats.tested_moves.fetch_add(delta.tested_moves, AtomicOrdering::Relaxed);
        self.stats.feasible_moves.fetch_add(delta.feasible_moves, AtomicOrdering::Relaxed);
        self.stats.improving_moves.fetch_add(delta.improving_moves, AtomicOrdering::Relaxed);
        self.stats.skipped_calls.fetch_add(delta.skipped_calls, AtomicOrdering::Relaxed);
        self.stats.rejected_unassigned.fetch_add(delta.rejected_unassigned, AtomicOrdering::Relaxed);
        self.stats.rejected_non_improving.fetch_add(delta.rejected_non_improving, AtomicOrdering::Relaxed);
        self.stats.collect_nanos.fetch_add(delta.collect_nanos, AtomicOrdering::Relaxed);
        self.stats.eviction_nanos.fetch_add(delta.eviction_nanos, AtomicOrdering::Relaxed);
        self.stats.relocate_nanos.fetch_add(delta.relocate_nanos, AtomicOrdering::Relaxed);

        if accepted {
            self.stats.accepted_moves.fetch_add(1, AtomicOrdering::Relaxed);
        }

        let should_log =
            self.config.log && (accepted || (self.config.log_interval > 0 && call % self.config.log_interval == 0));
        if should_log {
            let calls = self.stats.calls.load(AtomicOrdering::Relaxed);
            let inbound = self.stats.inbound_candidates.load(AtomicOrdering::Relaxed);
            let tested = self.stats.tested_moves.load(AtomicOrdering::Relaxed);
            let feasible = self.stats.feasible_moves.load(AtomicOrdering::Relaxed);
            let improving = self.stats.improving_moves.load(AtomicOrdering::Relaxed);
            let skipped = self.stats.skipped_calls.load(AtomicOrdering::Relaxed);
            let accepted = self.stats.accepted_moves.load(AtomicOrdering::Relaxed);
            let rejected_unassigned = self.stats.rejected_unassigned.load(AtomicOrdering::Relaxed);
            let rejected_non_improving = self.stats.rejected_non_improving.load(AtomicOrdering::Relaxed);
            let collect_ms = nanos_to_millis(self.stats.collect_nanos.load(AtomicOrdering::Relaxed));
            let eviction_ms = nanos_to_millis(self.stats.eviction_nanos.load(AtomicOrdering::Relaxed));
            let relocate_ms = nanos_to_millis(self.stats.relocate_nanos.load(AtomicOrdering::Relaxed));

            (logger)(
                format!(
                    "cluster-relocate: calls={calls}, skipped={skipped}, inbound={inbound}, tested={tested}, feasible={feasible}, improving={improving}, accepted={accepted}, rejected_unassigned={rejected_unassigned}, rejected_non_improving={rejected_non_improving}, collect_ms={collect_ms:.1}, eviction_ms={eviction_ms:.1}, relocate_ms={relocate_ms:.1}"
                )
                .as_str(),
            );
        }
    }
}

#[derive(Default)]
struct ClusterRelocateStats {
    calls: AtomicUsize,
    inbound_candidates: AtomicUsize,
    tested_moves: AtomicUsize,
    feasible_moves: AtomicUsize,
    improving_moves: AtomicUsize,
    skipped_calls: AtomicUsize,
    accepted_moves: AtomicUsize,
    rejected_unassigned: AtomicUsize,
    rejected_non_improving: AtomicUsize,
    collect_nanos: AtomicU64,
    eviction_nanos: AtomicU64,
    relocate_nanos: AtomicU64,
}

#[derive(Default)]
struct ClusterRelocateStatsDelta {
    inbound_candidates: usize,
    tested_moves: usize,
    feasible_moves: usize,
    improving_moves: usize,
    skipped_calls: usize,
    rejected_unassigned: usize,
    rejected_non_improving: usize,
    collect_nanos: u64,
    eviction_nanos: u64,
    relocate_nanos: u64,
}

#[derive(Clone)]
struct MoveCandidate {
    target_idx: usize,
    source_idx: usize,
    inbound: Job,
    net_delta: Cost,
    target_delta: Cost,
    shared: usize,
    proximity: Cost,
}

impl Default for ClusterRelocate {
    fn default() -> Self {
        Self::new(ClusterRelocateConfig::default())
    }
}

impl LocalOperator for ClusterRelocate {
    fn explore(
        &self,
        refinement_ctx: &RefinementContext,
        insertion_ctx: &InsertionContext,
    ) -> Option<InsertionContext> {
        let call = self.stats.calls.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        if insertion_ctx.solution.routes.len() < 2 || self.config.job_candidates == 0 {
            self.record(&insertion_ctx.environment.logger, call, ClusterRelocateStatsDelta::default(), false);
            return None;
        }

        if !should_run_in_phase(insertion_ctx, &self.config) {
            self.record(
                &insertion_ctx.environment.logger,
                call,
                ClusterRelocateStatsDelta { skipped_calls: 1, ..ClusterRelocateStatsDelta::default() },
                false,
            );
            return None;
        }

        let route_groups = group_routes_by_proximity(insertion_ctx);
        let original_unassigned = insertion_ctx.solution.unassigned.len();

        let mut stats_delta = ClusterRelocateStatsDelta::default();

        let collect_start = self.config.log.then(Instant::now);
        let mut candidates = collect_global_candidates(insertion_ctx, &route_groups, &self.config);
        if let Some(start) = collect_start {
            stats_delta.collect_nanos += elapsed_nanos(start);
        }

        stats_delta.inbound_candidates += candidates.len();
        if self.config.candidate_pool_size > 0 {
            candidates.truncate(self.config.candidate_pool_size);
        }

        for candidate in candidates {
            let eviction_start = self.config.log.then(Instant::now);
            let eviction_candidates =
                collect_eviction_candidates(insertion_ctx, candidate.target_idx, &candidate.inbound, &self.config);
            if let Some(start) = eviction_start {
                stats_delta.eviction_nanos += elapsed_nanos(start);
            }
            let max_evictions = self.config.max_evictions.min(eviction_candidates.len());

            for eviction_count in 0..=max_evictions {
                let evicted = eviction_candidates.iter().take(eviction_count).cloned().collect::<Vec<_>>();
                stats_delta.tested_moves += 1;
                let relocate_start = self.config.log.then(Instant::now);
                let candidate_ctx = try_relocate(
                    refinement_ctx,
                    insertion_ctx,
                    candidate.target_idx,
                    candidate.source_idx,
                    &candidate.inbound,
                    evicted.as_slice(),
                    &self.config,
                );
                if let Some(start) = relocate_start {
                    stats_delta.relocate_nanos += elapsed_nanos(start);
                }

                let Some(candidate_ctx) = candidate_ctx else {
                    continue;
                };
                stats_delta.feasible_moves += 1;

                if !self.config.allow_unassigned && candidate_ctx.solution.unassigned.len() > original_unassigned {
                    stats_delta.rejected_unassigned += 1;
                    continue;
                }

                if refinement_ctx.problem.goal.total_order(insertion_ctx, &candidate_ctx) != Ordering::Greater {
                    stats_delta.rejected_non_improving += 1;
                    continue;
                }
                stats_delta.improving_moves += 1;

                self.record(&insertion_ctx.environment.logger, call, stats_delta, true);
                return Some(candidate_ctx);
            }
        }

        self.record(&insertion_ctx.environment.logger, call, stats_delta, false);

        None
    }
}

fn elapsed_nanos(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn nanos_to_millis(nanos: u64) -> Cost {
    nanos as Cost / 1_000_000.
}

fn should_run_in_phase(insertion_ctx: &InsertionContext, config: &ClusterRelocateConfig) -> bool {
    if !config.phase_aware {
        return true;
    }

    let probability =
        get_phase_activation_probability(insertion_ctx.solution.unassigned.len(), insertion_ctx.problem.jobs.size());
    insertion_ctx.environment.random.is_hit(probability)
}

fn get_phase_activation_probability(unassigned: usize, jobs: usize) -> Cost {
    if jobs == 0 {
        return 0.;
    }

    let ratio = unassigned as Cost / jobs as Cost;

    if ratio >= 0.20 {
        1.
    } else if ratio >= 0.08 {
        0.5
    } else if unassigned > 0 {
        0.2
    } else {
        0.05
    }
}

fn collect_global_candidates(
    insertion_ctx: &InsertionContext,
    route_groups: &[Vec<usize>],
    config: &ClusterRelocateConfig,
) -> Vec<MoveCandidate> {
    let mut candidates = route_groups
        .iter()
        .enumerate()
        .flat_map(|(target_idx, source_indices)| {
            source_indices.iter().take(config.route_neighbors).filter_map(move |&source_idx| {
                if target_idx == source_idx {
                    None
                } else {
                    Some(collect_inbound_candidates(insertion_ctx, target_idx, source_idx, config))
                }
            })
        })
        .flatten()
        .collect::<Vec<_>>();

    sort_by_cluster_affinity(&mut candidates);

    if config.candidate_pool_size > 0 {
        candidates.truncate((config.candidate_pool_size * 4).max(config.candidate_pool_size));
    }

    let mut candidates = candidates
        .into_iter()
        .filter_map(|candidate| score_move_candidate(insertion_ctx, candidate))
        .collect::<Vec<_>>();

    sort_move_candidates(&mut candidates);
    candidates
}

fn collect_inbound_candidates(
    insertion_ctx: &InsertionContext,
    target_idx: usize,
    source_idx: usize,
    config: &ClusterRelocateConfig,
) -> Vec<MoveCandidate> {
    let locked = &insertion_ctx.solution.locked;
    let target_route = &insertion_ctx.solution.routes[target_idx];
    let source_route = &insertion_ctx.solution.routes[source_idx];

    let mut candidates = source_route
        .route()
        .tour
        .jobs()
        .filter(|job| !locked.contains(*job) && !target_route.route().tour.has_job(job))
        .filter_map(|job| {
            let shared = count_shared_route_neighbors(insertion_ctx, target_idx, job, config.neighbor_radius);
            if shared < config.min_shared_neighbors {
                return None;
            }

            let proximity = route_proximity(insertion_ctx, target_idx, job, None)?;

            Some(MoveCandidate {
                target_idx,
                source_idx,
                inbound: job.clone(),
                net_delta: proximity,
                target_delta: proximity,
                shared,
                proximity,
            })
        })
        .collect::<Vec<_>>();

    sort_by_cluster_affinity(&mut candidates);

    candidates.into_iter().take(config.job_candidates).collect()
}

fn score_move_candidate(insertion_ctx: &InsertionContext, candidate: MoveCandidate) -> Option<MoveCandidate> {
    let target_delta = estimate_best_insertion_delta(insertion_ctx, candidate.target_idx, &candidate.inbound)?;
    let source_saving = estimate_job_removal_saving(insertion_ctx, candidate.source_idx, &candidate.inbound);

    Some(MoveCandidate { target_delta, net_delta: target_delta - source_saving, ..candidate })
}

fn sort_by_cluster_affinity(candidates: &mut [MoveCandidate]) {
    candidates
        .sort_by(|left, right| right.shared.cmp(&left.shared).then_with(|| left.proximity.total_cmp(&right.proximity)));
}

fn sort_move_candidates(candidates: &mut [MoveCandidate]) {
    candidates.sort_by(|left, right| {
        left.net_delta
            .total_cmp(&right.net_delta)
            .then_with(|| left.target_delta.total_cmp(&right.target_delta))
            .then_with(|| right.shared.cmp(&left.shared))
            .then_with(|| left.proximity.total_cmp(&right.proximity))
    });
}

fn collect_eviction_candidates(
    insertion_ctx: &InsertionContext,
    target_idx: usize,
    inbound: &Job,
    config: &ClusterRelocateConfig,
) -> Vec<Job> {
    let locked = &insertion_ctx.solution.locked;

    let mut candidates = insertion_ctx.solution.routes[target_idx]
        .route()
        .tour
        .jobs()
        .filter(|job| !locked.contains(*job))
        .filter_map(|job| {
            let shared = count_shared_route_neighbors(insertion_ctx, target_idx, job, config.neighbor_radius);
            let proximity = route_proximity(insertion_ctx, target_idx, job, Some(inbound))?;

            Some((job.clone(), shared, proximity, 0.))
        })
        .collect::<Vec<_>>();

    candidates.sort_by(|(_, left_shared, left_proximity, _), (_, right_shared, right_proximity, _)| {
        left_shared.cmp(right_shared).then_with(|| right_proximity.total_cmp(left_proximity))
    });
    candidates.truncate((config.max_evictions + 1) * 4);

    candidates.iter_mut().for_each(|(job, _, _, removal_saving)| {
        *removal_saving = estimate_job_removal_saving(insertion_ctx, target_idx, job);
    });

    candidates.sort_by(
        |(_, left_shared, left_proximity, left_saving), (_, right_shared, right_proximity, right_saving)| {
            right_saving
                .total_cmp(left_saving)
                .then_with(|| left_shared.cmp(right_shared))
                .then_with(|| right_proximity.total_cmp(left_proximity))
        },
    );

    candidates.into_iter().map(|(job, ..)| job).collect()
}

fn estimate_best_insertion_delta(insertion_ctx: &InsertionContext, route_idx: usize, job: &Job) -> Option<Cost> {
    let route = insertion_ctx.solution.routes[route_idx].route();
    let transport = insertion_ctx.problem.transport.as_ref();
    let locations = job.places().filter_map(|place| place.location).collect::<Vec<_>>();

    if locations.is_empty() {
        return None;
    }

    route
        .tour
        .all_activities()
        .as_slice()
        .windows(2)
        .flat_map(|activities| match activities {
            [start, end] => locations
                .iter()
                .map(move |&location| {
                    get_cost_from_activity(route, start, location, transport)
                        + get_cost_to_activity(route, location, end, transport)
                        - get_cost_between_activities(route, start, end, transport)
                })
                .collect::<Vec<_>>(),
            _ => unreachable!("Unexpected activity window"),
        })
        .filter(|cost| *cost >= 0.)
        .min_by(|left, right| left.total_cmp(right))
}

fn estimate_job_removal_saving(insertion_ctx: &InsertionContext, route_idx: usize, job: &Job) -> Cost {
    let route = insertion_ctx.solution.routes[route_idx].route();
    let transport = &insertion_ctx.problem.transport;

    route
        .tour
        .all_activities()
        .as_slice()
        .windows(3)
        .filter_map(|activities| match activities {
            [start, middle, end] if middle.retrieve_job().as_ref() == Some(job) => {
                Some(get_removal_saving(route, start, middle, end, transport.as_ref()))
            }
            [_, _, _] => None,
            _ => unreachable!("Unexpected activity window"),
        })
        .sum()
}

fn get_removal_saving(
    route: &Route,
    start: &Activity,
    middle: &Activity,
    end: &Activity,
    transport: &dyn TransportCost,
) -> Cost {
    let actor = route.actor.as_ref();
    let waiting_costs = (middle.place.time.start - middle.schedule.arrival).max(0.)
        * (actor.driver.costs.per_waiting_time + actor.vehicle.costs.per_waiting_time);

    waiting_costs
        + get_cost_between_activities(route, start, middle, transport)
        + get_cost_between_activities(route, middle, end, transport)
        - get_cost_between_activities(route, start, end, transport)
}

fn get_cost_between_activities(route: &Route, from: &Activity, to: &Activity, transport: &dyn TransportCost) -> Cost {
    transport.cost(route, from.place.location, to.place.location, TravelTime::Departure(from.schedule.departure))
}

fn get_cost_from_activity(
    route: &Route,
    from: &Activity,
    to_location: Location,
    transport: &dyn TransportCost,
) -> Cost {
    transport.cost(route, from.place.location, to_location, TravelTime::Departure(from.schedule.departure))
}

fn get_cost_to_activity(route: &Route, from_location: Location, to: &Activity, transport: &dyn TransportCost) -> Cost {
    transport.cost(route, from_location, to.place.location, TravelTime::Departure(to.schedule.arrival))
}

fn try_relocate(
    refinement_ctx: &RefinementContext,
    insertion_ctx: &InsertionContext,
    target_idx: usize,
    source_idx: usize,
    inbound: &Job,
    evicted: &[Job],
    config: &ClusterRelocateConfig,
) -> Option<InsertionContext> {
    let mut candidate_ctx = insertion_ctx.deep_copy();

    if !candidate_ctx.solution.routes.get_mut(source_idx)?.route_mut().tour.remove(inbound) {
        return None;
    }
    candidate_ctx.problem.goal.accept_route_state(candidate_ctx.solution.routes.get_mut(source_idx)?);

    for job in evicted {
        if !candidate_ctx.solution.routes.get_mut(target_idx)?.route_mut().tour.remove(job) {
            return None;
        }
        candidate_ctx.solution.required.push(job.clone());
    }
    candidate_ctx.problem.goal.accept_route_state(candidate_ctx.solution.routes.get_mut(target_idx)?);

    let leg_selection = LegSelection::Stochastic(candidate_ctx.environment.random.clone());
    let result_selector = BestResultSelector::default();
    let success = {
        let target_route = candidate_ctx.solution.routes.get(target_idx)?;
        test_job_insertion(&candidate_ctx, target_route, inbound, &leg_selection, &result_selector)?
    };

    apply_insertion_success(&mut candidate_ctx, success);

    if !candidate_ctx.solution.required.is_empty() {
        candidate_ctx =
            RecreateWithCheapest::new(candidate_ctx.environment.random.clone()).run(refinement_ctx, candidate_ctx);
    }
    candidate_ctx.restore();

    if !config.allow_unassigned && candidate_ctx.solution.unassigned.keys().any(|job| evicted.contains(job)) {
        return None;
    }

    Some(candidate_ctx)
}

fn test_job_insertion(
    insertion_ctx: &InsertionContext,
    route_ctx: &RouteContext,
    job: &Job,
    leg_selection: &LegSelection,
    result_selector: &dyn ResultSelector,
) -> Option<InsertionSuccess> {
    let eval_ctx = EvaluationContext { goal: &insertion_ctx.problem.goal, job, leg_selection, result_selector };
    match eval_job_insertion_in_route(
        insertion_ctx,
        &eval_ctx,
        route_ctx,
        InsertionPosition::Any,
        InsertionResult::make_failure(),
    ) {
        InsertionResult::Success(success) => Some(success),
        InsertionResult::Failure(_) => None,
    }
}

fn count_shared_route_neighbors(
    insertion_ctx: &InsertionContext,
    route_idx: usize,
    job: &Job,
    neighbor_radius: usize,
) -> usize {
    let route_ctx = &insertion_ctx.solution.routes[route_idx];
    let route = route_ctx.route();
    let departure = route.tour.start().map_or(Timestamp::default(), |start| start.schedule.departure);

    insertion_ctx
        .problem
        .jobs
        .neighbors(&route.actor.vehicle.profile, job, departure)
        .take(neighbor_radius)
        .filter(|(neighbor, _)| route.tour.has_job(neighbor))
        .count()
}

fn route_proximity(
    insertion_ctx: &InsertionContext,
    route_idx: usize,
    job: &Job,
    extra_job: Option<&Job>,
) -> Option<Cost> {
    let route = insertion_ctx.solution.routes[route_idx].route();
    let profile = &route.actor.vehicle.profile;
    let transport = insertion_ctx.problem.transport.as_ref();
    let job_locations = job.places().filter_map(|place| place.location).collect::<Vec<_>>();

    if job_locations.is_empty() {
        return None;
    }

    route
        .tour
        .jobs()
        .filter(|route_job| *route_job != job)
        .chain(extra_job)
        .flat_map(|route_job| {
            route_job.places().filter_map(|place| place.location).flat_map(|route_location| {
                job_locations.iter().map(move |&job_location| {
                    let forward = transport.distance_approx(profile, job_location, route_location);
                    let backward = transport.distance_approx(profile, route_location, job_location);

                    forward.min(backward)
                })
            })
        })
        .filter(|cost| *cost >= 0.)
        .min_by(|left, right| left.total_cmp(right))
}
