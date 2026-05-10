#[cfg(test)]
#[path = "../../../../tests/unit/solver/search/local/cluster_relocate_test.rs"]
mod cluster_relocate_test;

use super::*;
use crate::models::common::{Cost, Timestamp};
use crate::models::problem::Job;
use crate::solver::search::{Recreate, RecreateWithCheapest};
use std::cmp::Ordering;

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
        }
    }
}

/// A local search operator which moves a job into a spatially compatible route and can eject
/// poorly fitting jobs from the target route, repairing them with cheapest insertion.
pub struct ClusterRelocate {
    config: ClusterRelocateConfig,
}

impl ClusterRelocate {
    /// Creates a new instance of `ClusterRelocate`.
    pub fn new(config: ClusterRelocateConfig) -> Self {
        Self { config }
    }
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
        if insertion_ctx.solution.routes.len() < 2 || self.config.job_candidates == 0 {
            return None;
        }

        let route_groups = group_routes_by_proximity(insertion_ctx);
        let original_unassigned = insertion_ctx.solution.unassigned.len();

        let mut best_ctx = None;

        route_groups.iter().enumerate().for_each(|(target_idx, source_indices)| {
            source_indices.iter().take(self.config.route_neighbors).for_each(|&source_idx| {
                if target_idx == source_idx {
                    return;
                }

                collect_inbound_candidates(insertion_ctx, target_idx, source_idx, &self.config).into_iter().for_each(
                    |inbound| {
                        let eviction_candidates =
                            collect_eviction_candidates(insertion_ctx, target_idx, &inbound, &self.config);
                        let max_evictions = self.config.max_evictions.min(eviction_candidates.len());

                        (0..=max_evictions).for_each(|eviction_count| {
                            let evicted = eviction_candidates.iter().take(eviction_count).cloned().collect::<Vec<_>>();
                            let Some(candidate_ctx) = try_relocate(
                                refinement_ctx,
                                insertion_ctx,
                                target_idx,
                                source_idx,
                                &inbound,
                                evicted.as_slice(),
                                &self.config,
                            ) else {
                                return;
                            };

                            if !self.config.allow_unassigned
                                && candidate_ctx.solution.unassigned.len() > original_unassigned
                            {
                                return;
                            }

                            if refinement_ctx.problem.goal.total_order(insertion_ctx, &candidate_ctx)
                                != Ordering::Greater
                            {
                                return;
                            }

                            let is_better = best_ctx.as_ref().is_none_or(|best| {
                                refinement_ctx.problem.goal.total_order(best, &candidate_ctx) == Ordering::Greater
                            });

                            if is_better {
                                best_ctx = Some(candidate_ctx);
                            }
                        });
                    },
                );
            });
        });

        best_ctx
    }
}

fn collect_inbound_candidates(
    insertion_ctx: &InsertionContext,
    target_idx: usize,
    source_idx: usize,
    config: &ClusterRelocateConfig,
) -> Vec<Job> {
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

            route_proximity(insertion_ctx, target_idx, job, None).map(|proximity| (job.clone(), shared, proximity))
        })
        .collect::<Vec<_>>();

    candidates.sort_by(|(_, left_shared, left_proximity), (_, right_shared, right_proximity)| {
        right_shared.cmp(left_shared).then_with(|| left_proximity.total_cmp(right_proximity))
    });

    candidates.into_iter().take(config.job_candidates).map(|(job, ..)| job).collect()
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
            route_proximity(insertion_ctx, target_idx, job, Some(inbound)).map(|proximity| {
                let shared = count_shared_route_neighbors(insertion_ctx, target_idx, job, config.neighbor_radius);
                (job.clone(), shared, proximity)
            })
        })
        .collect::<Vec<_>>();

    candidates.sort_by(|(_, left_shared, left_proximity), (_, right_shared, right_proximity)| {
        left_shared.cmp(right_shared).then_with(|| right_proximity.total_cmp(left_proximity))
    });

    candidates.into_iter().map(|(job, ..)| job).collect()
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
    } else {
        candidate_ctx.restore();
    }

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
