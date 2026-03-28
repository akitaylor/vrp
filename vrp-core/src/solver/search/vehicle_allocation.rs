use crate::construction::heuristics::InsertionContext;
use crate::models::GoalContext;
use crate::solver::processing::{VehicleAllocation, VehicleAllocationSettings};
use crate::solver::search::{Recreate, RecreateWithBlinks, RecreateWithCheapest, RecreateWithRegret, WeightedRecreate};
use crate::solver::RefinementContext;
use rosomaxa::{HeuristicContext, prelude::{HeuristicObjective, HeuristicSearchOperator, HeuristicSolution, Random}};
use std::cmp::Ordering;
use std::sync::Arc;
use std::sync::Mutex;

/// Applies vehicle allocation periodically during search.
pub struct VehicleAllocationSearch {
    inner: Arc<dyn HeuristicSearchOperator<Context = RefinementContext, Objective = GoalContext, Solution = InsertionContext>
        + Send
        + Sync>,
    allocation: VehicleAllocation,
    interval: usize,
    log: bool,
    top_k: usize,
    samples: usize,
    state: Option<Mutex<AllocationState>>,
}

impl VehicleAllocationSearch {
    /// Creates a new operator which applies allocation at a given interval.
    pub fn new(
        inner: Arc<
            dyn HeuristicSearchOperator<Context = RefinementContext, Objective = GoalContext, Solution = InsertionContext>
                + Send
                + Sync,
        >,
        settings: Arc<VehicleAllocationSettings>,
    ) -> Self {
        Self {
            inner,
            allocation: VehicleAllocation::new(
                settings.max_iterations,
                settings.allow_unused,
                settings.allow_swaps,
                settings.log,
            ),
            interval: settings.interval.max(1),
            log: settings.log,
            top_k: settings.search_top_k,
            samples: settings.search_samples,
            state: if settings.search_top_k > 0 || settings.search_samples > 0 {
                Some(Mutex::new(AllocationState::default()))
            } else {
                None
            },
        }
    }
}

impl HeuristicSearchOperator for VehicleAllocationSearch {
    type Context = RefinementContext;
    type Objective = GoalContext;
    type Solution = InsertionContext;

    fn search(&self, heuristic_ctx: &Self::Context, solution: &Self::Solution) -> Self::Solution {
        let new_solution = self.inner.search(heuristic_ctx, solution);
        let generation = heuristic_ctx.statistics().generation;

        if generation > 0 && generation % self.interval == 0 {
            let recreate = create_vehicle_allocation_recreate(new_solution.environment.random.clone());

            if let Some(state) = self.state.as_ref() {
                if let Some(best) = self.next_top_k_solution(heuristic_ctx, state, generation, recreate.as_ref()) {
                    return best;
                }
                return new_solution;
            }

            return self.allocation.apply_with_recreate(heuristic_ctx, new_solution, recreate.as_ref());
        }

        new_solution
    }
}

#[derive(Default)]
struct AllocationState {
    generation: usize,
    pending: Vec<InsertionContext>,
}

impl VehicleAllocationSearch {
    fn next_top_k_solution(
        &self,
        heuristic_ctx: &RefinementContext,
        state: &Mutex<AllocationState>,
        generation: usize,
        recreate: &dyn Recreate,
    ) -> Option<InsertionContext> {
        let mut guard = state.lock().expect("vehicle allocation state poisoned");

        if guard.generation != generation {
            guard.generation = generation;
            guard.pending = self.collect_candidate_solutions(heuristic_ctx, recreate);

            if self.log {
                let produced = guard.pending.len();
                (heuristic_ctx.environment.logger)(
                    format!(
                        "vehicle allocation top-k: generation={generation}, produced={produced}, top_k={}, samples={}",
                        self.top_k, self.samples
                    )
                    .as_str(),
                );
            }
        }

        guard.pending.pop()
    }

    fn collect_candidate_solutions(&self, heuristic_ctx: &RefinementContext, recreate: &dyn Recreate) -> Vec<InsertionContext> {
        if self.top_k == 0 && self.samples == 0 {
            return Vec::new();
        }

        let goal = heuristic_ctx.objective();
        let ranked = heuristic_ctx.ranked().collect::<Vec<_>>();
        let mut improved = Vec::new();

        if ranked.is_empty() {
            return improved;
        }

        let max_top = self.top_k.min(ranked.len());
        let mut selected = vec![false; ranked.len()];
        let mut selected_indices = Vec::new();

        for idx in 0..max_top {
            selected[idx] = true;
            selected_indices.push(idx);
        }

        let mut remaining_samples = self.samples.min(ranked.len().saturating_sub(max_top));
        if remaining_samples > 0 {
            let random = heuristic_ctx.environment().random.as_ref();
            let start = if ranked.len() == max_top { 0 } else { max_top };

            while remaining_samples > 0 {
                let idx = if ranked.len() == 1 {
                    0
                } else {
                    random.uniform_int(start as i32, (ranked.len() - 1) as i32) as usize
                };

                if !selected[idx] {
                    selected[idx] = true;
                    selected_indices.push(idx);
                    remaining_samples -= 1;
                }
            }
        }

        for idx in selected_indices {
            let original = ranked[idx].deep_copy();
            let updated = self.allocation.apply_with_recreate(heuristic_ctx, original.deep_copy(), recreate);

            if goal.total_order(&updated, &original) == Ordering::Less {
                improved.push(updated);
            }
        }

        improved
    }
}

fn create_vehicle_allocation_recreate(random: Arc<dyn Random>) -> Arc<dyn Recreate> {
    Arc::new(WeightedRecreate::new(vec![
        (Arc::new(RecreateWithCheapest::new(random.clone())), 4),
        (Arc::new(RecreateWithRegret::new(1, 3, random.clone())), 2),
        (Arc::new(RecreateWithBlinks::new_with_defaults(random)), 1),
    ]))
}
