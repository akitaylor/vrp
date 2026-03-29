use crate::construction::heuristics::InsertionContext;
use crate::models::GoalContext;
use crate::solver::processing::{VehicleAllocation, VehicleAllocationSettings};
use crate::solver::RefinementContext;
use rosomaxa::{
    HeuristicContext,
    population::SelectionPhase,
    prelude::{HeuristicObjective, HeuristicSearchOperator, HeuristicSolution},
};
use std::cmp::Ordering;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

const MAX_IN_SEARCH_TOP_K: usize = 1;
const MAX_IN_SEARCH_SAMPLES: usize = 0;
const IN_SEARCH_MAX_ITERATIONS: usize = 1;

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
}

#[derive(Default)]
struct SharedAllocationState {
    last_run_generation: usize,
    last_report_generation: usize,
    window_elapsed_ms: u64,
    window_runs: usize,
    window_found: usize,
    window_generated: usize,
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
                IN_SEARCH_MAX_ITERATIONS.min(settings.max_iterations),
                settings.allow_unused,
                settings.allow_swaps,
                settings.log,
            ),
            interval: settings.interval.max(1),
            log: settings.log,
            top_k: settings.search_top_k.min(MAX_IN_SEARCH_TOP_K),
            samples: settings.search_samples.min(MAX_IN_SEARCH_SAMPLES),
        }
    }

    fn shared_state() -> &'static Mutex<SharedAllocationState> {
        static SHARED_STATE: OnceLock<Mutex<SharedAllocationState>> = OnceLock::new();
        SHARED_STATE.get_or_init(|| Mutex::new(SharedAllocationState::default()))
    }

    fn try_begin_run(&self, generation: usize) -> bool {
        let mut guard = Self::shared_state().lock().expect("vehicle allocation state poisoned");
        if guard.last_run_generation == generation {
            return false;
        }

        guard.last_run_generation = generation;
        true
    }

    fn record_stats(&self, elapsed_ms: u64, found: usize, generated: usize, runs: usize) {
        let mut guard = Self::shared_state().lock().expect("vehicle allocation state poisoned");
        guard.window_elapsed_ms += elapsed_ms;
        guard.window_found += found;
        guard.window_generated += generated;
        guard.window_runs += runs;
    }

    fn try_log_stats(&self, heuristic_ctx: &RefinementContext, generation: usize) {
        if !self.log || generation == 0 || generation % 100 != 0 {
            return;
        }

        let mut guard = Self::shared_state().lock().expect("vehicle allocation state poisoned");
        if guard.last_report_generation == generation {
            return;
        }

        let elapsed_ms = guard.window_elapsed_ms;
        let runs = guard.window_runs;
        let found = guard.window_found;
        let produced = guard.window_generated;
        let avg_ms = if runs == 0 { 0.0 } else { elapsed_ms as f64 / runs as f64 };

        (heuristic_ctx.environment.logger)(
            format!(
                "vehicle allocation summary: gen={generation}, window=100, search_ms={elapsed_ms}, runs={runs}, avg_ms={avg_ms:.1}, found={found}, candidates={produced}"
            )
            .as_str(),
        );

        guard.window_elapsed_ms = 0;
        guard.window_runs = 0;
        guard.window_found = 0;
        guard.window_generated = 0;
        guard.last_report_generation = generation;
    }

    fn collect_best_candidate_solution(&self, heuristic_ctx: &RefinementContext) -> Option<(InsertionContext, u64, usize, usize)> {
        if self.top_k == 0 && self.samples == 0 {
            return None;
        }

        let goal = heuristic_ctx.objective();
        let ranked = heuristic_ctx.ranked().collect::<Vec<_>>();
        if ranked.is_empty() {
            return None;
        }

        let mut selected_indices = Vec::new();
        if self.top_k > 0 {
            selected_indices.push(0);
        } else if self.samples > 0 {
            let random = heuristic_ctx.environment().random.as_ref();
            let idx = if ranked.len() == 1 { 0 } else { random.uniform_int(0, (ranked.len() - 1) as i32) as usize };
            selected_indices.push(idx);
        }

        let mut best_solution = None;
        let mut elapsed_ms = 0_u64;
        let mut found = 0_usize;

        for idx in selected_indices {
            let original = ranked[idx].deep_copy();
            let started_at = Instant::now();
            let updated = self.allocation.apply(original.deep_copy());
            elapsed_ms += started_at.elapsed().as_millis() as u64;

            if goal.total_order(&updated, &original) == Ordering::Less {
                found += 1;
                match &best_solution {
                    Some(best) if goal.total_order(best, &updated) != Ordering::Greater => {}
                    _ => best_solution = Some(updated),
                }
            }
        }

        best_solution.map(|solution| (solution, elapsed_ms, found, 1))
    }
}

impl HeuristicSearchOperator for VehicleAllocationSearch {
    type Context = RefinementContext;
    type Objective = GoalContext;
    type Solution = InsertionContext;

    fn search(&self, heuristic_ctx: &Self::Context, solution: &Self::Solution) -> Self::Solution {
        let new_solution = self.inner.search(heuristic_ctx, solution);
        let generation = heuristic_ctx.statistics().generation;

        self.try_log_stats(heuristic_ctx, generation);

        if generation == 0
            || generation % self.interval != 0
            || heuristic_ctx.selection_phase() != SelectionPhase::Exploitation
            || !self.try_begin_run(generation)
        {
            return new_solution;
        }

        if let Some((best, elapsed_ms, found, runs)) = self.collect_best_candidate_solution(heuristic_ctx) {
            self.record_stats(elapsed_ms, found, found, runs);
            return best;
        }

        let started_at = Instant::now();
        let updated = self.allocation.apply(new_solution.deep_copy());
        let elapsed = started_at.elapsed();
        let found = heuristic_ctx.objective().total_order(&updated, &new_solution) == Ordering::Less;
        self.record_stats(elapsed.as_millis() as u64, usize::from(found), usize::from(found), 1);

        updated
    }
}
