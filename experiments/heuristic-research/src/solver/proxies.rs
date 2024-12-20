use crate::*;
use rosomaxa::example::VectorSolution;
use rosomaxa::population::{RosomaxaWeighted, Shuffled};
use rosomaxa::prelude::*;
use std::any::TypeId;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::MutexGuard;
use vrp_scientific::core::construction::heuristics::InsertionContext;

/// Keeps track of all experiment data for visualization purposes.
#[derive(Default, Serialize, Deserialize)]
pub struct ExperimentData {
    /// Current generation.
    pub generation: usize,
    /// Called on new individuals addition.
    pub on_add: HashMap<usize, Vec<ObservationData>>,
    /// Called on individual selection.
    pub on_select: HashMap<usize, Vec<ObservationData>>,
    /// Called on generation.
    pub on_generation: HashMap<usize, ((), Vec<ObservationData>)>,
    /// Keeps track of population state at specific generation.
    pub population_state: HashMap<usize, PopulationState>,
    /// Keeps track of heuristic state at specific generation.
    pub heuristic_state: HyperHeuristicState,
}

impl ExperimentData {
    /// Clears all data strored.
    pub fn clear(&mut self) {
        self.generation = 0;
        self.on_add.clear();
        self.on_select.clear();
        self.on_generation.clear();
    }
}

impl<'a> TryFrom<&'a str> for ExperimentData {
    type Error = String;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        serde_json::from_str(value).map_err(|err| format!("cannot deserialize experiment data: {err}"))
    }
}

impl<S> From<&S> for ObservationData
where
    S: HeuristicSolution + RosomaxaWeighted + 'static,
{
    fn from(solution: &S) -> Self {
        if TypeId::of::<S>() == TypeId::of::<VectorSolution>() {
            // SAFETY: type id check above ensures that S-type is the right one
            let solution = unsafe { std::mem::transmute::<&S, &VectorSolution>(solution) };
            let fitness = solution.fitness().next().expect("should have fitness");
            assert_eq!(solution.data.len(), 2);
            return ObservationData::Function(DataPoint3D(solution.data[0], fitness, solution.data[1]));
        }

        if TypeId::of::<S>() == TypeId::of::<InsertionContext>() {
            // SAFETY: type id check above ensures that S-type is the right one
            let insertion_ctx = unsafe { std::mem::transmute::<&S, &InsertionContext>(solution) };

            // NOTE a naive conversion to 3D point
            let fitness = insertion_ctx.fitness().collect::<Vec<_>>();
            let (x, y, z) = match fitness.len() {
                0 => (0., 0., 0.),
                1 => (fitness[0], 0., 0.),
                2 => (fitness[0], fitness[1], 0.),
                len if len >= 3 => (fitness[0], fitness[1], fitness[2]),
                _ => unreachable!(),
            };

            // NOTE temporarily disable graph as it is not really used, but consumes resources
            let graph = Default::default(); // insertion_ctx.into()

            return ObservationData::Vrp { graph, point: DataPoint3D(x, y, z) };
        }

        unreachable!()
    }
}

/// A population type which provides way to intercept some of population data.
pub struct ProxyPopulation<P, O, S>
where
    P: HeuristicPopulation<Objective = O, Individual = S> + 'static,
    O: HeuristicObjective<Solution = S> + Shuffled + 'static,
    S: HeuristicSolution + RosomaxaWeighted + 'static,
{
    generation: usize,
    inner: P,
}

impl<P, O, S> ProxyPopulation<P, O, S>
where
    P: HeuristicPopulation<Objective = O, Individual = S> + 'static,
    O: HeuristicObjective<Solution = S> + Shuffled + 'static,
    S: HeuristicSolution + RosomaxaWeighted + 'static,
{
    /// Creates a new instance of `ProxyPopulation`.
    pub fn new(inner: P) -> Self {
        EXPERIMENT_DATA.lock().unwrap().clear();
        Self { generation: 0, inner }
    }

    fn acquire(&self) -> MutexGuard<ExperimentData> {
        EXPERIMENT_DATA.lock().unwrap()
    }
}

impl<P, O, S> HeuristicPopulation for ProxyPopulation<P, O, S>
where
    P: HeuristicPopulation<Objective = O, Individual = S> + 'static,
    O: HeuristicObjective<Solution = S> + Shuffled,
    S: HeuristicSolution + RosomaxaWeighted,
{
    type Objective = O;
    type Individual = S;

    fn add_all(&mut self, individuals: Vec<Self::Individual>) -> bool {
        self.acquire().on_add.entry(self.generation).or_default().extend(individuals.iter().map(|i| i.into()));

        self.inner.add_all(individuals)
    }

    fn add(&mut self, individual: Self::Individual) -> bool {
        self.acquire().on_add.entry(self.generation).or_default().push((&individual).into());

        self.inner.add(individual)
    }

    fn on_generation(&mut self, statistics: &HeuristicStatistics) {
        self.generation = statistics.generation;
        self.acquire().generation = statistics.generation;

        let individuals = self.inner.all().map(|individual| individual.into()).collect();
        self.acquire().on_generation.insert(self.generation, ((), individuals));

        self.acquire().population_state.insert(self.generation, get_population_state(&self.inner));

        self.inner.on_generation(statistics)
    }

    fn cmp(&self, a: &Self::Individual, b: &Self::Individual) -> Ordering {
        self.inner.cmp(a, b)
    }

    fn select<'a>(&'a self) -> Box<dyn Iterator<Item = &Self::Individual> + 'a> {
        Box::new(self.inner.select().inspect(|&individual| {
            self.acquire().on_select.entry(self.generation).or_default().push(individual.into());
        }))
    }

    fn ranked<'a>(&'a self) -> Box<dyn Iterator<Item = &Self::Individual> + 'a> {
        self.inner.ranked()
    }

    fn all<'a>(&'a self) -> Box<dyn Iterator<Item = &Self::Individual> + 'a> {
        self.inner.all()
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn selection_phase(&self) -> SelectionPhase {
        self.inner.selection_phase()
    }
}

/// Creates info logger proxy to catch dynamic heuristic state.
pub fn create_info_logger_proxy(inner: InfoLogger) -> InfoLogger {
    Arc::new(move |msg| {
        if let Some(state) = HyperHeuristicState::try_parse_all(msg) {
            EXPERIMENT_DATA.lock().unwrap().heuristic_state = state;
        } else {
            (inner)(msg)
        }
    })
}
