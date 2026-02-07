#[cfg(test)]
#[path = "../../../tests/unit/construction/enablers/reserved_time_test.rs"]
mod reserved_time_test;

use crate::models::common::*;
use crate::models::problem::{ActivityCost, Actor, TransportCost, TravelTime};
use crate::models::solution::{Activity, Route};
use rosomaxa::prelude::GenericError;
use std::collections::HashMap;
use std::mem;
use std::ops::ControlFlow;
use std::sync::Arc;

/// Represent a reserved time span entity.
#[derive(Clone, Debug)]
pub struct ReservedTimeSpan {
    /// A specific time span when an extra reserved duration should be applied.
    pub time: TimeSpan,
    /// An extra duration to be applied at given time.
    pub duration: Duration,
}

impl ReservedTimeSpan {
    /// Converts `ReservedTimeSpan` to `ReservedTimeWindow`.
    pub fn to_reserved_time_window(&self, offset: Timestamp) -> ReservedTimeWindow {
        ReservedTimeWindow { time: self.time.to_time_window(offset), duration: self.duration }
    }
}

/// Represent a reserved time window entity.
#[derive(Clone, Debug)]
pub struct ReservedTimeWindow {
    /// A specific time window when an extra reserved duration should be applied.
    pub time: TimeWindow,
    /// An extra duration to be applied at given time.
    pub duration: Duration,
}

/// Specifies reserved time index type.
pub type ReservedTimesIndex = HashMap<Arc<Actor>, Vec<ReservedTimeSpan>>;

/// Specifies a function which returns an extra reserved time window for given actor. This reserved
/// time should be considered for planning.
pub(crate) type ReservedTimesFn = Arc<dyn Fn(&Route, &TimeWindow) -> Option<ReservedTimeWindow> + Send + Sync>;

struct ReservedTimesEntry {
    indices: Vec<u64>,
    intervals: Vec<ReservedTimeSpan>,
    min_start: Timestamp,
    max_end: Timestamp,
}

/// Provides way to calculate activity costs which might contain reserved time.
pub struct DynamicActivityCost {
    reserved_times_fn: ReservedTimesFn,
}

impl DynamicActivityCost {
    /// Creates a new instance of `DynamicActivityCost` with given reserved time function.
    pub fn new(reserved_times_index: ReservedTimesIndex) -> Result<Self, GenericError> {
        Ok(Self { reserved_times_fn: create_reserved_times_fn(reserved_times_index)? })
    }
}

impl ActivityCost for DynamicActivityCost {
    fn estimate_departure(
        &self,
        route: &Route,
        activity: &Activity,
        arrival: Timestamp,
    ) -> ControlFlow<Timestamp, Timestamp> {
        let activity_start = arrival.max(activity.place.time.start);
        let departure = activity_start + activity.place.duration;
        let schedule = TimeWindow::new(arrival, departure);

        (self.reserved_times_fn)(route, &schedule).map_or(ControlFlow::Continue(departure), |reserved_time| {
            // NOTE we ignore reserved_time.time.start and consider the latest possible time only
            let reserved_tw = &reserved_time.time;
            let reserved_tw = TimeWindow::new(reserved_tw.end, reserved_tw.end + reserved_time.duration);

            assert!(reserved_tw.intersects(&schedule));

            let activity_tw = &activity.place.time;

            let extra_duration = if reserved_tw.start < activity_tw.start {
                let waiting_time = TimeWindow::new(arrival, activity_tw.start);
                let overlapping = waiting_time.overlapping(&reserved_tw).map(|tw| tw.duration()).unwrap_or(0.);

                reserved_time.duration - overlapping
            } else {
                reserved_time.duration
            };

            // NOTE: do not allow to start or restart work after break finished
            if activity_start + extra_duration > activity.place.time.end {
                // TODO this branch is the reason why departure rescheduling is disabled.
                //      theoretically, rescheduling should be aware somehow about dynamic costs
                ControlFlow::Break(departure + extra_duration)
            } else {
                ControlFlow::Continue(departure + extra_duration)
            }
        })
    }

    fn estimate_arrival(
        &self,
        route: &Route,
        activity: &Activity,
        departure: Timestamp,
    ) -> ControlFlow<Timestamp, Timestamp> {
        let arrival = activity.place.time.end.min(departure - activity.place.duration);
        let schedule = TimeWindow::new(arrival, departure);

        let value = (self.reserved_times_fn)(route, &schedule)
            .map_or(arrival, |reserved_time| (arrival - reserved_time.duration).max(activity.place.time.start));

        ControlFlow::Continue(value)
    }
}

/// Provides way to calculate transport costs which might contain reserved time.
pub struct DynamicTransportCost {
    reserved_times_fn: ReservedTimesFn,
    inner: Arc<dyn TransportCost>,
}

impl DynamicTransportCost {
    /// Creates a new instance of `DynamicTransportCost`.
    pub fn new(reserved_times_index: ReservedTimesIndex, inner: Arc<dyn TransportCost>) -> Result<Self, GenericError> {
        Ok(Self { reserved_times_fn: create_reserved_times_fn(reserved_times_index)?, inner })
    }
}

impl TransportCost for DynamicTransportCost {
    fn duration_approx(&self, profile: &Profile, from: Location, to: Location) -> Duration {
        self.inner.duration_approx(profile, from, to)
    }

    fn distance_approx(&self, profile: &Profile, from: Location, to: Location) -> Distance {
        self.inner.distance_approx(profile, from, to)
    }

    fn duration(&self, route: &Route, from: Location, to: Location, travel_time: TravelTime) -> Duration {
        let duration = self.inner.duration(route, from, to, travel_time);

        let time_window = match travel_time {
            TravelTime::Arrival(arrival) => TimeWindow::new(arrival - duration, arrival),
            TravelTime::Departure(departure) => TimeWindow::new(departure, departure + duration),
        };

        (self.reserved_times_fn)(route, &time_window)
            .map_or(duration, |reserved_time| duration + reserved_time.duration)
    }

    fn distance(&self, route: &Route, from: Location, to: Location, travel_time: TravelTime) -> Distance {
        self.inner.distance(route, from, to, travel_time)
    }

    fn size(&self) -> usize {
        self.inner.size()
    }
}

/// Provides way to calculate transport costs using precomputed per-actor costs.
/// Reserved time is still applied at runtime to keep correctness with breaks.
pub struct PrecomputedActorCostTransportCost {
    reserved_times_fn: ReservedTimesFn,
    inner: Arc<dyn TransportCost>,
    actor_index: HashMap<Arc<Actor>, usize>,
    base_costs: Vec<Vec<Cost>>,
    durations: Vec<Vec<Duration>>,
    distances: Vec<Vec<Distance>>,
    size: usize,
    use_precomputed: bool,
}

/// Summarizes memory footprint of precomputed transport tables.
pub struct PrecomputedActorCostStats {
    /// Number of profiles with precomputed matrices.
    pub profile_count: usize,
    /// Matrix dimension (locations count).
    pub size: usize,
    /// Bytes used by precomputed durations.
    pub duration_bytes: usize,
    /// Bytes used by precomputed distances.
    pub distance_bytes: usize,
    /// Bytes used by precomputed per-actor base costs.
    pub base_cost_bytes: usize,
}

impl PrecomputedActorCostTransportCost {
    /// Creates a new instance of `PrecomputedActorCostTransportCost`.
    pub fn new(
        reserved_times_index: ReservedTimesIndex,
        inner: Arc<dyn TransportCost>,
        actors: Vec<Arc<Actor>>,
        use_precomputed: bool,
    ) -> Result<Self, GenericError> {
        let reserved_times_fn = create_reserved_times_fn(reserved_times_index)?;
        let size = inner.size();

        let max_profile = actors.iter().map(|actor| actor.vehicle.profile.index).max().unwrap_or(0);
        let mut durations = vec![Vec::new(); max_profile + 1];
        let mut distances = vec![Vec::new(); max_profile + 1];

        for actor in actors.iter() {
            let profile = &actor.vehicle.profile;
            if !durations.get(profile.index).is_some_and(|data| !data.is_empty()) {
                let mut profile_durations = Vec::with_capacity(size * size);
                let mut profile_distances = Vec::with_capacity(size * size);
                for from in 0..size {
                    for to in 0..size {
                        profile_durations.push(inner.duration_approx(profile, from, to));
                        profile_distances.push(inner.distance_approx(profile, from, to));
                    }
                }
                durations[profile.index] = profile_durations;
                distances[profile.index] = profile_distances;
            }
        }

        let mut actor_index = HashMap::with_capacity(actors.len());
        let mut base_costs = Vec::with_capacity(actors.len());

        for (idx, actor) in actors.into_iter().enumerate() {
            actor_index.insert(actor.clone(), idx);

            let rate_distance = actor.driver.costs.per_distance + actor.vehicle.costs.per_distance;
            let rate_time = actor.driver.costs.per_driving_time + actor.vehicle.costs.per_driving_time;

            let mut costs = Vec::with_capacity(size * size);
            let profile_idx = actor.vehicle.profile.index;
            if let (Some(profile_distances), Some(profile_durations)) =
                (distances.get(profile_idx), durations.get(profile_idx))
            {
                if !profile_distances.is_empty() && !profile_durations.is_empty() {
                    for idx in 0..profile_distances.len() {
                        costs.push(profile_distances[idx] * rate_distance + profile_durations[idx] * rate_time);
                    }
                } else {
                    for from in 0..size {
                        for to in 0..size {
                            let distance = inner.distance_approx(&actor.vehicle.profile, from, to);
                            let duration = inner.duration_approx(&actor.vehicle.profile, from, to);
                            costs.push(distance * rate_distance + duration * rate_time);
                        }
                    }
                }
            }
            base_costs.push(costs);
        }

        Ok(Self { reserved_times_fn, inner, actor_index, base_costs, durations, distances, size, use_precomputed })
    }

    fn get_base_cost(&self, route: &Route, from: Location, to: Location) -> Cost {
        self.actor_index
            .get(&route.actor)
            .and_then(|idx| self.base_costs.get(*idx))
            .and_then(|costs| costs.get(from * self.size + to))
            .copied()
            .unwrap_or_else(|| self.inner.cost(route, from, to, TravelTime::Departure(0.)))
    }

    fn get_precomputed_duration(&self, profile: &Profile, from: Location, to: Location) -> Option<Duration> {
        self.durations
            .get(profile.index)
            .filter(|data| !data.is_empty())
            .and_then(|durations| durations.get(from * self.size + to))
            .copied()
    }

    fn get_precomputed_distance(&self, profile: &Profile, from: Location, to: Location) -> Option<Distance> {
        self.distances
            .get(profile.index)
            .filter(|data| !data.is_empty())
            .and_then(|distances| distances.get(from * self.size + to))
            .copied()
    }

    /// Returns a summary of precomputed table sizes.
    pub fn stats(&self) -> PrecomputedActorCostStats {
        let duration_len = self.durations.iter().map(|data| data.len()).sum::<usize>();
        let distance_len = self.distances.iter().map(|data| data.len()).sum::<usize>();
        let base_cost_len = self.base_costs.iter().map(|data| data.len()).sum::<usize>();
        let profile_count = self.durations.iter().filter(|data| !data.is_empty()).count();

        PrecomputedActorCostStats {
            profile_count,
            size: self.size,
            duration_bytes: duration_len * mem::size_of::<Duration>(),
            distance_bytes: distance_len * mem::size_of::<Distance>(),
            base_cost_bytes: base_cost_len * mem::size_of::<Cost>(),
        }
    }

    fn get_reserved_extra_duration(
        &self,
        route: &Route,
        travel_time: TravelTime,
        base_duration: Duration,
    ) -> Duration {
        let time_window = match travel_time {
            TravelTime::Arrival(arrival) => TimeWindow::new(arrival - base_duration, arrival),
            TravelTime::Departure(departure) => TimeWindow::new(departure, departure + base_duration),
        };

        (self.reserved_times_fn)(route, &time_window).map_or(0., |reserved_time| reserved_time.duration)
    }
}

impl TransportCost for PrecomputedActorCostTransportCost {
    fn cost(&self, route: &Route, from: Location, to: Location, travel_time: TravelTime) -> Cost {
        let base_cost = self.get_base_cost(route, from, to);
        let base_duration = if self.use_precomputed {
            self.get_precomputed_duration(&route.actor.vehicle.profile, from, to)
                .unwrap_or_else(|| self.inner.duration(route, from, to, travel_time))
        } else {
            self.inner.duration(route, from, to, travel_time)
        };
        let extra_duration = self.get_reserved_extra_duration(route, travel_time, base_duration);

        let rate_time = route.actor.driver.costs.per_driving_time + route.actor.vehicle.costs.per_driving_time;
        base_cost + extra_duration * rate_time
    }

    fn duration_approx(&self, profile: &Profile, from: Location, to: Location) -> Duration {
        self.get_precomputed_duration(profile, from, to).unwrap_or_else(|| self.inner.duration_approx(profile, from, to))
    }

    fn distance_approx(&self, profile: &Profile, from: Location, to: Location) -> Distance {
        self.get_precomputed_distance(profile, from, to)
            .unwrap_or_else(|| self.inner.distance_approx(profile, from, to))
    }

    fn duration(&self, route: &Route, from: Location, to: Location, travel_time: TravelTime) -> Duration {
        let base_duration = if self.use_precomputed {
            self.get_precomputed_duration(&route.actor.vehicle.profile, from, to)
                .unwrap_or_else(|| self.inner.duration(route, from, to, travel_time))
        } else {
            self.inner.duration(route, from, to, travel_time)
        };
        base_duration + self.get_reserved_extra_duration(route, travel_time, base_duration)
    }

    fn distance(&self, route: &Route, from: Location, to: Location, travel_time: TravelTime) -> Distance {
        if self.use_precomputed {
            self.get_precomputed_distance(&route.actor.vehicle.profile, from, to)
                .unwrap_or_else(|| self.inner.distance(route, from, to, travel_time))
        } else {
            self.inner.distance(route, from, to, travel_time)
        }
    }

    fn size(&self) -> usize {
        self.inner.size()
    }
}

/// Optimizes reserved time schedules by rescheduling it to earlier time (e.g. to avoid transit stops,
/// reduce waiting time).
pub(crate) fn optimize_reserved_times_schedule(route: &mut Route, reserved_times_fn: &ReservedTimesFn) {
    // NOTE run in this order as reducing waiting time can be also applied on top of avoiding travel time
    avoid_reserved_time_when_driving(route, reserved_times_fn);
    reduce_waiting_by_reserved_time(route, reserved_times_fn);
}

fn avoid_reserved_time_when_driving(route: &mut Route, reserved_times_fn: &ReservedTimesFn) {
    // NOTE assume reserved times has no intersection
    let schedule_shifts = route
        .tour
        .legs()
        .filter_map(|(leg, idx)| match &leg {
            &[from, to] => Some((from, to, idx)),
            _ => None,
        })
        .filter_map(|(from, to, idx)| {
            let travel_tw = TimeWindow::new(from.schedule.departure, to.schedule.arrival);
            reserved_times_fn(route, &travel_tw).map(|reserved_time| (idx, from, reserved_time))
        })
        .filter(|(_, from, reserved_time)| from.schedule.departure > reserved_time.time.start)
        .map(|(idx, _, reserved_time)| (idx, reserved_time.duration))
        .collect::<Vec<_>>();

    schedule_shifts.into_iter().for_each(|(idx, duration)| {
        route.tour.get_mut(idx).unwrap().schedule.departure += duration;
    });
}

fn reduce_waiting_by_reserved_time(_route: &mut Route, _reserved_times_fn: &ReservedTimesFn) {
    // TODO: could be added if necessary, but it should be thought carefully to keep solution feasibility
}

/// Creates a reserved time function from reserved time index.
pub(crate) fn create_reserved_times_fn(
    reserved_times_index: ReservedTimesIndex,
) -> Result<ReservedTimesFn, GenericError> {
    if reserved_times_index.is_empty() {
        return Ok(Arc::new(|_, _| None));
    }

    let reserved_times = reserved_times_index.into_iter().try_fold(
        HashMap::<_, ReservedTimesEntry>::new(),
        |mut acc, (actor, mut times)| {
            // NOTE do not allow different types to simplify interval searching
            let are_same_types = times.windows(2).all(|pair| {
                if let [ReservedTimeSpan { time: a, .. }, ReservedTimeSpan { time: b, .. }] = pair {
                    matches!(
                        (a, b),
                        (TimeSpan::Window(_), TimeSpan::Window(_)) | (TimeSpan::Offset(_), TimeSpan::Offset(_))
                    )
                } else {
                    false
                }
            });

            if !are_same_types {
                return Err("has reserved types of different time span types".to_string());
            }

            times.sort_by(|ReservedTimeSpan { time: a, .. }, ReservedTimeSpan { time: b, .. }| {
                let (a, b) = match (a, b) {
                    (TimeSpan::Window(a), TimeSpan::Window(b)) => (a.start, b.start),
                    (TimeSpan::Offset(a), TimeSpan::Offset(b)) => (a.start, b.start),
                    _ => unreachable!(),
                };
                a.total_cmp(&b)
            });
            let has_no_intersections = times.windows(2).all(|pair| {
                if let [ReservedTimeSpan { time: a, .. }, ReservedTimeSpan { time: b, .. }] = pair {
                    !a.intersects(0., &b.to_time_window(0.))
                } else {
                    false
                }
            });

            if has_no_intersections {
                let (indices, intervals): (Vec<_>, Vec<_>) = times
                    .into_iter()
                    .map(|span| {
                        let start = match &span.time {
                            TimeSpan::Window(time) => time.end,
                            TimeSpan::Offset(time) => time.end,
                        };

                        (start as u64, span)
                    })
                    .unzip();
                let (min_start, max_end) = intervals.iter().fold(
                    (Timestamp::MAX, Timestamp::MIN),
                    |(min_start, max_end), reserved_time| {
                        let start = match &reserved_time.time {
                            TimeSpan::Window(time) => time.end,
                            TimeSpan::Offset(time) => time.end,
                        };
                        let end = start + reserved_time.duration;
                        (min_start.min(start), max_end.max(end))
                    },
                );
                acc.insert(actor, ReservedTimesEntry { indices, intervals, min_start, max_end });

                Ok(acc)
            } else {
                Err("reserved times have intersections".to_string())
            }
        },
    )?;

    // NOTE: this function considers only latest time from reserved time
    //       reserved_time.time.start is ignored and should be handled by post processing
    Ok(Arc::new(move |route: &Route, time_window: &TimeWindow| {
        reserved_times.get(&route.actor).and_then(|entry| {
            let offset = route.tour.start().map(|a| a.schedule.departure).unwrap_or(0.);

            // NOTE map external absolute time window to time span's start/end
            let (interval_start, interval_end) = match entry.intervals.first().map(|rt| &rt.time) {
                Some(TimeSpan::Offset(_)) => (time_window.start - offset, time_window.end - offset),
                Some(TimeSpan::Window(_)) => (time_window.start, time_window.end),
                _ => unreachable!(),
            };
            if interval_end <= entry.min_start || interval_start >= entry.max_end {
                return None;
            }

            match entry.indices.binary_search(&(interval_start as u64)) {
                Ok(idx) => entry.intervals.get(idx),
                Err(idx) => (idx.max(1) - 1..=idx) // NOTE left (earliest) wins
                    .map(|idx| entry.intervals.get(idx))
                    .find(|reserved_time| {
                        reserved_time.is_some_and(|reserved_time| {
                            let (reserved_start, reserved_end) = match &reserved_time.time {
                                TimeSpan::Offset(to) => (to.end, to.end + reserved_time.duration),
                                TimeSpan::Window(tw) => (tw.end, tw.end + reserved_time.duration),
                            };

                            // NOTE use exclusive intersection
                            interval_start < reserved_end && reserved_start < interval_end
                        })
                    })
                    .flatten(),
            }
            .map(|reserved_time| reserved_time.to_reserved_time_window(offset))
        })
    }))
}
