#[cfg(test)]
#[path = "../../../tests/unit/construction/enablers/reserved_time_test.rs"]
mod reserved_time_test;

use crate::construction::heuristics::{RouteContext, RouteState};
use crate::models::common::*;
use crate::models::problem::{ActivityCost, Actor, TransportCost, TravelTime};
use crate::models::solution::{Activity, Route};
use rosomaxa::prelude::GenericError;
use rustc_hash::FxHashMap;
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

// Stores route-local resolved reserved windows after route schedule refresh.
custom_tour_state!(pub(crate) ResolvedReservedTimes typeof Vec<ReservedTimeWindow>);

#[derive(Clone)]
struct ReservedTimesEntry {
    // Actor-specific reserved spans, already sorted and validated.
    intervals: ReservedTimesIntervals,
    // Fast rejection bounds in the span's own time domain.
    min_start: Timestamp,
    max_end: Timestamp,
    // Offset spans are interpreted relative to the route departure.
    is_offset: bool,
}

#[derive(Clone)]
enum ReservedTimesIntervals {
    Single(ReservedTimeSpan),
    Multiple(Vec<ReservedTimeSpan>),
}

#[derive(Clone)]
pub(crate) struct ReservedTimes {
    // Actor identity is stable through Arc pointers and avoids string/id lookup in hot paths.
    entries: Arc<FxHashMap<usize, ReservedTimesEntry>>,
}

fn get_reserved_time_window(schedule: &TimeWindow, reserved_time: &ReservedTimeWindow) -> Option<TimeWindow> {
    // A reserved span is charged only when its concrete duration overlaps the checked schedule.
    let reserved_start = reserved_time.time.start;
    let reserved_end = reserved_time.time.end;
    let actual_start = schedule.start.clamp(reserved_start, reserved_end);
    let actual_time = TimeWindow::new(actual_start, actual_start + reserved_time.duration);
    let intersects = if schedule.start == schedule.end {
        actual_time.contains(schedule.start)
    } else {
        actual_time.intersects_exclusive(schedule)
    };

    intersects.then_some(actual_time)
}

fn resolve_reserved_time_window(route: &Route, reserved_time: ReservedTimeWindow) -> Option<ReservedTimeWindow> {
    // Exact reserved times already have a concrete start.
    if reserved_time.time.start == reserved_time.time.end {
        return Some(reserved_time);
    }

    // Flexible reserved intervals are shifted to the latest overlapping activity/leg boundary.
    let latest_start = route
        .tour
        .all_activities()
        .map(|activity| TimeWindow::new(activity.schedule.arrival, activity.schedule.departure))
        .chain(route.tour.legs().filter_map(|(leg, _)| match leg {
            [from, to] => Some(TimeWindow::new(from.schedule.departure, to.schedule.arrival)),
            _ => None,
        }))
        .filter(|schedule| schedule.end >= reserved_time.time.start && schedule.start <= reserved_time.time.end)
        .map(|schedule| schedule.end.min(reserved_time.time.end))
        .max_by(|left, right| left.total_cmp(right));

    latest_start
        .map(|start| ReservedTimeWindow { time: TimeWindow::new(start, start), duration: reserved_time.duration })
}

impl ReservedTimes {
    fn new(reserved_times_index: ReservedTimesIndex) -> Result<Self, GenericError> {
        // Normalize the public index once so runtime checks only do lookup and interval matching.
        let entries = reserved_times_index.into_iter().try_fold(
            FxHashMap::<_, ReservedTimesEntry>::default(),
            |mut acc, (actor, mut times)| {
                // NOTE do not allow different types to simplify interval searching.
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
                    let (min_start, max_end) =
                        times.iter().fold((Timestamp::MAX, Timestamp::MIN), |(min_start, max_end), reserved_time| {
                            let (start, end) = match &reserved_time.time {
                                TimeSpan::Window(time) => (time.start, time.end),
                                TimeSpan::Offset(time) => (time.start, time.end),
                            };
                            (min_start.min(start), max_end.max(end + reserved_time.duration))
                        });
                    let is_offset =
                        times.first().is_some_and(|reserved_time| matches!(reserved_time.time, TimeSpan::Offset(_)));
                    let intervals = match times.len() {
                        1 => ReservedTimesIntervals::Single(times.pop().unwrap()),
                        _ => ReservedTimesIntervals::Multiple(times),
                    };
                    acc.insert(get_actor_key(&actor), ReservedTimesEntry { intervals, min_start, max_end, is_offset });

                    Ok(acc)
                } else {
                    Err("reserved times have intersections".to_string())
                }
            },
        )?;

        Ok(Self { entries: Arc::new(entries) })
    }

    fn find(&self, route: &Route, time_window: &TimeWindow) -> Option<ReservedTimeWindow> {
        // This is the correctness-preserving slow path: resolve against the current route schedule.
        self.entries.get(&get_actor_key(&route.actor)).and_then(|entry| {
            let offset = route.tour.start().map(|a| a.schedule.departure).unwrap_or(0.);

            // NOTE map external absolute time window to time span's start/end.
            let (interval_start, interval_end) = if entry.is_offset {
                (time_window.start - offset, time_window.end - offset)
            } else {
                (time_window.start, time_window.end)
            };
            if interval_end <= entry.min_start || interval_start >= entry.max_end {
                return None;
            }

            let check_reserved_time = |reserved_time: &ReservedTimeSpan| {
                let reserved_time = reserved_time.to_reserved_time_window(offset);
                let reserved_time = resolve_reserved_time_window(route, reserved_time)?;

                get_reserved_time_window(time_window, &reserved_time).map(|_| reserved_time)
            };

            match &entry.intervals {
                ReservedTimesIntervals::Single(interval) => check_reserved_time(interval),
                ReservedTimesIntervals::Multiple(intervals) => intervals.iter().find_map(check_reserved_time),
            }
        })
    }

    fn resolve_route(&self, route: &Route) -> Option<Vec<ReservedTimeWindow>> {
        // Pre-resolve all actor reserved spans once per accepted route state.
        self.entries.get(&get_actor_key(&route.actor)).map(|entry| {
            let offset = route.tour.start().map(|a| a.schedule.departure).unwrap_or(0.);
            let resolve = |reserved_time: &ReservedTimeSpan| {
                resolve_reserved_time_window(route, reserved_time.to_reserved_time_window(offset))
            };

            match &entry.intervals {
                ReservedTimesIntervals::Single(interval) => resolve(interval).into_iter().collect(),
                ReservedTimesIntervals::Multiple(intervals) => intervals.iter().filter_map(resolve).collect(),
            }
        })
    }
}

#[inline]
pub(crate) fn get_reserved_extra_duration_from_resolved(
    reserved_times: &[ReservedTimeWindow],
    travel_time: TravelTime,
    base_duration: Duration,
) -> Duration {
    // The resolved route cache lets hot transport evaluation skip route-wide schedule scans.
    let time_window = match travel_time {
        TravelTime::Arrival(arrival) => TimeWindow::new(arrival - base_duration, arrival),
        TravelTime::Departure(departure) => TimeWindow::new(departure, departure + base_duration),
    };

    reserved_times
        .iter()
        .find_map(|reserved_time| get_reserved_time_window(&time_window, reserved_time).map(|_| reserved_time.duration))
        .unwrap_or(0.)
}

/// Provides way to calculate activity costs which might contain reserved time.
pub struct DynamicActivityCost {
    reserved_times: ReservedTimes,
}

impl DynamicActivityCost {
    /// Creates a new instance of `DynamicActivityCost` with given reserved time function.
    pub fn new(reserved_times_index: ReservedTimesIndex) -> Result<Self, GenericError> {
        Ok(Self { reserved_times: ReservedTimes::new(reserved_times_index)? })
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

        // Activity service can also consume a reserved span, not only driving.
        self.reserved_times.find(route, &schedule).map_or(ControlFlow::Continue(departure), |reserved_time| {
            let reserved_tw =
                get_reserved_time_window(&schedule, &reserved_time).expect("reserved time must intersect");

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

        let value = self.reserved_times.find(route, &schedule).map_or(arrival, |reserved_time| {
            if get_reserved_time_window(&schedule, &reserved_time).is_some() {
                (arrival - reserved_time.duration).max(activity.place.time.start)
            } else {
                arrival
            }
        });

        ControlFlow::Continue(value)
    }
}

/// Provides way to calculate transport costs which might contain reserved time.
pub struct DynamicTransportCost {
    reserved_times: ReservedTimes,
    inner: Arc<dyn TransportCost>,
}

impl DynamicTransportCost {
    /// Creates a new instance of `DynamicTransportCost`.
    pub fn new(reserved_times_index: ReservedTimesIndex, inner: Arc<dyn TransportCost>) -> Result<Self, GenericError> {
        Ok(Self { reserved_times: ReservedTimes::new(reserved_times_index)?, inner })
    }

    pub(crate) fn resolve_reserved_times(&self, route: &Route) -> Option<Vec<ReservedTimeWindow>> {
        self.reserved_times.resolve_route(route)
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
        // Generic wrapper: delegate base travel to inner transport, then add reserved time.
        let duration = self.inner.duration(route, from, to, travel_time);

        let time_window = match travel_time {
            TravelTime::Arrival(arrival) => TimeWindow::new(arrival - duration, arrival),
            TravelTime::Departure(departure) => TimeWindow::new(departure, departure + duration),
        };

        self.reserved_times.find(route, &time_window).map_or(duration, |reserved_time| {
            duration + get_reserved_time_window(&time_window, &reserved_time).map_or(0., |_| reserved_time.duration)
        })
    }

    #[inline]
    fn cost_without_reserved_time(&self, route: &Route, from: Location, to: Location, travel_time: TravelTime) -> Cost {
        self.inner.cost(route, from, to, travel_time)
    }

    #[inline]
    fn duration_without_reserved_time(
        &self,
        route: &Route,
        from: Location,
        to: Location,
        travel_time: TravelTime,
    ) -> Duration {
        self.inner.duration(route, from, to, travel_time)
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
    // Shared reserved-time index; route-specific resolved windows are cached in RouteState.
    reserved_times: ReservedTimes,
    // Fallback transport for unsupported/time-aware cases and original matrix access.
    inner: Arc<dyn TransportCost>,
    // Maps Actor Arc pointers to rows in base_costs.
    actor_index: FxHashMap<usize, usize>,
    // Fully actor-specific base transport cost matrix: distance and duration rates already applied.
    base_costs: Vec<Vec<Cost>>,
    // Profile-specific duration matrix, stored unscaled and scaled per actor profile at lookup time.
    durations: Vec<Vec<Duration>>,
    // Profile-specific distance matrix.
    distances: Vec<Vec<Distance>>,
    // Matrix dimension used for flat index calculation.
    size: usize,
    // Disabled for time-aware matrices where departure/arrival time must be delegated to inner.
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
        let reserved_times = ReservedTimes::new(reserved_times_index)?;
        let size = inner.size();

        // Precompute one duration/distance table per profile. Actor-specific scaling is applied later.
        let max_profile = actors.iter().map(|actor| actor.vehicle.profile.index).max().unwrap_or(0);
        let mut durations = vec![Vec::new(); max_profile + 1];
        let mut distances = vec![Vec::new(); max_profile + 1];

        for actor in actors.iter() {
            let profile = &actor.vehicle.profile;
            if !durations.get(profile.index).is_some_and(|data| !data.is_empty()) {
                let unscaled_profile = Profile::new(profile.index, Some(1.));
                let mut profile_durations = Vec::with_capacity(size * size);
                let mut profile_distances = Vec::with_capacity(size * size);
                for from in 0..size {
                    for to in 0..size {
                        profile_durations.push(inner.duration_approx(&unscaled_profile, from, to));
                        profile_distances.push(inner.distance_approx(&unscaled_profile, from, to));
                    }
                }
                durations[profile.index] = profile_durations;
                distances[profile.index] = profile_distances;
            }
        }

        let mut actor_index = FxHashMap::default();
        let mut base_costs = Vec::with_capacity(actors.len());

        for (idx, actor) in actors.into_iter().enumerate() {
            // Precompute one cost matrix per actor because driver/vehicle cost rates can differ.
            actor_index.insert(get_actor_key(&actor), idx);

            let rate_distance = actor.driver.costs.per_distance + actor.vehicle.costs.per_distance;
            let rate_time = actor.driver.costs.per_driving_time + actor.vehicle.costs.per_driving_time;

            let mut costs = Vec::with_capacity(size * size);
            let profile_idx = actor.vehicle.profile.index;
            if let (Some(profile_distances), Some(profile_durations)) =
                (distances.get(profile_idx), durations.get(profile_idx))
            {
                if !profile_distances.is_empty() && !profile_durations.is_empty() {
                    for idx in 0..profile_distances.len() {
                        costs.push(
                            profile_distances[idx] * rate_distance
                                + profile_durations[idx] * actor.vehicle.profile.scale * rate_time,
                        );
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

        Ok(Self { reserved_times, inner, actor_index, base_costs, durations, distances, size, use_precomputed })
    }

    #[inline]
    fn get_base_cost(&self, route: &Route, from: Location, to: Location) -> Cost {
        // Base cost excludes reserved-time penalties; those are added separately when needed.
        let matrix_idx = from * self.size + to;
        let actor_key = get_actor_key(&route.actor);

        self.actor_index
            .get(&actor_key)
            .and_then(|idx| self.base_costs.get(*idx))
            .and_then(|costs| costs.get(matrix_idx))
            .copied()
            .unwrap_or_else(|| self.inner.cost(route, from, to, TravelTime::Departure(0.)))
    }

    #[inline]
    fn get_precomputed_duration(&self, profile: &Profile, from: Location, to: Location) -> Option<Duration> {
        self.durations
            .get(profile.index)
            .filter(|data| !data.is_empty())
            .and_then(|durations| durations.get(from * self.size + to))
            .map(|duration| duration * profile.scale)
    }

    #[inline]
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

    fn get_reserved_extra_duration(&self, route: &Route, travel_time: TravelTime, base_duration: Duration) -> Duration {
        let time_window = match travel_time {
            TravelTime::Arrival(arrival) => TimeWindow::new(arrival - base_duration, arrival),
            TravelTime::Departure(departure) => TimeWindow::new(departure, departure + base_duration),
        };

        self.reserved_times.find(route, &time_window).map_or(0., |reserved_time| {
            get_reserved_time_window(&time_window, &reserved_time).map_or(0., |_| reserved_time.duration)
        })
    }

    pub(crate) fn resolve_reserved_times(&self, route: &Route) -> Option<Vec<ReservedTimeWindow>> {
        self.reserved_times.resolve_route(route)
    }

    #[inline]
    fn get_base_duration(&self, route: &Route, from: Location, to: Location, travel_time: TravelTime) -> Duration {
        // Keep exact inner transport for time-aware matrices or missing precomputed data.
        if self.use_precomputed {
            self.get_precomputed_duration(&route.actor.vehicle.profile, from, to)
                .unwrap_or_else(|| self.inner.duration(route, from, to, travel_time))
        } else {
            self.inner.duration(route, from, to, travel_time)
        }
    }
}

impl TransportCost for PrecomputedActorCostTransportCost {
    fn cost(&self, route: &Route, from: Location, to: Location, travel_time: TravelTime) -> Cost {
        // Public cost remains fully correct: cached base cost plus dynamic reserved-time cost.
        let base_cost = self.get_base_cost(route, from, to);
        let base_duration = self.get_base_duration(route, from, to, travel_time);
        let extra_duration = self.get_reserved_extra_duration(route, travel_time, base_duration);

        let rate_time = route.actor.driver.costs.per_driving_time + route.actor.vehicle.costs.per_driving_time;
        base_cost + extra_duration * rate_time
    }

    fn duration_approx(&self, profile: &Profile, from: Location, to: Location) -> Duration {
        self.get_precomputed_duration(profile, from, to)
            .unwrap_or_else(|| self.inner.duration_approx(profile, from, to))
    }

    fn distance_approx(&self, profile: &Profile, from: Location, to: Location) -> Distance {
        self.get_precomputed_distance(profile, from, to)
            .unwrap_or_else(|| self.inner.distance_approx(profile, from, to))
    }

    fn duration(&self, route: &Route, from: Location, to: Location, travel_time: TravelTime) -> Duration {
        // Public duration remains fully correct when no route-state cache is available.
        let base_duration = self.get_base_duration(route, from, to, travel_time);
        base_duration + self.get_reserved_extra_duration(route, travel_time, base_duration)
    }

    #[inline]
    fn cost_without_reserved_time(&self, route: &Route, from: Location, to: Location, _: TravelTime) -> Cost {
        // Hot evaluators use this with the route-state reserved cache to avoid resolving twice.
        self.get_base_cost(route, from, to)
    }

    #[inline]
    fn duration_without_reserved_time(
        &self,
        route: &Route,
        from: Location,
        to: Location,
        travel_time: TravelTime,
    ) -> Duration {
        // Hot evaluators add reserved-time extra duration from RouteState.
        self.get_base_duration(route, from, to, travel_time)
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

pub(crate) fn update_resolved_reserved_times_state(route_ctx: &mut RouteContext, transport: &dyn TransportCost) {
    // RouteState is cleared and rebuilt on accept_route_state, so this cache is naturally invalidated.
    if let Some(reserved_times) = resolve_reserved_times(transport, route_ctx.route()) {
        route_ctx.state_mut().set_resolved_reserved_times(reserved_times);
    }
}

#[inline]
pub(crate) fn duration_with_reserved_times(
    transport: &dyn TransportCost,
    route: &Route,
    reserved_times: Option<&[ReservedTimeWindow]>,
    from: Location,
    to: Location,
    travel_time: TravelTime,
) -> Duration {
    // Fast path used by insertion evaluation: base duration plus already-resolved route break windows.
    if let Some(reserved_times) = reserved_times {
        let base_duration = transport.duration_without_reserved_time(route, from, to, travel_time);

        base_duration + get_reserved_extra_duration_from_resolved(reserved_times, travel_time, base_duration)
    } else {
        transport.duration(route, from, to, travel_time)
    }
}

#[inline]
pub(crate) fn cost_with_reserved_times(
    transport: &dyn TransportCost,
    route: &Route,
    reserved_times: Option<&[ReservedTimeWindow]>,
    from: Location,
    to: Location,
    travel_time: TravelTime,
) -> Cost {
    // Cost uses the same resolved-window cache and only adds the reserved driving-time rate.
    if let Some(reserved_times) = reserved_times {
        let base_cost = transport.cost_without_reserved_time(route, from, to, travel_time);
        let base_duration = transport.duration_without_reserved_time(route, from, to, travel_time);
        let extra_duration = get_reserved_extra_duration_from_resolved(reserved_times, travel_time, base_duration);
        let rate_time = route.actor.driver.costs.per_driving_time + route.actor.vehicle.costs.per_driving_time;

        base_cost + extra_duration * rate_time
    } else {
        transport.cost(route, from, to, travel_time)
    }
}

fn resolve_reserved_times(transport: &dyn TransportCost, route: &Route) -> Option<Vec<ReservedTimeWindow>> {
    // Only transports with reserved-time awareness can populate the route-state cache.
    transport
        .as_any()
        .downcast_ref::<PrecomputedActorCostTransportCost>()
        .and_then(|transport| transport.resolve_reserved_times(route))
        .or_else(|| {
            transport
                .as_any()
                .downcast_ref::<DynamicTransportCost>()
                .and_then(|transport| transport.resolve_reserved_times(route))
        })
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

    let reserved_times = ReservedTimes::new(reserved_times_index)?;

    Ok(Arc::new(move |route: &Route, time_window: &TimeWindow| reserved_times.find(route, time_window)))
}

#[inline]
fn get_actor_key(actor: &Arc<Actor>) -> usize {
    Arc::as_ptr(actor) as usize
}
