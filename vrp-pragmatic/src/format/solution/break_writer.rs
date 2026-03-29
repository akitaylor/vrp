use super::*;
use std::cmp::Ordering;
use vrp_core::construction::enablers::ReservedTimesIndex;
use vrp_core::models::common::{Cost, TimeWindow};
use vrp_core::models::solution::Route;
use vrp_core::prelude::Float;

/// Converts reserved time duration applied to activity or travel time to break activity.
pub(super) fn insert_reserved_times_as_breaks(
    route: &Route,
    tour: &mut Tour,
    reserved_times_index: &ReservedTimesIndex,
) {
    let shift_time = route
        .tour
        .start()
        .zip(route.tour.end())
        .map(|(start, end)| TimeWindow::new(start.schedule.departure, end.schedule.arrival))
        .expect("empty tour");

    reserved_times_index
        .get(&route.actor)
        .iter()
        .flat_map(|times| times.iter())
        .map(|reserved_time| reserved_time.to_reserved_time_window(shift_time.start))
        .filter_map(|reserved_time| resolve_reserved_time_window(route, reserved_time))
        .for_each(|reserved_time| {
            // NOTE scan and insert a new stop if necessary
            let break_info = tour.stops.windows(2).enumerate().find_map(|(leg_idx, stops)| {
                if let &[prev, next] = &stops {
                    let travel_tw =
                        TimeWindow::new(parse_time(&prev.schedule().departure), parse_time(&next.schedule().arrival));

                    if let Some(reserved_tw) = get_reserved_time_window(&travel_tw, &reserved_time) {
                        return Some(BreakInsertion::TransitBreakUsed {
                            leg_idx,
                            load: prev.load().clone(),
                            break_tw: reserved_tw,
                        });
                    }
                }

                None
            });

            if let Some(BreakInsertion::TransitBreakUsed { leg_idx, load, break_tw }) = break_info.clone() {
                tour.stops.insert(
                    leg_idx + 1,
                    Stop::Transit(TransitStop {
                        time: ApiSchedule {
                            arrival: format_time(break_tw.start),
                            departure: format_time(break_tw.end),
                        },
                        load,
                        activities: vec![],
                    }),
                )
            }

            let break_time = reserved_time.duration as i64;
            let break_cost = break_time as Float * route.actor.vehicle.costs.per_service_time;

            if let Some((stop_idx, reserved_tw)) = tour.stops.iter().enumerate().find_map(|(stop_idx, stop)| {
                let stop_tw = TimeWindow::new(parse_time(&stop.schedule().arrival), parse_time(&stop.schedule().departure));
                get_reserved_time_window(&stop_tw, &reserved_time).map(|reserved_tw| (stop_idx, reserved_tw))
            }) {
                let stop = tour.stops.get_mut(stop_idx).expect("expected stop");
                let stop_tw = TimeWindow::new(parse_time(&stop.schedule().arrival), parse_time(&stop.schedule().departure));

                insert_break(
                    (stop, stop_tw, stop_idx),
                    (break_time, break_cost, break_info.clone()),
                    &reserved_tw,
                    &mut tour.statistic,
                );
            }

            tour.statistic.times.break_time += break_time;
        });
}

/// Inserts a break activity into the tour and updates schedules and statistics.
fn insert_break(
    stop_data: (&mut Stop, TimeWindow, usize),
    break_data: (i64, Cost, Option<BreakInsertion>),
    reserved_tw: &TimeWindow,
    statistic: &mut Statistic,
) {
    let (stop, stop_tw, _) = stop_data;
    let (break_time, break_cost, _) = break_data;
    let break_idx = stop
        .activities()
        .iter()
        .enumerate()
        .filter_map(|(activity_idx, activity)| {
            let activity_tw = activity.time.as_ref().map_or(stop_tw.clone(), |interval| {
                TimeWindow::new(parse_time(&interval.start), parse_time(&interval.end))
            });

            if activity_tw.intersects(reserved_tw) { Some(activity_idx + 1) } else { None }
        })
        .next()
        .unwrap_or(stop.activities().len());

    let activities = match stop {
        Stop::Point(point) => {
            statistic.cost += break_cost;
            &mut point.activities
        }
        Stop::Transit(transit) => {
            statistic.times.driving -= break_time;
            &mut transit.activities
        }
    };

    let activity_time = reserved_tw;

    activities.insert(
        break_idx,
        ApiActivity {
            job_id: "break".to_string(),
            activity_type: "break".to_string(),
            location: None,
            time: Some(Interval { start: format_time(activity_time.start), end: format_time(activity_time.end) }),
            job_tag: None,
            commute: None,
        },
    );

    activities.iter_mut().enumerate().filter(|(idx, _)| *idx != break_idx).for_each(|(_, activity)| {
        if let Some(time) = &mut activity.time {
            let start = parse_time(&time.start);
            let end = parse_time(&time.end);
            let overlap = TimeWindow::new(start, end).overlapping(reserved_tw);

            if let Some(overlap) = overlap {
                let extra_time = reserved_tw.end - overlap.end + overlap.duration();
                time.end = format_time(end + extra_time);
            }
        }
    });

    activities.sort_by(|a, b| match (&a.time, &b.time) {
        (Some(a), Some(b)) => parse_time(&a.start).total_cmp(&parse_time(&b.start)),
        (Some(_), None) => Ordering::Greater,
        (None, Some(_)) => Ordering::Less,
        (None, None) => Ordering::Equal,
    })
}

#[derive(Clone)]
enum BreakInsertion {
    TransitBreakUsed { leg_idx: usize, load: Vec<i32>, break_tw: TimeWindow },
}

fn get_reserved_time_window(schedule: &TimeWindow, reserved_time: &vrp_core::construction::enablers::ReservedTimeWindow) -> Option<TimeWindow> {
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

fn resolve_reserved_time_window(
    route: &Route,
    reserved_time: vrp_core::construction::enablers::ReservedTimeWindow,
) -> Option<vrp_core::construction::enablers::ReservedTimeWindow> {
    if reserved_time.time.start == reserved_time.time.end {
        return Some(reserved_time);
    }

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
        .map(|start| vrp_core::construction::enablers::ReservedTimeWindow { time: TimeWindow::new(start, start), duration: reserved_time.duration })
}
