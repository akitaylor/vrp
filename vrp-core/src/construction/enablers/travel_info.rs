use crate::construction::enablers::{ReservedTimeWindow, ResolvedReservedTimesTourState, duration_with_reserved_times};
use crate::construction::heuristics::{ActivityContext, RouteContext};
use crate::models::common::{Distance, Duration, Timestamp};
use crate::models::problem::{TransportCost, TravelTime};
use crate::models::solution::Activity;

/// Calculates a travel info from prev to next directly.
pub fn calculate_travel(
    route_ctx: &RouteContext,
    activity_ctx: &ActivityContext,
    transport: &dyn TransportCost,
) -> ((Distance, Distance), (Duration, Duration)) {
    let prev = activity_ctx.prev;
    let tar = activity_ctx.target;
    let next = activity_ctx.next;

    let prev_dep = prev.schedule.departure;
    // Keep reserved-time lookup outside individual leg calculations in this hot helper.
    let reserved_times = route_ctx.state().get_resolved_reserved_times().map(Vec::as_slice);

    let (prev_to_tar_dis, prev_to_tar_dur) =
        calculate_travel_leg(route_ctx, reserved_times, prev, tar, prev_dep, transport);

    if let Some(next) = next {
        let tar_dep = prev_dep + prev_to_tar_dur;

        let (tar_to_next_dis, tar_to_next_dur) =
            calculate_travel_leg(route_ctx, reserved_times, tar, next, tar_dep, transport);

        ((prev_to_tar_dis, tar_to_next_dis), (prev_to_tar_dur, tar_to_next_dur))
    } else {
        ((prev_to_tar_dis, Distance::default()), (prev_to_tar_dur, Duration::default()))
    }
}

/// Calculates delta in distance and duration for target activity in given activity context.
pub fn calculate_travel_delta(
    route_ctx: &RouteContext,
    activity_ctx: &ActivityContext,
    transport: &dyn TransportCost,
) -> (Distance, Duration) {
    // NOTE accept some code duplication between methods in that module as they are called often,
    //      generalization might require some redundancy in calculations

    let prev = activity_ctx.prev;
    let tar = activity_ctx.target;
    let next = activity_ctx.next;

    let prev_dep = prev.schedule.departure;
    // RouteState cache is used for all three legs in the local insertion delta.
    let reserved_times = route_ctx.state().get_resolved_reserved_times().map(Vec::as_slice);

    let (prev_to_tar_dis, prev_to_tar_dur) =
        calculate_travel_leg(route_ctx, reserved_times, prev, tar, prev_dep, transport);

    if let Some(next) = next {
        let tar_dep = prev_dep + prev_to_tar_dur;

        let (prev_to_next_dis, prev_to_next_dur) =
            calculate_travel_leg(route_ctx, reserved_times, prev, next, prev_dep, transport);
        let (tar_to_next_dis, tar_to_next_dur) =
            calculate_travel_leg(route_ctx, reserved_times, tar, next, tar_dep, transport);

        (prev_to_tar_dis + tar_to_next_dis - prev_to_next_dis, prev_to_tar_dur + tar_to_next_dur - prev_to_next_dur)
    } else {
        (prev_to_tar_dis, prev_to_tar_dur)
    }
}

/// Calculates a travel leg info.
fn calculate_travel_leg(
    route_ctx: &RouteContext,
    reserved_times: Option<&[ReservedTimeWindow]>,
    first: &Activity,
    second: &Activity,
    departure: Timestamp,
    transport: &dyn TransportCost,
) -> (Distance, Duration) {
    let route = route_ctx.route();
    // Distance is independent of reserved time; duration can include a cached break/service extra.
    let first_to_second_dis =
        transport.distance(route, first.place.location, second.place.location, TravelTime::Departure(departure));
    let first_to_second_dur = duration_with_reserved_times(
        transport,
        route,
        reserved_times,
        first.place.location,
        second.place.location,
        TravelTime::Departure(departure),
    );

    let second_arr = departure + first_to_second_dur;
    let second_wait = (second.place.time.start - second_arr).max(0.);
    let second_dep = second_arr + second_wait + second.place.duration;

    (first_to_second_dis, second_dep - departure)
}
