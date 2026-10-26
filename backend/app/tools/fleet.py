"""GPS trip analysis. Plain code finds the anomalies; the fleet agent decides what to do."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..schemas import Trip

EARTH_KM = 6371.0088
STOP_SPEED_KMPH = 5.0
UNPLANNED_STOP_MIN = 45.0  # drivers get a meal/rest stop; longer than this is flagged
DEVIATION_KM = 8.0


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(h))


def _point_segment_km(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    """Distance from p to segment ab, using a local equirectangular projection (fine below ~500 km)."""
    lat0 = math.radians((a[0] + b[0] + p[0]) / 3)

    def xy(q):
        return (math.radians(q[1]) * math.cos(lat0) * EARTH_KM, math.radians(q[0]) * EARTH_KM)

    (px, py), (ax, ay), (bx, by) = xy(p), xy(a), xy(b)
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def distance_to_route_km(p: tuple[float, float], route: list[tuple[float, float]]) -> float:
    if len(route) == 1:
        return haversine_km(p, route[0])
    return min(_point_segment_km(p, route[i], route[i + 1]) for i in range(len(route) - 1))


def remaining_route_km(p: tuple[float, float], route: list[tuple[float, float]]) -> float:
    """Distance along the planned route from the nearest segment to the destination."""
    best_i = min(range(len(route) - 1), key=lambda i: _point_segment_km(p, route[i], route[i + 1]))
    total = haversine_km(p, route[best_i + 1])
    for i in range(best_i + 1, len(route) - 1):
        total += haversine_km(route[i], route[i + 1])
    return total


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass
class Anomaly:
    kind: str  # unplanned_stop | route_deviation | eta_breach | gps_gap
    severity: str
    detail: str
    at: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class TripAnalysis:
    trip_id: str
    anomalies: list[Anomaly] = field(default_factory=list)
    features: dict = field(default_factory=dict)
    eta: str | None = None

    def to_dict(self) -> dict:
        return {
            "trip_id": self.trip_id,
            "anomalies": [a.to_dict() for a in self.anomalies],
            "features": self.features,
            "eta": self.eta,
        }


def analyse_trip(trip: Trip) -> TripAnalysis:
    result = TripAnalysis(trip.trip_id)
    pts = trip.points
    if len(pts) < 2:
        return result
    route = [tuple(p) for p in trip.planned_route]

    stop_start = None
    total_stop_min = 0.0
    stops = 0
    max_dev = 0.0
    for prev, cur in zip(pts, pts[1:], strict=False):
        gap_min = (_ts(cur.ts) - _ts(prev.ts)).total_seconds() / 60
        if gap_min > 60:
            result.anomalies.append(Anomaly("gps_gap", "medium", f"no GPS signal for {gap_min:.0f} min", prev.ts))
        if cur.speed_kmph < STOP_SPEED_KMPH:
            stop_start = stop_start or prev
        elif stop_start is not None:
            dur = (_ts(cur.ts) - _ts(stop_start.ts)).total_seconds() / 60
            total_stop_min += dur
            stops += 1
            if dur >= UNPLANNED_STOP_MIN:
                result.anomalies.append(
                    Anomaly("unplanned_stop", "high" if dur > 120 else "medium", f"stationary {dur:.0f} min", stop_start.ts)
                )
            stop_start = None
        dev = distance_to_route_km((cur.lat, cur.lon), route)
        max_dev = max(max_dev, dev)

    if stop_start is not None:  # still stopped at the last ping
        dur = (_ts(pts[-1].ts) - _ts(stop_start.ts)).total_seconds() / 60
        total_stop_min += dur
        stops += 1
        if dur >= UNPLANNED_STOP_MIN:
            result.anomalies.append(
                Anomaly(
                    "unplanned_stop", "high" if dur > 120 else "medium", f"stationary {dur:.0f} min and counting", stop_start.ts
                )
            )

    if max_dev > DEVIATION_KM:
        result.anomalies.append(
            Anomaly("route_deviation", "high" if max_dev > 25 else "medium", f"{max_dev:.1f} km off planned route", pts[-1].ts)
        )

    last = pts[-1]
    window = [p for p in pts if (_ts(last.ts) - _ts(p.ts)).total_seconds() <= 3 * 3600]
    moving = [p.speed_kmph for p in window if p.speed_kmph >= STOP_SPEED_KMPH]
    avg_speed = sum(moving) / len(moving) if moving else 0.0
    remaining = remaining_route_km((last.lat, last.lon), route)
    hours_left = (_ts(trip.deadline) - _ts(last.ts)).total_seconds() / 3600
    # Indian highway trucks average ~35-45 km/h door to door; assume 40 when unknown.
    eff_speed = avg_speed * 0.8 if avg_speed else 40.0
    hours_needed = remaining / eff_speed
    result.eta = (_ts(last.ts) + timedelta(hours=hours_needed)).isoformat(timespec="minutes")
    if hours_needed > hours_left:
        result.anomalies.append(
            Anomaly("eta_breach", "high", f"needs {hours_needed:.1f} h, {max(hours_left, 0):.1f} h left before deadline", last.ts)
        )

    result.features = {
        "remaining_km": round(remaining, 1),
        "avg_speed_kmph": round(avg_speed, 1),
        "stop_count": stops,
        "stop_minutes": round(total_stop_min, 1),
        "max_deviation_km": round(max_dev, 2),
        "hours_to_deadline": round(hours_left, 2),
        "hour_of_day": _ts(last.ts).hour,
    }
    return result
