from datetime import datetime

from app.schemas import Trip
from app.tools.delay_model import predict_delay_risk, train
from app.tools.fleet import analyse_trip, distance_to_route_km, haversine_km
from app.tools.retrieval import chunk_markdown, get_index, lookup_lane_rate
from app.tools.vision import Detection, DwellCalculator, clean_plate
from evals.dataset import build_fleet_scenarios


def test_haversine_hyderabad_chennai():
    assert 510 < haversine_km((17.385, 78.4867), (13.0827, 80.2707)) < 530  # straight line, road is ~630


def test_distance_to_route():
    route = [(17.0, 78.0), (17.0, 79.0)]
    assert distance_to_route_km((17.0, 78.5), route) < 0.01
    assert 10 < distance_to_route_km((17.1, 78.5), route) < 12


def test_chunking_keeps_table_header_with_each_row():
    chunks = chunk_markdown(
        "c", "# Rates\n| From | To | Cap | Rate |\n|---|---|---|---|\n| A | B | 10 T | 100 |\n| A | C | 10 T | 200 |"
    )
    assert len(chunks) == 2 and all("| From |" in c.text for c in chunks)


def test_hybrid_search_finds_detention_clause():
    hits = get_index().search("detention charges per day", k=1, source_prefix="acme-steel")
    assert "Rs 2,500" in hits[0].chunk.text


def test_lane_rate_picks_smallest_fitting_truck_and_aliases():
    small = lookup_lane_rate("acme-steel", "Hyd", "Madras", 9500)
    big = lookup_lane_rate("acme-steel", "Hyderabad", "Chennai", 18000)
    assert (small["rate"], small["capacity_t"]) == (42000, 10)
    assert (big["rate"], big["capacity_t"]) == (64000, 20)
    assert small["citation"]["id"].startswith("acme-steel#")


def test_lane_rate_is_tenant_scoped():
    assert lookup_lane_rate("deccan-pharma", "Hyderabad", "Mumbai", 9000) is None


def _scenario(name):
    return next(s for s in build_fleet_scenarios() if s.name == name)


def test_fleet_detects_each_anomaly():
    kinds = lambda name: {a.kind for a in analyse_trip(Trip(**_scenario(name).trip)).anomalies}  # noqa: E731
    assert kinds("on_track") == set()
    assert "unplanned_stop" in kinds("long_unplanned_stop")
    assert "route_deviation" in kinds("route_deviation")
    assert "eta_breach" in kinds("running_late")


def test_delay_model_beats_naive_rule(tmp_path):
    report = train(tmp_path / "m.joblib", n=2000)
    assert report["auc"] > 0.9
    assert report["accuracy"] > report["naive_rule_accuracy"]


def test_delay_risk_monotonic_in_time_left():
    base = {
        "remaining_km": 400,
        "avg_speed_kmph": 40,
        "stop_count": 1,
        "stop_minutes": 30,
        "max_deviation_km": 1,
        "hour_of_day": 14,
    }
    assert predict_delay_risk({**base, "hours_to_deadline": 4}) > predict_delay_risk({**base, "hours_to_deadline": 20})


def test_plate_cleanup():
    assert clean_plate("TS O9 UB 452I") == "TS09UB4521"
    assert clean_plate("T5 09 UB 4521") == "TS09UB4521"
    assert clean_plate("garbage") is None


def test_dwell_from_gate_crossings_with_plate_voting():
    calc = DwellCalculator(fps=1.0, start_time=datetime(2026, 10, 20, 8, 0), gate_y=300)
    path = [(0, 100), (1, 250), (2, 350), (3, 500)] + [(f, 500) for f in range(4, 1800, 60)] + [(1800, 350), (1801, 250)]
    for i, (frame, y) in enumerate(path):
        plate = "TS09UB4521" if i % 4 else "TS09UB4S21"  # occasional misread
        calc.update(Detection(frame, 7, "truck", (0, y - 50, 100, y + 50), plate, 0.9))
    (event,) = calc.events()
    assert event.vehicle_no == "TS09UB4521"
    assert event.entered_at == "2026-10-20T08:00:02"
    assert event.dwell_minutes == 30.0
