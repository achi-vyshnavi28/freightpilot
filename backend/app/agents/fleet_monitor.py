"""Fleet monitor agent: anomalies come from code, the response plan from the agent.

Offline it follows a fixed playbook. With Claude it plans its own sequence over
the same tools (it may check yard events before deciding a long stop is just
unloading, for example), and every action it takes is recorded on the alert.
"""

from __future__ import annotations

import json

from ..db import FleetAlert, session_scope
from ..llm import ClaudeLLM, LLMRefusal, Tool
from ..schemas import Trip
from ..tools.delay_model import predict_delay_risk
from ..tools.fleet import analyse_trip
from ..tools.vision import get_yard_events
from ..tracing import Tracer
from .common import call_tool, decision
from .comms import run_comms

AGENT = "fleet_monitor"
RISK_NOTIFY = 0.6

SYSTEM = (
    "You monitor trucks for an Indian transporter. Use analyse_trip first, then predict_delay_risk with its features. "
    "Check get_yard_events before treating a long stop at a warehouse as a problem. Take the smallest set of actions "
    "that protects the delivery: update_eta when the ETA moves, request_driver_call for unexplained long stops or "
    "route deviations, raise_alert for anything a manager must see. Only call notify_client=true on update_eta when "
    "the consignee will actually be late. Finish with one sentence stating what you did and why."
)


class _Actions:
    def __init__(self) -> None:
        self.items: list[dict] = []
        self.notify: dict | None = None

    def update_eta(self, eta: str, reason: str, notify_client: bool) -> str:
        self.items.append({"action": "update_eta", "eta": eta, "reason": reason})
        if notify_client:
            self.notify = {"eta": eta, "reason": reason}
        return "eta updated"

    def raise_alert(self, severity: str, detail: str) -> str:
        self.items.append({"action": "raise_alert", "severity": severity, "detail": detail})
        return "alert raised"

    def request_driver_call(self, reason: str) -> str:
        self.items.append({"action": "request_driver_call", "reason": reason})
        return "driver call queued for the ops desk"


def _tool_defs(trip: Trip, acts: _Actions) -> list[Tool]:
    def s(props: dict) -> dict:
        return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}

    return [
        Tool(
            "analyse_trip",
            "Detect stops, route deviation and ETA breach on this trip's GPS trace.",
            s({}),
            lambda: analyse_trip(trip).to_dict(),
        ),
        Tool(
            "predict_delay_risk",
            "Probability (0-1) the trip misses its deadline. Pass the features object from analyse_trip as JSON.",
            s({"features_json": {"type": "string"}}),
            lambda features_json: predict_delay_risk(json.loads(features_json)),
        ),
        Tool(
            "get_yard_events",
            "Gate camera entry/exit events for this vehicle (dwell time at warehouses).",
            s({}),
            lambda: get_yard_events(trip.vehicle_no),
        ),
        Tool(
            "update_eta",
            "Record a revised ETA. notify_client drafts a delay notice for human approval.",
            s({"eta": {"type": "string"}, "reason": {"type": "string"}, "notify_client": {"type": "boolean"}}),
            acts.update_eta,
        ),
        Tool(
            "raise_alert",
            "Show an alert on the ops dashboard.",
            s({"severity": {"type": "string", "enum": ["low", "medium", "high"]}, "detail": {"type": "string"}}),
            acts.raise_alert,
        ),
        Tool(
            "request_driver_call",
            "Ask the ops desk to phone the driver.",
            s({"reason": {"type": "string"}}),
            acts.request_driver_call,
        ),
    ]


def _playbook(trip: Trip, tracer: Tracer, acts: _Actions) -> tuple[dict, float]:
    analysis = call_tool(tracer, AGENT, "analyse_trip", lambda: analyse_trip(trip).to_dict())
    risk = (
        call_tool(tracer, AGENT, "predict_delay_risk", predict_delay_risk, features=analysis["features"])
        if analysis["features"]
        else 0.0
    )
    yard = call_tool(tracer, AGENT, "get_yard_events", get_yard_events, vehicle_no=trip.vehicle_no)
    at_yard = any(e.get("entered_at") and not e.get("exited_at") for e in yard)
    for a in analysis["anomalies"]:
        if a["kind"] == "unplanned_stop" and at_yard:
            decision(tracer, AGENT, "stop_explained_by_yard", a)
            continue
        if a["kind"] in {"unplanned_stop", "route_deviation"} and a["severity"] == "high":
            call_tool(tracer, AGENT, "request_driver_call", acts.request_driver_call, reason=f"{a['kind']}: {a['detail']}")
        call_tool(tracer, AGENT, "raise_alert", acts.raise_alert, severity=a["severity"], detail=f"{a['kind']}: {a['detail']}")
    late = any(a["kind"] == "eta_breach" for a in analysis["anomalies"]) or risk >= RISK_NOTIFY
    if analysis.get("eta"):
        reason = next((a["detail"] for a in analysis["anomalies"] if a["kind"] != "gps_gap"), "traffic and stops on route")
        call_tool(tracer, AGENT, "update_eta", acts.update_eta, eta=analysis["eta"], reason=reason, notify_client=late)
    return analysis, risk


def run_fleet_check(trip: Trip, tracer: Tracer, llm: ClaudeLLM | None) -> dict:
    acts = _Actions()
    summary = ""
    if llm is not None:
        try:
            summary, _ = llm.run_tools(
                agent=AGENT,
                system=SYSTEM,
                prompt=f"Check trip {trip.trip_id}: {trip.vehicle_no} {trip.origin} -> {trip.destination}, deadline {trip.deadline}.",
                tools=_tool_defs(trip, acts),
                tracer=tracer,
            )
        except LLMRefusal as exc:
            decision(tracer, AGENT, "llm_refused", str(exc))
            acts = _Actions()
    if not acts.items and not summary:
        analysis, risk = _playbook(trip, tracer, acts)
    else:
        analysis = analyse_trip(trip).to_dict()
        risk = predict_delay_risk(analysis["features"]) if analysis["features"] else 0.0

    approval_id = None
    if acts.notify:
        approval_id = run_comms(
            trip.client_id,
            "delay_notice",
            {"vehicle_no": trip.vehicle_no, "origin": trip.origin, "destination": trip.destination, **acts.notify},
            tracer,
            llm,
            trip_id=trip.trip_id,
        )

    with session_scope() as s:
        for a in analysis["anomalies"] or [{"kind": "on_track", "severity": "low", "detail": "no anomalies"}]:
            s.add(
                FleetAlert(
                    trip_id=trip.trip_id,
                    client_id=trip.client_id,
                    vehicle_no=trip.vehicle_no,
                    kind=a["kind"],
                    severity=a["severity"],
                    detail=a["detail"],
                    actions=acts.items,
                    delay_risk=risk,
                )
            )
    return {
        "trip_id": trip.trip_id,
        "analysis": analysis,
        "delay_risk": risk,
        "actions": acts.items,
        "approval_id": approval_id,
        "agent_summary": summary or None,
        "run_id": tracer.run_id,
    }
