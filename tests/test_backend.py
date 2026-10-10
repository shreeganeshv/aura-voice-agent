"""Offline tests: the LLM is mocked, so these verify tools, policy logic and the agent loop."""
import json, sys, pathlib
from types import SimpleNamespace as NS
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
import data, main

# ---- policy / tool logic ----
def test_normalize():
    for raw in ["ORD-101", "ord 101", "ORD101", "o r d - 1 0 1", "101"]:
        assert data.normalize_order_id(raw) == "ORD-101"
    assert data.normalize_order_id("") is None and data.normalize_order_id("abc") is None

def test_lookup_errors():
    assert data.get_order_details(None)["error"] == "MISSING_ORDER_ID"
    assert data.get_order_details("ORD-999")["error"] == "ORDER_NOT_FOUND"
    assert data.get_order_details("ord 101")["order"]["status"] == "Out for Delivery"

def test_eligibility():
    assert data.check_eligibility("ORD-103", "cancel")["eligible"] is True
    assert data.check_eligibility("ORD-101", "cancel")["eligible"] is False
    assert "refuse delivery" in data.check_eligibility("ORD-101", "cancel")["reason"]
    assert data.check_eligibility("ORD-102", "return")["eligible"] is False
    assert data.check_eligibility("ORD-102", "damaged_report")["eligible"] is False
    assert data.check_eligibility("ORD-101", "return")["eligible"] is False

def test_cancel_is_per_session():
    ov = {}
    assert data.cancel_order("ORD-103", ov)["success"] is True
    assert data.get_order_details("ORD-103", ov)["order"]["status"] == "Cancelled"
    assert data.ORDERS["ORD-103"]["status"] == "Processing"        # shared data untouched
    assert data.cancel_order("ORD-103", ov)["success"] is False     # can't cancel twice
    assert data.cancel_order("ORD-101", {})["success"] is False

# ---- agent loop with a scripted fake LLM ----
def tool_msg(name, args, id="call_1"):
    return NS(choices=[NS(message=NS(content=None, tool_calls=[
        NS(id=id, function=NS(name=name, arguments=json.dumps(args)))]))])
def text_msg(t):
    return NS(choices=[NS(message=NS(content=t, tool_calls=None))])

@pytest.fixture
def client(monkeypatch):
    main.SESSIONS.clear()
    return TestClient(main.app)

def script(monkeypatch, responses):
    calls = []
    async def fake(messages, **kw):
        calls.append(json.loads(json.dumps(messages, default=str)))
        return responses.pop(0)
    monkeypatch.setattr(main, "llm_call", fake)
    return calls

def test_order_lookup_flow(client, monkeypatch):
    calls = script(monkeypatch, [tool_msg("get_order_details", {"order_id": "ORD-101"}),
                                 text_msg("It's out for delivery, arriving by 6 PM.")])
    sid = client.post("/api/session").json()["session_id"]
    r = client.post("/api/chat", json={"session_id": sid, "message": "Where is ORD-101?"}).json()
    assert "6 PM" in r["reply"]
    assert r["tool_calls"][0]["result"]["order"]["tracking_id"] == "BD-982103"
    # second LLM call must include the tool result
    assert any(m.get("role") == "tool" for m in calls[1])

def test_invalid_order_returns_structured_error(client, monkeypatch):
    script(monkeypatch, [tool_msg("get_order_details", {"order_id": "ORD-999"}), text_msg("Couldn't find it.")])
    sid = client.post("/api/session").json()["session_id"]
    r = client.post("/api/chat", json={"session_id": sid, "message": "ORD-999"}).json()
    assert r["tool_calls"][0]["result"]["error"] == "ORDER_NOT_FOUND"

def test_empty_input_skips_llm(client, monkeypatch):
    script(monkeypatch, [])  # any LLM call would raise IndexError
    sid = client.post("/api/session").json()["session_id"]
    r = client.post("/api/chat", json={"session_id": sid, "message": " "}).json()
    assert "repeat" in r["reply"]

def test_llm_failure_degrades_gracefully_and_keeps_history_valid(client, monkeypatch):
    async def boom(*a, **k): raise RuntimeError("groq down")
    monkeypatch.setattr(main, "llm_call", boom)
    sid = client.post("/api/session").json()["session_id"]
    r = client.post("/api/chat", json={"session_id": sid, "message": "hello there"})
    assert r.status_code == 200 and r.json()["reply"] == main.FALLBACK_REPLY
    assert len(main.SESSIONS[sid].messages) == 2  # system + greeting only

def test_cancel_and_summary(client, monkeypatch):
    summary = json.dumps({"customer_intent": "ORDER_CANCELLATION", "order_id": "ord 103",
                          "resolution_status": "RESOLVED", "call_summary": "Cancelled ORD-103."})
    script(monkeypatch, [tool_msg("cancel_order", {"order_id": "ORD-103"}), text_msg("Done, it's cancelled."),
                         text_msg(summary)])
    sid = client.post("/api/session").json()["session_id"]
    client.post("/api/chat", json={"session_id": sid, "message": "yes please cancel ORD-103"})
    s = client.post("/api/summarize", json={"session_id": sid}).json()
    assert s["outcome"]["order_id"] == "ORD-103" and s["outcome"]["resolution_status"] == "RESOLVED"
    assert [t["speaker"] for t in s["transcript"]] == ["agent", "customer", "agent"]

def test_summary_fallback_on_bad_json(client, monkeypatch):
    script(monkeypatch, [tool_msg("get_order_details", {"order_id": "ORD-102"}), text_msg("Delivered."),
                         text_msg("not json at all")])
    sid = client.post("/api/session").json()["session_id"]
    client.post("/api/chat", json={"session_id": sid, "message": "status of ORD-102"})
    s = client.post("/api/summarize", json={"session_id": sid}).json()
    assert s["outcome"]["order_id"] == "ORD-102" and s["outcome"]["resolution_status"] == "UNRESOLVED"

def test_summary_empty_call_and_unknown_session(client):
    sid = client.post("/api/session").json()["session_id"]
    assert client.post("/api/summarize", json={"session_id": sid}).json()["outcome"]["resolution_status"] == "UNRESOLVED"
    assert client.post("/api/summarize", json={"session_id": "nope"}).json()["outcome"] is None

def test_orders_endpoint(client):
    assert [o["order_id"] for o in client.get("/api/orders").json()["orders"]] == ["ORD-101", "ORD-102", "ORD-103"]
