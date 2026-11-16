"""The desktop web app's endpoints, against the demo broker."""
import time

import pytest
from fastapi.testclient import TestClient

from agent.web import create_web_app, demo_setup


@pytest.fixture()
def client(tmp_path):
    journal = str(tmp_path / "stan.db")
    broker, loader = demo_setup(journal)
    app = create_web_app(journal, broker, loader, demo=True)
    return TestClient(app), broker


def _wait(c):
    for _ in range(100):
        job = c.get("/paper/job").json()
        if not job["running"]:
            return job
        time.sleep(0.05)
    raise AssertionError("job never finished")


def test_page_and_live(client):
    c, _ = client
    assert "Stan" in c.get("/").text
    live = c.get("/live").json()
    assert live["paper"] and live["strategy"] == "medium" and live["positions"][0]["symbol"] == "SPY"
    assert live["equity"] > 0 and live["kill_floor"] == pytest.approx(live["start_equity"] * 0.8)
    assert c.get("/live/chart", params={"symbol": "SPY"}).json()["points"]
    assert c.get("/live/chart", params={"symbol": "../x"}).status_code == 400


def test_choose_tier_preview_then_send(client):
    c, broker = client
    assert c.post("/paper/use", json={"strategy": "high"}).json()["strategy"] == "high"
    assert c.post("/paper/use", json={"strategy": "yolo"}).status_code == 400
    before = dict(broker.orders)
    c.post("/paper/run", json={"dry_run": True})
    job = _wait(c)
    assert job["error"] is None and job["strategy"] == "high"
    assert broker.orders == before                       # a preview sends nothing
    c.post("/paper/run", json={"dry_run": False})
    job = _wait(c)
    assert job["error"] is None and len(broker.orders) > len(before)
    assert c.get("/live").json()["strategy"] == "high"


def test_pause_stops_orders_and_resume_restores(client):
    c, broker = client
    c.post("/paper/use", json={"strategy": "high"})
    c.post("/paper/pause")
    assert c.get("/live").json()["halted"]
    n = len(broker.orders)
    c.post("/paper/run", json={"dry_run": False})
    job = _wait(c)
    assert job["halted"] and len(broker.orders) == n
    c.post("/paper/resume")
    assert not c.get("/live").json()["halted"]


def test_other_websites_cannot_change_anything(client):
    c, _ = client
    r = c.post("/paper/use", json={"strategy": "high"}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    assert c.post("/paper/run", json={"dry_run": False}, headers={"Origin": "http://evil.example"}).status_code == 403


def test_demo_label_is_really_hidden_in_real_mode():
    """The pill's display style must not override the hidden attribute."""
    from agent.web import WEBAPP
    html = (WEBAPP / "index.html").read_text()
    assert "[hidden] { display: none !important; }" in html
