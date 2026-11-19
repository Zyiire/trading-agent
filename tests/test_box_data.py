"""Box minute-bar store: resume from the last bar and never ask for the newest 20 minutes."""
from datetime import date, datetime, timezone

from agent.stan.box import data


def test_backfill_resumes_and_respects_sip_delay(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(client, symbol, start, end, feed="sip"):
        calls.append((start, end, feed))
        first = int(start.timestamp())
        return [(first, 1.0, 2.0, 0.5, 1.5, 100, 1.2), (first + 60, 1.5, 2.0, 1.0, 1.8, 50, None)]

    monkeypatch.setattr(data, "fetch_alpaca", fake_fetch)
    path = tmp_path / "box.db"
    now = datetime(2024, 3, 4, 15, 0, tzinfo=timezone.utc)
    assert data.backfill(None, path, since=date(2024, 3, 1), now=now, log=lambda *_: None) == 2
    assert calls[0][0] == datetime(2024, 3, 1, tzinfo=timezone.utc) and calls[0][2] == "sip"
    assert calls[-1][1] == datetime(2024, 3, 4, 14, 40, tzinfo=timezone.utc)

    calls.clear()
    later = datetime(2024, 3, 4, 16, 0, tzinfo=timezone.utc)
    data.backfill(None, path, since=date(2024, 3, 1), now=later, log=lambda *_: None)
    assert calls[0][0] == datetime.fromtimestamp(int(datetime(2024, 3, 1, tzinfo=timezone.utc).timestamp()) + 120,
                                                 tz=timezone.utc)
    df = data.load(path)
    assert list(df.columns) == ["o", "h", "l", "c"] and len(df) == 4 and str(df.index.tz) == "America/New_York"


def test_backfill_up_to_date_fetches_nothing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(data, "fetch_alpaca", lambda *a, **k: calls.append(a) or [])
    assert data.backfill(None, tmp_path / "b.db", since=date(2024, 3, 4),
                         now=datetime(2024, 3, 4, 0, 10, tzinfo=timezone.utc), log=lambda *_: None) == 0
    assert calls == []
