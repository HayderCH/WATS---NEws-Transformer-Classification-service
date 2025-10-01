from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
client.headers.update({"x-api-key": "test-key"})


def test_review_queue_filters_and_pagination():
    # Enqueue a few items with varying predicted labels
    payloads = [
        {
            "text": f"ambiguous text {i}",
            "predicted_label": "TECH" if i % 2 == 0 else "SPORTS",
            "confidence_score": 0.2,
            "confidence_margin": 0.01,
            "model_version": "test",
        }
        for i in range(4)
    ]
    for p in payloads:
        r = client.post("/review/enqueue", json=p)
        assert r.status_code == 200
        assert r.json()["queued"] is True

    # Filter by predicted_label
    r = client.get("/review/queue", params={"predicted_label": "TECH", "limit": 2})
    assert r.status_code == 200
    items = r.json()
    assert len(items) <= 2
    assert all(it["predicted_label"] == "TECH" for it in items)
    assert all("created_at" in it for it in items)

    # Pagination using offset
    r1 = client.get("/review/queue", params={"limit": 2, "offset": 0})
    r2 = client.get("/review/queue", params={"limit": 2, "offset": 2})
    assert r1.status_code == 200 and r2.status_code == 200
    ids1 = [it["id"] for it in r1.json()]
    ids2 = [it["id"] for it in r2.json()]
    assert set(ids1).isdisjoint(set(ids2))

    # Sorting controls
    r_desc = client.get(
        "/review/queue",
        params={"limit": 3, "sort_by": "created_at", "sort_order": "desc"},
    )
    assert r_desc.status_code == 200
    desc_items = r_desc.json()
    if len(desc_items) >= 2:
        assert desc_items[0]["created_at"] >= desc_items[1]["created_at"]


def test_review_enqueue_requires_api_key():
    unauth_client = TestClient(app)
    payload = {
        "text": "unauthorized item",
        "predicted_label": "TECH",
        "confidence_score": 0.1,
        "confidence_margin": 0.05,
        "model_version": "test",
    }
    resp = unauth_client.post("/review/enqueue", json=payload)
    assert resp.status_code == 401


def test_metrics_counters_present():
    # Trigger a couple of requests to generate metrics
    client.get("/health")
    client.get("/labels")
    m = client.get("/metrics")
    assert m.status_code == 200
    data = m.json()
    assert "request_counters" in data
    assert "latency_ms" in data
    latencies = data["latency_ms"]
    assert isinstance(latencies, dict)
    health_metrics = latencies.get("/health")
    assert health_metrics is not None
    assert "count" in health_metrics
    if health_metrics.get("recent"):
        assert "window" in health_metrics["recent"]
