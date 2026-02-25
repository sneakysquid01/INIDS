from src import run_end_to_end_demo as demo


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def json(self):
        return self._payload


class FakeSession:
    def get(self, url, timeout=10):
        if url.endswith('/api/health'):
            return FakeResponse(payload={"status": "ok"})
        if '/api/alerts' in url:
            return FakeResponse(payload={"count": 0, "alerts": []})
        return FakeResponse(payload={})

    def post(self, url, json=None, timeout=10):
        if url.endswith('/api/ingest'):
            return FakeResponse(payload={"queued": 2, "queue_size": 2})
        if url.endswith('/api/ingest/process'):
            return FakeResponse(payload={"processed": 2, "results": []})
        return FakeResponse(payload={})


def test_run_demo():
    out = demo.run_demo("http://localhost:5000", session=FakeSession())
    assert out["health"]["status"] == "ok"
    assert out["ingest"]["queued"] == 2
    assert out["processed"]["processed"] == 2
