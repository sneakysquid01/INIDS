from src.metrics_service import MetricsService


def test_metrics_counter_and_format():
    svc = MetricsService()
    svc.inc("requests_total")
    svc.inc("predictions_total", amount=2)

    assert svc.get("requests_total") == 1
    assert svc.get("predictions_total") == 2

    output = svc.as_prometheus()
    assert "inids_requests_total 1" in output
    assert "inids_predictions_total 2" in output
