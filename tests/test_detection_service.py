from src.detection_service import DetectionService, InMemoryAlertStore


class FakeModel:
    def __init__(self, pred: int, proba: list[float]):
        self.pred = pred
        self.proba = proba

    def predict(self, _df):
        return [self.pred]

    def predict_proba(self, _df):
        return [self.proba]


def test_attack_prediction_creates_alert():
    service = DetectionService(FakeModel(pred=1, proba=[0.01, 0.99]), InMemoryAlertStore())
    result = service.predict_from_features({"duration": 1.0}, profile="balanced")

    assert result.prediction == "Attack"
    assert result.alert is not None
    assert result.alert.severity == "critical"


def test_low_confidence_normal_marked_suspicious():
    service = DetectionService(FakeModel(pred=0, proba=[0.58, 0.42]), InMemoryAlertStore())
    result = service.predict_from_features({"duration": 1.0}, profile="strict")

    assert result.prediction == "Normal"
    assert result.suspicious is True
    assert result.alert is not None
    assert result.alert.reason == "below_confidence_threshold"
