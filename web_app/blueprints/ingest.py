"""Ingestion pipeline endpoints — Step 39, third blueprint extraction."""
import logging

from flask import Blueprint, current_app, jsonify, request

from src.auth.decorators import require_roles
from src.log_parsers import parse_suricata_eve_flow, parse_zeek_conn_log
from src.pipeline.backpressure import BackpressureLevel

logger = logging.getLogger(__name__)
ingest_bp = Blueprint("ingest", __name__)


@ingest_bp.route("/api/ingest", methods=["POST"])
@require_roles("analyst")
def api_ingest():
    import web_app.app as _m
    payload = request.get_json(silent=True) or {}
    source = payload.get("source", "ingestion_api")
    rows = payload.get("rows")
    pipeline_started = _m._ensure_pipeline_started()

    backpressure = getattr(current_app._get_current_object(), "_pipeline_backpressure", None)
    if pipeline_started and backpressure is not None and backpressure.level == BackpressureLevel.SHEDDING:
        return jsonify({"error": "pipeline_backpressure", "pipeline": _m._pipeline_status()}), 503

    try:
        if isinstance(rows, list):
            if not rows:
                return jsonify({"error": "rows cannot be empty"}), 400
            if pipeline_started:
                added = _m._stream_ingest_records(rows, source=source)
            else:
                added = _m.ingestion_service.enqueue_batch(rows, source=source)
        else:
            features = payload.get("features")
            if not isinstance(features, dict) or not features:
                return jsonify({"error": "provide either non-empty 'rows' or 'features'"}), 400
            if pipeline_started:
                added = _m._stream_ingest_records([features], source=source)
            else:
                _m.ingestion_service.enqueue_record(features, source=source)
                added = 1

        _m.metrics_service.inc("ingested_total", amount=added)
        return jsonify({
            "queued": added,
            "queue_size": _m.ingestion_queue.size(),
            "pipeline": _m._pipeline_status(),
        })
    except Exception:
        return jsonify({"error": "invalid_request"}), 400


@ingest_bp.route("/api/ingest/log", methods=["POST"])
@require_roles("analyst")
def api_ingest_log():
    import web_app.app as _m
    payload = request.get_json(silent=True) or {}
    source_type = str(payload.get("type", "zeek")).lower()
    records = payload.get("records")
    pipeline_started = _m._ensure_pipeline_started()
    backpressure = getattr(current_app._get_current_object(), "_pipeline_backpressure", None)
    if pipeline_started and backpressure is not None and backpressure.level == BackpressureLevel.SHEDDING:
        return jsonify({"error": "pipeline_backpressure", "pipeline": _m._pipeline_status()}), 503
    if not isinstance(records, list) or not records:
        return jsonify({"error": "'records' must be a non-empty list"}), 400

    try:
        transformed = []
        for rec in records:
            if source_type == "zeek":
                transformed.append(parse_zeek_conn_log(rec))
            elif source_type == "suricata":
                transformed.append(parse_suricata_eve_flow(rec))
            else:
                return jsonify({"error": "type must be 'zeek' or 'suricata'"}), 400

        if pipeline_started:
            added = _m._stream_ingest_records(transformed, source=f"{source_type}_log")
        else:
            added = _m.ingestion_service.enqueue_batch(transformed, source=f"{source_type}_log")
        _m.metrics_service.inc("ingested_total", amount=added)
        return jsonify({"queued": added, "queue_size": _m.ingestion_queue.size(), "type": source_type})
    except Exception:
        return jsonify({"error": "invalid_request"}), 400


@ingest_bp.route("/api/ingest/process", methods=["POST"])
@require_roles("analyst")
def api_ingest_process():
    import web_app.app as _m
    if not _m.ensure_detection_service():
        return jsonify({"error": "No trained model found"}), 503

    payload = request.get_json(silent=True) or {}
    try:
        max_items = int(payload.get("max_items", 50))
    except (ValueError, TypeError):
        return jsonify({"error": "max_items must be an integer"}), 400
    max_items = max(1, min(max_items, 500))

    def _handler(features, source):
        result = _m.detection_service.predict_from_features(
            features,
            profile="balanced",
            source_ip=source,
        )
        _m.metrics_service.inc("processed_ingestion_total")
        result_payload = result.to_dict()
        result_payload["prevention_action"] = None
        return result_payload

    processed = _m.ingestion_service.process_all(_handler, max_items=max_items)
    return jsonify({
        "processed": len(processed),
        "queue_size": _m.ingestion_queue.size(),
        "results": processed,
    })
