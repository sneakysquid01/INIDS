"""Dashboard, demo, and perception layer API endpoints — Step 39."""
import logging
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from src.auth.decorators import require_roles

logger = logging.getLogger(__name__)
dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/api/dashboard/metrics", methods=["GET"])
@require_roles("viewer")
def api_dashboard_metrics():
    import web_app.app as _m
    try:
        payload = _m._build_dashboard_metrics_payload()
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        return jsonify(payload)
    except Exception as e:
        logger.exception("Error retrieving dashboard metrics")
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/api/dashboard/refresh", methods=["POST"])
@require_roles("analyst")
def api_dashboard_refresh():
    try:
        return jsonify({
            "status": "refreshed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        logger.exception("Error refreshing dashboard")
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/api/demo/start", methods=["POST"])
@require_roles("admin")
def api_demo_start():
    try:
        logger.info("Demo mode started")
        return jsonify({
            "status": "started",
            "message": "Demo traffic simulation started"
        })
    except Exception as e:
        logger.exception("Error starting demo")
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/api/demo/stop", methods=["POST"])
@require_roles("admin")
def api_demo_stop():
    try:
        logger.info("Demo mode stopped")
        return jsonify({
            "status": "stopped",
            "message": "Demo traffic simulation stopped"
        })
    except Exception as e:
        logger.exception("Error stopping demo")
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/api/perception/pulse", methods=["GET"])
@require_roles("analyst")
def api_perception_pulse():
    import web_app.app as _m
    try:
        pulse = _m.live_system_pulse.get_pulse_status()
        return jsonify(pulse), 200
    except Exception as e:
        logger.exception("Error getting system pulse")
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/api/perception/pulse/timeseries/<metric>", methods=["GET"])
@require_roles("analyst")
def api_perception_pulse_timeseries(metric):
    import web_app.app as _m
    try:
        series = _m.live_system_pulse.get_time_series(metric)
        return jsonify({
            "metric": metric,
            "data": series,
            "count": len(series)
        }), 200
    except Exception as e:
        logger.exception(f"Error getting timeseries for {metric}")
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/api/perception/confidence/<detection_id>", methods=["GET"])
@require_roles("analyst")
def api_perception_confidence(detection_id):
    import web_app.app as _m
    try:
        breakdown = _m.confidence_breakdown_engine.get_breakdown(detection_id)
        if not breakdown:
            return jsonify({"error": "Detection not found"}), 404
        return jsonify(breakdown), 200
    except Exception as e:
        logger.exception(f"Error getting confidence breakdown for {detection_id}")
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/api/perception/attack-story/<attack_id>", methods=["GET"])
@require_roles("analyst")
def api_perception_attack_story(attack_id):
    import web_app.app as _m
    try:
        story = _m.attack_story_engine.get_attack_story(attack_id)
        return jsonify(story), 200
    except Exception as e:
        logger.exception(f"Error getting attack story for {attack_id}")
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/api/perception/attack-stories", methods=["GET"])
@require_roles("analyst")
def api_perception_attack_stories():
    import web_app.app as _m
    try:
        limit = request.args.get("limit", 10, type=int)
        stories = _m.attack_story_engine.get_recent_stories(limit=limit)
        return jsonify({
            "stories": stories,
            "count": len(stories)
        }), 200
    except Exception as e:
        logger.exception("Error getting attack stories")
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/api/perception/feature-importance", methods=["GET"])
@require_roles("analyst")
def api_perception_feature_importance():
    import web_app.app as _m
    try:
        ranking = _m.confidence_breakdown_engine.get_feature_importance_ranking()
        return jsonify({
            "features": [
                {
                    "rank": i + 1,
                    "feature_name": name,
                    "total_contribution": score
                }
                for i, (name, score) in enumerate(ranking)
            ],
            "count": len(ranking)
        }), 200
    except Exception as e:
        logger.exception("Error getting feature importance")
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/api/perception/integration-status", methods=["GET"])
@require_roles("analyst")
def api_perception_integration_status():
    import web_app.app as _m
    try:
        status = _m.perception_integration.get_status()
        return jsonify(status), 200
    except Exception as e:
        logger.exception("Error getting perception integration status")
        return jsonify({"error": str(e)}), 500
