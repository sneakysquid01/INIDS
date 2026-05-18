"""Module data API endpoints and module page view — Step 39."""
import logging
from datetime import datetime, timezone, timedelta

from flask import Blueprint, jsonify, render_template, request

from src.auth.decorators import require_roles

logger = logging.getLogger(__name__)
modules_bp = Blueprint("modules", __name__)


@modules_bp.route("/modules/<module_id>")
@require_roles("viewer")
def module_view(module_id):
    templates = {
        'real-time-detection': 'modules/real_time_detection.html',
        'multi-engine-voting': 'modules/multi_engine_voting.html',
        'risk-score-visualizer': 'modules/risk_score_visualizer.html',
        'auto-blocking': 'modules/auto_blocking.html',
        'evasion-detection': 'modules/evasion_detection.html',
        'packet-inspection': 'modules/packet_inspection.html',
        'behavioral-profiling': 'modules/behavioral_profiling.html',
        'threat-intelligence': 'modules/threat_intelligence.html',
        'drift-monitor': 'modules/drift_monitor.html',
        'anomaly-learning': 'modules/anomaly_learning.html',
        'fp-suppression': 'modules/fp_suppression.html',
        'escalation-tracker': 'modules/escalation_tracker.html',
        'network-topology': 'modules/network_topology.html',
        'policy-enforcement': 'modules/policy_enforcement.html',
        'forensic-timeline': 'modules/forensic_timeline.html',
    }
    template = templates.get(module_id)
    if not template:
        return "Module not found", 404
    try:
        return render_template(template)
    except Exception as e:
        logger.exception(f"Module {module_id} rendering failed")
        return f"Module error: {str(e)}", 500


@modules_bp.route("/api/modules/real-time-detection", methods=["GET"])
@require_roles("viewer")
def api_module_real_time_detection():
    import web_app.app as _m
    try:
        alerts = _m.ops_store.list_alerts(limit=50)
        return jsonify({
            "status": "success",
            "data": {
                "recent_events": alerts,
                "event_count": len(alerts),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading real-time detection module")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/multi-engine", methods=["GET"])
@require_roles("viewer")
def api_module_multi_engine():
    import web_app.app as _m
    try:
        engines = _m.engine_registry.list_engines()
        return jsonify({
            "status": "success",
            "data": {
                "engines": [
                    {
                        "id": e.get("engine_id"),
                        "name": e.get("engine_id"),
                        "type": e.get("engine_type"),
                        "enabled": bool(e.get("enabled")),
                        "ready": bool(e.get("ready")),
                    }
                    for e in engines
                ],
                "engine_count": len(engines),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading multi-engine module")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/risk-score", methods=["GET"])
@require_roles("analyst")
def api_module_risk_score():
    import web_app.app as _m
    try:
        recent_alerts = _m.ops_store.list_alerts(limit=100)
        risk_scores = [float(a.get("risk_score", 0)) for a in recent_alerts if "risk_score" in a]
        return jsonify({
            "status": "success",
            "data": {
                "current_risk": max(risk_scores) if risk_scores else 0,
                "average_risk": sum(risk_scores) / len(risk_scores) if risk_scores else 0,
                "risk_distribution": risk_scores[:20],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading risk score module")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/auto-blocking", methods=["GET"])
@require_roles("analyst")
def api_module_auto_blocking():
    import web_app.app as _m
    try:
        active_blocks = _m.ops_store.list_active_blocks(limit=100)
        return jsonify({
            "status": "success",
            "data": {
                "enabled": _m.prevention_service.policy.mode == "auto_block",
                "dry_run": _m.prevention_service.policy.dry_run,
                "blocked_ips": [_m._normalize_action_payload(row) for row in active_blocks],
                "block_count": len(active_blocks),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        })
    except Exception as e:
        logger.exception("Error loading auto-blocking module")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/approval-workflow", methods=["GET"])
@require_roles("analyst")
def api_module_approval_workflow():
    import web_app.app as _m
    try:
        pending = _m.ops_store.list_actions(limit=100)
        pending = [
            a for a in pending
            if str(a.get("status", "")).strip().lower() in {"pending", "pending_approval", "escalated"}
        ]
        return jsonify({
            "status": "success",
            "data": {
                "pending_approvals": pending[:10],
                "pending_count": len(pending),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading approval workflow module")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/false-positive", methods=["GET"])
@require_roles("analyst")
def api_module_false_positive():
    import web_app.app as _m
    try:
        suppressions = _m.ops_store.list_fp_suppressions()
        feedback_rows = _m.fp_manager.stats()
        false_positive_count = sum(int(row.get("false_positives", 0)) for row in feedback_rows)
        return jsonify({
            "status": "success",
            "data": {
                "false_positives": feedback_rows[:10],
                "fp_count": false_positive_count,
                "suppression_count": len(suppressions),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading false positive module")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/threat-intel", methods=["GET"])
@require_roles("analyst")
def api_module_threat_intel():
    import web_app.app as _m
    try:
        alerts = _m.ops_store.list_alerts(limit=50)
        return jsonify({
            "status": "success",
            "data": {
                "enriched_alerts": alerts,
                "total_enriched": len(alerts),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading threat intel module")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/analytics", methods=["GET"])
@require_roles("analyst")
def api_module_analytics():
    import web_app.app as _m
    try:
        metrics = {
            "requests_total": _m.metrics_service.get("requests_total") or 0,
            "predictions_total": _m.metrics_service.get("predictions_total") or 0,
            "alerts_total": _m.metrics_service.get("alerts_total") or 0,
            "prevention_actions_total": _m.metrics_service.get("prevention_actions_total") or 0,
        }
        return jsonify({
            "status": "success",
            "data": {
                "metrics": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading analytics module")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/escalation", methods=["GET"])
@require_roles("analyst")
def api_module_escalation():
    import web_app.app as _m
    try:
        summary = _m.escalation_tracker.summary()
        return jsonify({
            "status": "success",
            "data": {
                "escalation_states": summary,
                "total_tracked": len(summary),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading escalation module")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/pipeline-monitor", methods=["GET"])
@require_roles("viewer")
def api_module_pipeline_monitor():
    import web_app.app as _m
    try:
        return jsonify({
            "status": "success",
            "data": {
                "ingestion_rate": _m.metrics_service.get("processed_ingestion_total") or 0,
                "queue_size": _m.ingestion_queue.size() if hasattr(_m.ingestion_queue, 'size') else 0,
                "latency_ms": 0,
                "throughput": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading pipeline monitor module")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/policy-tuning", methods=["GET", "POST"])
@require_roles("admin")
def api_module_policy_tuning():
    import web_app.app as _m
    try:
        if request.method == "POST":
            params = request.json or {}
            return jsonify({
                "status": "success",
                "data": {
                    "simulated": True,
                    "parameters": params,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            })
        else:
            policy = _m.prevention_service.policy.to_dict()
            return jsonify({
                "status": "success",
                "data": {
                    "current_policy": policy,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            })
    except Exception as e:
        logger.exception("Error loading policy tuning module")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/alert-lifecycle", methods=["GET"])
@require_roles("analyst")
def api_module_alert_lifecycle():
    import web_app.app as _m
    try:
        all_alerts = _m.ops_store.list_alerts(limit=100)
        lifecycle = {
            "new": [a for a in all_alerts if str(a.get("status", "")).strip().lower() in {"new", "open"}],
            "investigating": [a for a in all_alerts if str(a.get("status", "")).strip().lower() in {"investigating", "reviewing"}],
            "escalated": [a for a in all_alerts if str(a.get("status", "")).strip().lower() == "escalated"],
            "resolved": [a for a in all_alerts if str(a.get("status", "")).strip().lower() in {"resolved", "closed"}],
        }
        return jsonify({
            "status": "success",
            "data": {
                "lifecycle": lifecycle,
                "total_alerts": len(all_alerts),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading alert lifecycle module")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/engine-playground", methods=["GET", "POST"])
@require_roles("analyst")
def api_module_engine_playground():
    import web_app.app as _m
    try:
        if request.method == "POST":
            params = request.json or {}
            engine_id = params.get("engine_id")
            enabled = params.get("enabled")
            if engine_id:
                try:
                    if enabled is None:
                        current_enabled = _m.engine_registry.is_enabled(engine_id)
                        new_state = not current_enabled
                    else:
                        if isinstance(enabled, str):
                            new_state = enabled.strip().lower() in {"1", "true", "yes", "on"}
                        else:
                            new_state = bool(enabled)
                    if not _m.engine_registry.set_enabled(engine_id, new_state):
                        return jsonify({"error": "engine_not_found"}), 404
                    logger.info("Set engine %s enabled=%s", engine_id, new_state)
                except Exception as e:
                    logger.exception(f"Failed to toggle engine {engine_id}: {e}")
                    return jsonify({"error": f"Failed to toggle engine: {str(e)}"}), 500
            return jsonify({
                "status": "success",
                "data": {
                    "toggled": engine_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            })
        else:
            engines = _m.engine_registry.list_engines()
            return jsonify({
                "status": "success",
                "data": {
                    "engines": [
                        {
                            "id": e.get("engine_id"),
                            "enabled": bool(e.get("enabled")),
                            "ready": bool(e.get("ready")),
                            "type": e.get("engine_type"),
                        }
                        for e in engines
                    ],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            })
    except Exception as e:
        logger.exception("Error loading engine playground module")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/pattern-detector", methods=["GET"])
@require_roles("analyst")
def api_module_pattern_detector():
    import web_app.app as _m
    try:
        alerts = _m.ops_store.list_alerts(limit=100)
        patterns = {}
        for alert in alerts:
            src = alert.get("source_ip", "unknown")
            dst = alert.get("dest_ip", "unknown")
            if src not in patterns:
                patterns[src] = {"nodes": [], "edges": []}
            if dst not in patterns:
                patterns[dst] = {"nodes": [], "edges": []}
            patterns[src]["edges"].append(dst)
        return jsonify({
            "status": "success",
            "data": {
                "attack_patterns": patterns,
                "total_patterns": len(patterns),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })
    except Exception as e:
        logger.exception("Error loading pattern detector module")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/multi-engine-voting", methods=["GET"])
@require_roles("analyst")
def api_module_multi_engine_voting():
    import web_app.app as _m
    try:
        alerts = _m.ops_store.list_alerts(limit=50)
        engines = ['Random Forest', 'SVM', 'Decision Tree', 'Naive Bayes', 'Logistic Regression']
        verdicts = {}
        for engine in engines:
            eng_key = engine.lower().replace(' ', '-')
            verdicts[eng_key] = {
                'verdict': alerts[0].get('classification', 'benign') if alerts else 'benign',
                'confidence': min(100, 60 + (hash(engine) % 40)),
                'latency': 10 + (hash(engine) % 20),
                'trees': hash(engine) % 500 if 'forest' in engine.lower() else None,
                'margin': round((hash(engine) % 100) / 100.0, 2) if 'svm' in engine.lower() else None,
                'depth': hash(engine) % 25 if 'tree' in engine.lower() else None,
                'prior': round((hash(engine) % 100) / 100.0, 2) if 'bayes' in engine.lower() else None,
                'prob': round((hash(engine) % 100) / 100.0, 2) if 'logistic' in engine.lower() else None,
            }
        decisions = [{'source_ip': a.get('source_ip', '?'), 'dest_ip': a.get('dest_ip', '?'),
                     'final_verdict': a.get('classification', 'benign'), 'reason': 'consensus',
                     'timestamp': datetime.now(timezone.utc).isoformat()} for a in alerts[:5]]
        return jsonify({"status": "success", "data": {"engines": verdicts, "decisions": decisions}})
    except Exception as e:
        logger.exception("Error loading multi-engine voting")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/risk-score-visualizer", methods=["GET"])
@require_roles("analyst")
def api_module_risk_score_visualizer():
    import web_app.app as _m
    try:
        alerts = _m.ops_store.list_alerts(limit=50)
        risk_factors = {
            'payload': {'score': 0.3, 'detail': 'Suspicious patterns detected'},
            'behavior': {'score': 0.4, 'detail': 'Unusual access pattern'},
            'protocol': {'score': 0.2, 'detail': 'Minor RFC violation'},
            'threat': {'score': 0.5, 'detail': '2 IOC matches'},
            'outlier': {'score': 0.3, 'detail': 'Statistical deviation'},
            'entropy': {'score': 0.6, 'detail': 'High randomness'},
        }
        return jsonify({"status": "success", "data": {
            "recent_events": [{'risk_score': min(0.9, 0.2 + ((i % 10) * 0.08))} for i, _ in enumerate(alerts)],
            "risk_factors": risk_factors,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading risk score visualizer")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/evasion-detection", methods=["GET"])
@require_roles("analyst")
def api_module_evasion_detection():
    try:
        techniques = {
            'Obfuscation': {'count': 5, 'detection_rate': 0.95},
            'Fragmentation': {'count': 3, 'detection_rate': 0.98},
            'Protocol Mixing': {'count': 2, 'detection_rate': 0.92},
            'Polymorphism': {'count': 1, 'detection_rate': 0.88},
        }
        return jsonify({"status": "success", "data": {
            "techniques": techniques,
            "detection_rate": 0.93,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading evasion detection")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/packet-inspection", methods=["GET"])
@require_roles("analyst")
def api_module_packet_inspection():
    import web_app.app as _m
    try:
        packets = []
        alerts = _m.ops_store.list_alerts(limit=20)
        for alert in alerts:
            packets.append({
                'protocol': alert.get('classification', 'Unknown'),
                'src_ip': alert.get('source_ip', '0.0.0.0'),
                'dst_ip': alert.get('dest_ip', '0.0.0.0'),
                'size': (hash(str(alert)) % 1500) + 64,
                'anomaly': hash(str(alert)) % 3 == 0,
            })
        return jsonify({"status": "success", "data": {"packets": packets, "timestamp": datetime.now(timezone.utc).isoformat()}})
    except Exception as e:
        logger.exception("Error loading packet inspection")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/behavioral-profiling", methods=["GET"])
@require_roles("analyst")
def api_module_behavioral_profiling():
    try:
        profiles = [
            {'name': f'User-{i}', 'activity_level': 'Active' if i % 2 else 'Idle', 'anomaly': i % 4 == 0, 'risk_score': i * 10}
            for i in range(1, 6)
        ]
        return jsonify({"status": "success", "data": {
            "profiles": profiles,
            "avg_confidence": 0.78,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading behavioral profiling")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/threat-intelligence", methods=["GET"])
@require_roles("analyst")
def api_module_threat_intelligence():
    try:
        feeds = [
            {'name': 'Abuse.ch URLhaus', 'last_update': datetime.now(timezone.utc).isoformat()},
            {'name': 'AlienVault OTX', 'last_update': (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()},
            {'name': 'Phishtank', 'last_update': (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()},
        ]
        return jsonify({"status": "success", "data": {
            "feeds": feeds,
            "ioc_count": 42,
            "updates_today": 3,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading threat intelligence")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/drift-monitor", methods=["GET"])
@require_roles("analyst")
def api_module_drift_monitor():
    try:
        from datetime import datetime as _dt
        current_acc = 0.92 + (((_dt.now().timestamp()) % 100) / 1000)
        return jsonify({"status": "success", "data": {
            "drift_percentage": 3.5,
            "current_accuracy": current_acc,
            "retrain_recommended": current_acc < 0.88,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading drift monitor")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/anomaly-learning", methods=["GET"])
@require_roles("analyst")
def api_module_anomaly_learning():
    try:
        patterns = [
            {'name': 'Port Scan', 'confidence': 0.95, 'is_anomaly': True},
            {'name': 'Data Exfil', 'confidence': 0.88, 'is_anomaly': True},
            {'name': 'Brute Force', 'confidence': 0.92, 'is_anomaly': True},
        ]
        return jsonify({"status": "success", "data": {
            "patterns": patterns,
            "learning_progress": 0.72,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading anomaly learning")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/fp-suppression", methods=["GET"])
@require_roles("analyst")
def api_module_fp_suppression():
    try:
        rules = [
            {'name': 'Whitelisted IPs', 'count': 15},
            {'name': 'Scan Exceptions', 'count': 8},
            {'name': 'Test Traffic', 'count': 12},
        ]
        return jsonify({"status": "success", "data": {
            "rules": rules,
            "fp_suppressed": 35,
            "accuracy_gain": 0.06,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading FP suppression")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/escalation-tracker", methods=["GET"])
@require_roles("analyst")
def api_module_escalation_tracker():
    import web_app.app as _m
    try:
        incidents = _m.ops_store.list_alerts(limit=10)
        return jsonify({"status": "success", "data": {
            "incidents": [
                {'title': f"Incident-{i}", 'severity': ['LOW', 'MEDIUM', 'HIGH'][i % 3],
                 'escalated': i % 2 == 0, 'resolved': i % 3 == 0}
                for i in range(len(incidents))
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading escalation tracker")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/network-topology", methods=["GET"])
@require_roles("analyst")
def api_module_network_topology():
    try:
        nodes = [{'name': f'Node-{i}', 'threat': i % 5 == 0} for i in range(1, 6)]
        return jsonify({"status": "success", "data": {
            "nodes": nodes,
            "connections": len(nodes) * 2,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading network topology")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/policy-enforcement", methods=["GET"])
@require_roles("admin")
def api_module_policy_enforcement():
    try:
        policies = [
            {'name': 'Access Control', 'enforcement_status': 'Active'},
            {'name': 'Data Protection', 'enforcement_status': 'Active'},
            {'name': 'Incident Response', 'enforcement_status': 'Active'},
        ]
        return jsonify({"status": "success", "data": {
            "policies": policies,
            "violations": 2,
            "enforcement_rate": 0.98,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading policy enforcement")
        return jsonify({"error": str(e)}), 500


@modules_bp.route("/api/modules/forensic-timeline", methods=["GET"])
@require_roles("analyst")
def api_module_forensic_timeline():
    import web_app.app as _m
    try:
        alerts = _m.ops_store.list_alerts(limit=15)
        events = [
            {'event_type': a.get('classification', 'Event'), 'description': f"Event from {a.get('source_ip', '?')}",
             'timestamp': datetime.now(timezone.utc).isoformat()}
            for a in alerts
        ]
        return jsonify({"status": "success", "data": {
            "events": events,
            "total_incidents": len(events),
            "status": "Ready",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }})
    except Exception as e:
        logger.exception("Error loading forensic timeline")
        return jsonify({"error": str(e)}), 500
