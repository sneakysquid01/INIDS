"""
Real-time event broadcaster for INIDS 2.0
Subscribes to EventBus and emits events to WebSocket clients
"""

import logging
import threading
import time
from typing import Callable, Dict, Any
from datetime import datetime

from src._telemetry import get_streamer_errors

from src.core.event_bus import DetectionEvent, ActionEvent, RiskScoreEvent, PolicyDecisionEvent, AuditEvent

logger = logging.getLogger(__name__)


class RealTimeStreamer:
    """
    Subscribes to EventBus and broadcasts events to WebSocket clients.
    Handles real-time streaming of alerts, actions, and metrics.
    """

    def __init__(self, event_bus, socketio, namespace: str = "/events"):
        """
        Initialize the RealTimeStreamer.

        Args:
            event_bus: EventBus instance to subscribe to
            socketio: Flask-SocketIO instance for broadcasting
            namespace: WebSocket namespace to broadcast to (default: "/events")
        """
        self.event_bus = event_bus
        self.socketio = socketio
        self.namespace = namespace
        self._lock = threading.Lock()
        self._running = False
        # Rate-limit error log to once per 10s per room
        self._last_error_log: dict[str, float] = {}
        self._error_log_lock = threading.Lock()

    def _record_emit_error(self, room: str, exc: Exception) -> None:
        """Increment counter and emit a rate-limited log (once per 10s per room)."""
        get_streamer_errors(room).inc()
        now = time.monotonic()
        with self._error_log_lock:
            last = self._last_error_log.get(room, 0.0)
            if now - last >= 10.0:
                self._last_error_log[room] = now
                should_log = True
            else:
                should_log = False
        if should_log:
            logger.warning("streamer.emit_error room=%s err=%s", room, exc, exc_info=True)

    def start(self) -> None:
        """Start the RealTimeStreamer."""
        if self._running:
            logger.warning("RealTimeStreamer already running")
            return

        self._running = True
        logger.info("RealTimeStreamer started - real-time event broadcasting enabled")

        # Subscribe to all important event types
        self._subscribe_to_events()

    def stop(self) -> None:
        """Stop the RealTimeStreamer."""
        self._running = False
        logger.info("RealTimeStreamer stopped")

    def _subscribe_to_events(self) -> None:
        """Subscribe to EventBus events."""
        if self.event_bus is None:
            logger.warning("EventBus not available for real-time events")
            return

        try:
            # Subscribe to detection events
            self.event_bus.subscribe(DetectionEvent, self._on_detection_event)
            logger.debug("Subscribed to DetectionEvent")
            
            # Subscribe to risk score events
            self.event_bus.subscribe(RiskScoreEvent, self._on_risk_event)
            logger.debug("Subscribed to RiskScoreEvent")
            
            # Subscribe to policy decision events
            self.event_bus.subscribe(PolicyDecisionEvent, self._on_policy_decision)
            logger.debug("Subscribed to PolicyDecisionEvent")
            
            # Subscribe to action events
            self.event_bus.subscribe(ActionEvent, self._on_action_event)
            logger.debug("Subscribed to ActionEvent")
            
            # Subscribe to audit events
            self.event_bus.subscribe(AuditEvent, self._on_audit_event)
            logger.debug("Subscribed to AuditEvent")
        except Exception as e:
            logger.error(f"Failed to subscribe to events: {e}", exc_info=True)

    def _on_detection_event(self, event: DetectionEvent) -> None:
        """Handle detection event."""
        try:
            if not self._running:
                return

            payload = {
                "type": "detection.new",
                "timestamp": datetime.now().isoformat(),
                "data": event.to_dict(),
            }
            self.socketio.emit("alert.new", payload, namespace=self.namespace)
            logger.debug(f"Emitted detection event to WebSocket clients")
        except Exception as e:
            self._record_emit_error("detection", e)

    def _on_risk_event(self, event: RiskScoreEvent) -> None:
        """Handle risk score event."""
        try:
            if not self._running:
                return

            payload = {
                "type": "risk.update",
                "timestamp": datetime.now().isoformat(),
                "data": event.to_dict(),
            }
            self.socketio.emit("risk.update", payload, namespace=self.namespace)
            logger.debug(f"Emitted risk score event to WebSocket clients")
        except Exception as e:
            self._record_emit_error("risk", e)

    def _on_policy_decision(self, event: PolicyDecisionEvent) -> None:
        """Handle policy decision event."""
        try:
            if not self._running:
                return

            payload = {
                "type": "decision.made",
                "timestamp": datetime.now().isoformat(),
                "data": event.to_dict(),
            }
            self.socketio.emit("decision.made", payload, namespace=self.namespace)
            logger.debug(f"Emitted policy decision event to WebSocket clients")
        except Exception as e:
            self._record_emit_error("decision", e)

    def _on_action_event(self, event: ActionEvent) -> None:
        """Handle action event."""
        try:
            if not self._running:
                return

            # Distinguish between pending and executed actions
            event_name = "action.executed" if event.executed else "action.pending"
            
            payload = {
                "type": event_name,
                "timestamp": datetime.now().isoformat(),
                "data": event.to_dict(),
            }
            self.socketio.emit(event_name, payload, namespace=self.namespace)
            logger.debug(f"Emitted action event to WebSocket clients: {event_name}")
        except Exception as e:
            self._record_emit_error("actions", e)

    def _on_audit_event(self, event: AuditEvent) -> None:
        """Handle audit event."""
        try:
            if not self._running:
                return

            payload = {
                "type": "audit",
                "timestamp": datetime.now().isoformat(),
                "data": event.to_dict(),
            }
            self.socketio.emit("audit", payload, namespace=self.namespace)
            logger.debug(f"Emitted audit event to WebSocket clients")
        except Exception as e:
            self._record_emit_error("audit", e)

    def emit_alert(self, alert_data: Dict[str, Any]) -> None:
        """
        Manually emit an alert event.

        Args:
            alert_data: Dictionary containing alert information
        """
        try:
            if not self._running:
                return

            payload = {
                "type": "alert.new",
                "timestamp": datetime.now().isoformat(),
                "data": alert_data,
            }
            self.socketio.emit("alert.new", payload, namespace=self.namespace)
        except Exception as e:
            self._record_emit_error("alerts", e)

    def emit_action_pending(self, action_data: Dict[str, Any]) -> None:
        """
        Manually emit a pending action event.

        Args:
            action_data: Dictionary containing action information
        """
        try:
            if not self._running:
                return

            payload = {
                "type": "action.pending",
                "timestamp": datetime.now().isoformat(),
                "data": action_data,
            }
            self.socketio.emit("action.pending", payload, namespace=self.namespace)
        except Exception as e:
            self._record_emit_error("actions", e)

    def emit_metrics(self, metrics_data: Dict[str, Any]) -> None:
        """
        Manually emit a metrics update event.

        Args:
            metrics_data: Dictionary containing metrics
        """
        try:
            if not self._running:
                return

            payload = {
                "type": "metrics.update",
                "timestamp": datetime.now().isoformat(),
                "data": metrics_data,
            }
            self.socketio.emit("metrics.update", payload, namespace=self.namespace)
        except Exception as e:
            self._record_emit_error("metrics", e)
