let detections = 0;
let alerts = 0;
let blocks = 0;

const timeline = document.getElementById("timeline");
const riskFeed = document.getElementById("riskFeed");

function prepend(container, content) {
  if (container.querySelector(".empty-state")) {
    container.innerHTML = "";
  }
  const el = document.createElement("div");
  el.className = "feed-item";
  el.innerHTML = content;
  container.prepend(el);
  while (container.children.length > 100) {
    container.removeChild(container.lastChild);
  }
}

function refreshMetrics(latestRisk) {
  document.getElementById("mDetections").textContent = detections;
  document.getElementById("mAlerts").textContent = alerts;
  document.getElementById("mBlocks").textContent = blocks;
  if (typeof latestRisk === "number") {
    document.getElementById("mRisk").textContent = latestRisk.toFixed(3);
  }
}

const socket = io("/events", { transports: ["websocket", "polling"] });

socket.on("connect", () => {
  prepend(timeline, "<span class='badge-soft'>system</span> websocket connected");
});

socket.on("disconnect", () => {
  prepend(timeline, "<span class='badge-soft'>system</span> websocket disconnected");
});

socket.on("DetectionEvent", evt => {
  detections += 1;
  if (String(evt.prediction).toLowerCase() === "attack" || evt.suspicious) {
    alerts += 1;
  }
  prepend(
    timeline,
    "<span class='badge-soft'>detection</span> " +
    evt.timestamp +
    " src=" + (evt.source_ip || "unknown") +
    " pred=" + evt.prediction +
    " conf=" + evt.confidence
  );
  refreshMetrics();
});

socket.on("RiskScoreEvent", evt => {
  const risk = Number(evt.risk_score || 0);
  prepend(
    riskFeed,
    "<span class='badge-soft'>risk</span> " +
    evt.timestamp +
    " src=" + (evt.detection && evt.detection.source_ip ? evt.detection.source_ip : "unknown") +
    " score=" + risk.toFixed(3)
  );
  refreshMetrics(risk);
});

socket.on("ActionEvent", evt => {
  if (String(evt.action || "").toLowerCase().includes("block")) {
    blocks += 1;
  }
  prepend(
    timeline,
    "<span class='badge-soft'>action</span> " +
    evt.created_at +
    " action=" + evt.action +
    " target=" + evt.target +
    " status=" + evt.status
  );
  refreshMetrics();
});
