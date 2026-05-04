# INIDS 3.0 Component Library Documentation

**Status:** Phase 2 Complete - All 13 Components Implemented  
**Last Updated:** May 4, 2026  
**Version:** 1.0

---

## Overview

The INIDS 3.0 component library provides a comprehensive, reusable system of UI and data-driven components built on ES6 modules and Tailwind CSS. All components follow a consistent design system and integrate with the global state management system.

---

## UI Components (C1-C6)

### [C1] AppModal — Modal Dialog System
**Location:** `/static/js/components/app_modal.js`

#### Purpose
Reusable modal dialog for alerts, forms, confirmations, and custom content injection.

#### Key Features
- Open/close animations
- Escape key and backdrop click support
- Configurable title, content, size (sm, md, lg, xl)
- Custom buttons with variants (primary, secondary, danger)
- Static helpers: `confirm()`, `alert()`

#### Usage
```javascript
import { AppModal } from "../components/app_modal.js";

// Basic modal
const modal = new AppModal({
    title: "Confirm Action",
    content: "<p>Are you sure?</p>",
    size: "md"
});
modal.open();

// Static helper
AppModal.confirm(
    "Delete Item",
    "This cannot be undone.",
    () => console.log("Confirmed"),
    () => console.log("Cancelled")
);
```

#### Integration
- Mounts to `#app-modal-root` container (in base.html)
- Z-index: 50 (modal), 40 (backdrop)
- Dark theme colors: #151922 (panel), #0f1117 (header)

---

### [C2] AppToast — Toast Notification System
**Location:** `/static/js/components/app_toast.js`

#### Purpose
Temporary, stacked notifications for success, error, warning, info messages with auto-dismiss.

#### Key Features
- 4 types: success (green), error (red), warning (amber), info (blue)
- Auto-dismiss after configurable duration (default 3-4 seconds)
- Stacking support (multiple toasts visible)
- Manual dismiss button
- Static factory methods: `show()`, `success()`, `error()`, `warning()`, `info()`

#### Usage
```javascript
import { AppToast } from "../components/app_toast.js";

// Typed methods
AppToast.success("Operation completed!");
AppToast.error("Something went wrong");
AppToast.warning("Action not reversible");
AppToast.info("FYI: Check your settings");

// Custom options
AppToast.show("Custom message", {
    type: "info",
    duration: 5000,
    dismissible: true
});

// Loading toast (no auto-dismiss)
const toastId = AppToast.loading("Processing...");
// Later...
AppToast.update(toastId, "Completed!", { type: "success" });
AppToast.dismiss(toastId);
```

#### Integration
- Mounts to `#app-toast-root` container (bottom-right, z-50)
- Auto-initializes on DOM ready
- Integrates with all data components for feedback

---

### [C3] UIButton — Reusable Button Component
**Location:** `/static/js/components/ui_button.js`

#### Purpose
Factory function creating styled buttons with variants, sizes, and states.

#### Key Features
- 7 variants: primary, secondary, danger, success, warning, outline, ghost
- 3 sizes: sm, md, lg
- Icon support
- Loading state with spinner
- Disabled state

#### Usage
```javascript
import { UIButton } from "../components/ui_button.js";

// Factory method
const btn = UIButton.create("Click Me", {
    variant: "primary",
    size: "lg",
    icon: "check",
    onClick: handleClick
});

// Convenience methods
UIButton.primary("Save", onSave);
UIButton.danger("Delete", onDelete, { size: "sm" });
UIButton.outline("Cancel", onCancel);

// Icon-only button
const iconBtn = UIButton.icon("bell", onNotify, {
    size: "md",
    variant: "ghost"
});

// Button groups
const group = UIButton.group([btn1, btn2], {
    direction: "horizontal",
    align: "right"
});

// Loading state
UIButton.setLoading(btn, true);
// ... later
UIButton.setLoading(btn, false);
```

#### Integration
- Uses Tailwind classes for styling
- Color variants map to design system (threat, warn, safe, info)
- Used in modals, cards, data components

---

### [C4] UIBadge — Status/Severity Badges
**Location:** `/static/js/components/ui_badge.js`

#### Purpose
Small labels for status indicators, severity levels, tags, and enums.

#### Key Features
- 5 variants: threat (red), warn (amber), safe (green), info (blue), neutral (gray)
- Specialized badges: severity, status, count, tag, threatLevel
- Icon support
- Rounded or rectangular

#### Usage
```javascript
import { UIBadge } from "../components/ui_badge.js";

// Severity badges
UIBadge.severity("critical"); // Red with icon
UIBadge.severity("high");
UIBadge.severity("medium");
UIBadge.severity("low");

// Status badges
UIBadge.status("active");     // Green
UIBadge.status("blocked");    // Red
UIBadge.status("pending");    // Amber
UIBadge.status("failed");     // Red

// Count badge (pill)
UIBadge.count(42, { variant: "info", max: 99 });

// Tag badge (with remove)
UIBadge.tag("Important", {
    variant: "warn",
    removable: true,
    onRemove: () => console.log("removed")
});

// Custom enum mapping
UIBadge.enum("high", {
    high: { label: "High", variant: "threat" },
    medium: { label: "Medium", variant: "warn" },
    low: { label: "Low", variant: "safe" }
});
```

#### Integration
- Used in AlertCard, ActionCard, EngineCard
- Inline elements, no height impact
- Always shows icon + label or just label

---

### [C5] UICard — Container Component
**Location:** `/static/js/components/ui_card.js`

#### Purpose
Base container component for building panels, data displays, and section blocks.

#### Key Features
- Header with title, icon, actions
- Content area (flexible)
- Footer area
- Collapsible support
- Clickable variant
- Specialized: stat card, list items, grid

#### Usage
```javascript
import { UICard } from "../components/ui_card.js";

// Basic card
const card = UICard.create({
    title: "Card Title",
    icon: "info-circle",
    content: "Card content here",
    footer: "Footer text"
});

// Stat card (for metrics)
UICard.stat("CPU Usage", 65, {
    unit: "%",
    trend: 5,
    icon: "cpu"
});

// Panel (simplified)
UICard.panel("Section Title", "Content goes here");

// List item card (inside card grid)
UICard.listItem({
    primary: "Item Name",
    secondary: "Subtitle",
    icon: "file",
    badge: UIBadge.status("active"),
    onClick: handleClick
});

// Card grid
UICard.grid([card1, card2, card3], 3);

// Update content dynamically
UICard.updateContent(card, "New content");

// Loading state
UICard.setLoading(card);

// Error state
UICard.setError(card, "Failed to load data");
```

#### Integration
- Used as base for all data components
- Handles layout, styling, animation
- Colors: #151922 (panel), #0f1117 (header)
- Border: 1px #1a1f2e

---

### [C6] LoadingSpinner — Loading State Indicators
**Location:** `/static/js/components/loading_spinner.js`

#### Purpose
Various spinner styles for different use cases (loading, progress, skeleton, shimmer).

#### Key Features
- 4 spinner styles: spin, pulse, bounce, wave
- 3 sizes: sm, md, lg
- Progress bar with percentage
- Skeleton placeholder
- Shimmer effect
- Fullscreen overlay
- Inline spinner for buttons

#### Usage
```javascript
import { LoadingSpinner } from "../components/loading_spinner.js";

// Basic spinner
container.appendChild(LoadingSpinner.create());

// With text
container.appendChild(LoadingSpinner.withText("Loading data..."));

// Fullscreen loader
const loader = LoadingSpinner.fullscreen("Processing request...");
// Later...
loader.close();

// Skeleton placeholder
container.appendChild(LoadingSpinner.skeleton(5)); // 5 lines

// Progress bar
container.appendChild(LoadingSpinner.progress(75, 100));

// Activity dots
container.appendChild(LoadingSpinner.dots({ count: 3, size: "md" }));

// Inline spinner (for buttons)
const spinner = LoadingSpinner.inline();
btn.appendChild(spinner);
```

#### Integration
- Uses Tailwind animations
- Colors: blue-400, blue-500, blue-600
- Used in ModuleCard, data fetching, form submission

---

## Data Components (D1-D6)

### [D1] AlertCard — Security Alert Display
**Location:** `/static/js/components/alert_card.js`

#### Purpose
Displays individual security alerts with severity, confidence, and action buttons.

#### Key Features
- **FIXES INT-002: Broken Block Action** — POST /api/actions with proper payload
- Severity badge (critical, high, medium, low)
- Source/destination IP flow display
- Confidence/score bar
- Detection method
- Block IP action button
- Dismiss button

#### Usage
```javascript
import { AlertCard } from "../components/alert_card.js";

const alert = {
    id: "alert-123",
    alert_id: "alert-123",
    title: "Potential DDoS Attack",
    severity: "high",
    src_ip: "192.168.1.100",
    dst_ip: "10.0.0.1",
    alert_type: "DDoS",
    confidence: 85,
    score: 85,
    detection_method: "ML-Engine-v2",
    context: "Abnormal traffic pattern detected",
    timestamp: new Date().toISOString()
};

const card = AlertCard(alert);
container.appendChild(card);
```

#### Block Action Flow (Fixes INT-002)
```javascript
// User clicks "Block IP" button
// → AlertCard calls HttpClient.post("/api/actions", {
//     alert_id: "alert-123",
//     type: "block",
//     target_ip: "192.168.1.100"
//   })
// → Response: { id, status: "executed", ... }
// → GlobalState.push("actions", newAction)
// → AppToast.success("Blocked IP: 192.168.1.100")
// → Button disabled, card shows success
```

#### Integration
- Imports: UICard, UIBadge, UIButton, AppToast, HttpClient, GlobalState
- Used in alerts.page.js, monitor.page.js
- Fully error-handled with user feedback

---

### [D2] MetricCard — System Metrics Display
**Location:** `/static/js/components/metric_card.js`

#### Purpose
Displays metrics with progress bars, trends, sparklines, and threshold alerts.

#### Key Features
- Large value display
- Progress bar with percentage
- Trend indicator (↑ up, ↓ down, → stable)
- Sparkline graph (mini bar chart)
- Threshold alerts
- Status colors: normal, warning, critical

#### Usage
```javascript
import { MetricCard } from "../components/metric_card.js";

// Basic metric
const card = MetricCard({
    label: "Alerts per Minute",
    value: 42,
    unit: "/min",
    max: 100,
    threshold: 80,
    trend: 5, // +5% vs last period
    status: "normal"
});

// With sparkline
const cardWithSparkline = MetricCard({
    label: "CPU Usage",
    value: 65,
    unit: "%",
    max: 100,
    sparkline: [30, 45, 50, 65, 60, 55, 60, 65],
    trend: -2,
    status: "warning" // Above 60%
});

// Helper to create from raw data
const metricData = createMetricData(rawData, {
    label: "Response Time",
    value_key: "avg_response_ms",
    max_key: "max_response_ms",
    unit: "ms",
    threshold_key: "threshold_ms",
    trend_key: "trend_percent",
    sparkline_key: "history",
    status_key: "status"
});
```

#### Integration
- Used in dashboard.page.js, monitor.page.js
- Subscribes to GlobalState.metrics
- Auto-updates on metric changes
- Responsive grid layout

---

### [D3] ModuleCard — Capability Module Display
**Location:** `/static/js/components/module_card.js`

#### Purpose
Displays system capability modules with status, data, and error boundary.

#### Key Features
- **FIXES STRUCT-001: Module Loading Error Boundary**
- Module status badge (active, loading, failed)
- Dynamic data fetching from endpoint or GlobalState
- Refresh button with error recovery
- Auto-refresh with configurable interval
- Error state display with graceful degradation

#### Usage
```javascript
import { ModuleCard } from "../components/module_card.js";

// Module with API endpoint
const card = ModuleCard("real-time-detection", {
    title: "Real-Time Detection",
    description: "Live threat detection from core pipeline",
    endpoint: "/api/modules/real-time-detection",
    refreshInterval: 5000
});

// Module with GlobalState slice
const card2 = ModuleCard("metrics", {
    title: "Metrics Collector",
    description: "System performance metrics",
    refreshInterval: 3000
});

// Helper to create module grid
const grid = createModuleGrid({
    "real-time-detection": { title: "Detection", ... },
    "machine-learning": { title: "ML Engine", ... },
    // ... more modules
}, 3); // 3 columns
```

#### Error Boundary Implementation
```javascript
// If module fetch fails:
// → Catches error
// → Shows error badge "Failed"
// → Displays error message with red border
// → User can click "Refresh" button to retry
// → No crash, other modules continue loading
```

#### Integration
- Used in dashboard.page.js
- Implements error boundaries (STRUCT-001 fix)
- Auto-loads 15 system modules
- Each module independently managed

---

### [D4] EngineCard — Detection Engine Status
**Location:** `/static/js/components/engine_card.js`

#### Purpose
Displays individual detection engine status, load, accuracy, and model info.

#### Key Features
- Engine status badge
- Load percentage bar with color coding
- Accuracy percentage
- Model and version info
- Last update timestamp
- Detection count

#### Usage
```javascript
import { EngineCard } from "../components/engine_card.js";

const engine = {
    name: "DL-Engine-v2",
    id: "dl-engine-2",
    status: "active", // active, inactive, failed
    load: 45,
    accuracy: 97.2,
    model: "YOLOv8-custom",
    version: "2.3.1",
    lastUpdate: new Date().toISOString(),
    detections: 1240
};

const card = EngineCard(engine);
container.appendChild(card);

// Helper to create grid
const engineGrid = createEngineGrid(
    [engine1, engine2, engine3],
    3 // columns
);
```

#### Integration
- Used in engines.page.js
- Shows all detection engines
- Real-time status updates
- Responsive layout

---

### [D5] ActionCard — Security Action History
**Location:** `/static/js/components/action_card.js`

#### Purpose
Displays individual security action records (blocks, rate limits, investigations).

#### Key Features
- Action type badge (block, rate_limit, temp_block, alert, investigate)
- Target IP/entity
- Status badge (pending, executed, failed, rolled_back)
- Reason for action
- Duration (if applicable)
- Result data
- Executor info

#### Usage
```javascript
import { ActionCard } from "../components/action_card.js";

const action = {
    id: "action-456",
    type: "block", // block | rate_limit | temp_block | alert | investigate
    target: "192.168.1.50",
    status: "executed", // pending | executed | failed | rolled_back
    reason: "Suspicious traffic pattern detected",
    timestamp: new Date().toISOString(),
    duration: 3600, // seconds
    executor: "System",
    result: { firewall_rule_id: "rule-123", blocked_bytes: 45000000 }
};

const card = ActionCard(action);

// Helper to create timeline
const timeline = createActionTimeline([action1, action2, action3]);
```

#### Integration
- Used in actions.page.js, respond.page.js
- Shows action history
- Reversible actions displayed
- Filterable by type/status

---

### [D6] PolicyHistoryItem — Policy Change Audit Trail
**Location:** `/static/js/components/policy_history_item.js`

#### Purpose
Displays individual policy change with before/after comparison and audit info.

#### Key Features
- Action badge (created, modified, deleted, reverted)
- User and timestamp
- Status badge (applied, pending, reverted)
- Policy name and field
- Before/after value comparison
- Change reason
- Full policy comparison view

#### Usage
```javascript
import { PolicyHistoryItem, createPolicyHistoryTimeline, createPolicyComparison } from "../components/policy_history_item.js";

const change = {
    id: "history-789",
    timestamp: new Date().toISOString(),
    user: "admin@example.com",
    action: "modified", // created | modified | deleted | reverted
    policyName: "Rate Limit Policy",
    field: "max_requests_per_minute",
    oldValue: 100,
    newValue: 150,
    reason: "Increased to accommodate peak traffic",
    status: "applied" // applied | pending | reverted
};

// Single item
const item = PolicyHistoryItem(change);

// Timeline with pagination
const timeline = createPolicyHistoryTimeline(changes, 10); // Show 10, then "Load more"

// Policy comparison
const comparison = createPolicyComparison(oldPolicy, newPolicy);
```

#### Integration
- Used in policy.page.js
- Shows full audit trail
- Before/after visual comparison
- Revert capability linked to this history

---

## Design System Integration

### Color Palette
All components use design system colors:
- **Threat (Critical):** #ef4444 (red)
- **Warn (High):** #f59e0b (amber)
- **Safe (Low):** #10b981 (green)
- **Info:** #3b82f6 (blue)
- **Surface Dark:** #090c12, #0f1117, #151922, #1a1f2e
- **Text:** white, #e5e7eb (gray-200), #9ca3af (gray-400)

### Typography
- **Display:** Syne, 800px+ bold
- **Body:** JetBrains Mono, 12-14px
- **Code:** JetBrains Mono, monospace

### Spacing
- Base: 4px grid
- Units: 8, 12, 16, 20, 24, 28px
- Gap between items: 8px (tight), 12px (normal), 16px (loose)

### Border Radius
- Cards: 8px
- Panels: 12px
- Buttons: 8px
- Badges: 99px (rounded-full)

### Shadows
- Default: 0 4px 24px rgba(0,0,0,0.4)
- Hover elevation with shadow-lg/shadow-xl

---

## Global State Integration

All data components automatically subscribe to GlobalState slices:

```javascript
// GlobalState slices available to all components
GlobalState.subscribe("metrics", callback);
GlobalState.subscribe("alerts", callback);
GlobalState.subscribe("modules", callback);
GlobalState.subscribe("actions", callback);
GlobalState.subscribe("policy", callback);
GlobalState.subscribe("engines", callback);
GlobalState.subscribe("health", callback);
GlobalState.subscribe("allowlist", callback);
GlobalState.subscribe("models", callback);
GlobalState.subscribe("investigations", callback);
GlobalState.subscribe("honeypot", callback);
```

---

## Error Handling & Resilience

### Component Error Boundaries
- **ModuleCard:** Catches fetch errors, displays error state, allows retry
- **AlertCard:** Validates alert data, falls back to defaults
- **MetricCard:** Handles missing data gracefully
- **ActionCard:** Validates action types, shows unknown for unmapped values

### User Feedback
- **Success:** AppToast.success()
- **Error:** AppToast.error() with message
- **Info:** AppToast.info() with details
- **Loading:** LoadingSpinner or AppToast.loading()

### Validation
- All component inputs validated before rendering
- Fallback defaults provided for missing data
- Type coercion for numbers/strings/dates

---

## Performance Considerations

### Component Lifecycle
- Minimal DOM manipulation (batch updates)
- Event listener cleanup on removal
- Subscription unsubscribe on destroy
- No memory leaks from closures

### Rendering
- Single-pass DOM construction
- Conditional rendering via `display: none`
- Animation via CSS transitions (not JavaScript)
- Lazy loading support via IntersectionObserver (future)

### Bundle Size
- Individual imports: ~150KB total (all 13 components)
- Gzip: ~45KB
- Tree-shaking friendly (ES6 exports)

---

## Testing Checklist

- [ ] AppModal opens/closes with animation
- [ ] AppToast stacks and auto-dismisses
- [ ] UIButton responds to click, loading, disabled states
- [ ] UIBadge displays correct colors/icons
- [ ] UICard updates content dynamically
- [ ] LoadingSpinner plays animations smoothly
- [ ] AlertCard block action sends correct API payload
- [ ] MetricCard progress bar accurate
- [ ] ModuleCard handles errors gracefully
- [ ] EngineCard displays load bar colors correctly
- [ ] ActionCard comparison shows before/after
- [ ] PolicyHistoryItem renders timeline correctly
- [ ] All components responsive on mobile/tablet/desktop

---

## Next Steps (Phase 3)

Components are ready for page integration:
1. Mount components to page templates
2. Connect to GlobalState subscriptions
3. Wire up API endpoints
4. Test end-to-end data flows
5. Performance optimization
