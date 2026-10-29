"use strict";
const $ = (id) => document.getElementById(id);
const labels = {
  live: "Live monitoring",
  setup: "Setup",
  assistant: "Assistant",
  history: "History",
  settings: "Settings",
  report: "Session report",
};
let health = {},
  lastMessage = 0,
  lastFrame = null,
  trend = [],
  currentView = "live",
  pendingDelete = null,
  deleteTrigger = null;
let settingsFields = [];
const titleCase = (value) =>
  value ? value.charAt(0).toUpperCase() + value.slice(1) : "Idle";
function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
}
function notice(message) {
  $("notice").textContent = message;
  $("notice").hidden = !message;
}
async function api(path, options = {}) {
  const response = await fetch("/api/v1" + path, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  if (!response.ok) {
    let body;
    try {
      body = await response.json();
    } catch {
      body = { detail: "The request failed. Please try again." };
    }
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : "Check the supplied values.",
    );
  }
  return response.json();
}
async function busy(button, action) {
  if (button.disabled) return;
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  try {
    await action();
  } catch (error) {
    notice(error.message);
  } finally {
    button.disabled = false;
    button.removeAttribute("aria-busy");
    syncButtons();
  }
}
function route(name, focus = true) {
  notice("");
  if (!labels[name]) name = "live";
  currentView = name;
  for (const el of document.querySelectorAll(".view"))
    el.hidden = el.id !== "view-" + name;
  for (const el of document.querySelectorAll("nav button")) {
    if (el.dataset.view === name) el.setAttribute("aria-current", "page");
    else el.removeAttribute("aria-current");
  }
  $("breadcrumb").textContent = labels[name];
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  if (focus)
    $("view-" + name)
      .querySelector("h1")
      .focus();
  if (name === "history") loadHistory().catch((e) => notice(e.message));
  if (name === "setup") loadSetup().catch((e) => notice(e.message));
  if (name === "settings") loadSettings().catch((e) => notice(e.message));
  if (name === "assistant")
    loadAssistantStatus().catch((e) => notice(e.message));
}
for (const button of document.querySelectorAll("[data-view]"))
  button.addEventListener("click", () => route(button.dataset.view));
window.addEventListener("hashchange", () => route(location.hash.slice(1)));
function syncButtons() {
  const active = !!health.session_id;
  $("start-live").disabled = active;
  $("start-replay").disabled = active;
  $("stop-session").disabled = !active;
}
async function start(mode) {
  await api("/sessions", { method: "POST", body: JSON.stringify({ mode }) });
  trend = [];
  lastFrame = null;
  notice(
    mode === "replay"
      ? "Synthetic replay started. These values do not describe a real driver."
      : "Webcam session started. Check tracking quality before calibration.",
  );
  await refreshHealth();
}
$("start-live").onclick = () => busy($("start-live"), () => start("live"));
$("start-replay").onclick = () =>
  busy($("start-replay"), () => start("replay"));
$("stop-session").onclick = () =>
  busy($("stop-session"), async () => {
    await api("/sessions/stop", { method: "POST" });
    await refreshHealth();
    notice("Session saved locally. Open History to review it.");
  });
$("ack-alert").onclick = () =>
  busy($("ack-alert"), async () => {
    await api("/alerts/acknowledge", { method: "POST" });
    notice(
      "Announcement silenced. The monitoring condition remains active until it recovers.",
    );
  });
function measurement(value, digits = 2, suffix = "") {
  return value === null || value === undefined
    ? "—"
    : Number(value).toFixed(digits) + suffix;
}
function render(data) {
  health = data;
  syncButtons();
  const s = data.snapshot,
    m = s?.measurements || {};
  $("state").textContent = titleCase(s?.state);
  $("mode-badge").textContent = s?.synthetic
    ? "Synthetic replay"
    : data.session_id
      ? "Live webcam · local"
      : "No active session";
  $("metric-ear").textContent = measurement(m.ear);
  $("metric-closure").textContent = measurement(m.eye_closed_s, 1, " s");
  $("metric-perclos").textContent = measurement(
    m.perclos === null || m.perclos === undefined ? null : m.perclos * 100,
    0,
    "%",
  );
  $("metric-yaw").textContent = measurement(m.yaw, 0, "°");
  $("metric-mar").textContent = measurement(m.mar);
  $("metric-latency").textContent = measurement(
    data.metrics?.p95_capture_to_state_ms,
    0,
    " ms",
  );
  $("score").textContent = measurement(s?.score, 0);
  $("score-level").textContent =
    s?.score !== null && s?.score !== undefined
      ? titleCase(s.level) + " · heuristic indicator"
      : "No valid measurement";
  $("score-meter").style.width = (s?.score || 0) + "%";
  $("score-meter").style.background =
    s?.level === "critical"
      ? "var(--danger)"
      : s?.level === "warning"
        ? "var(--warning)"
        : "var(--accent)";
  $("state-reason").textContent = s?.reasons?.length
    ? s.reasons.map((r) => r.replaceAll("_", " ")).join(" · ")
    : data.session_id
      ? "Measurements are local. Watch tracking quality as conditions change."
      : "Your monitoring state will appear after a session starts.";
  $("tracking-label").textContent =
    s?.state === "monitoring"
      ? "Valid face tracking"
      : s?.state === "unavailable"
        ? "Tracking unavailable"
        : titleCase(s?.state);
  $("frame-counter").textContent = (data.metrics?.frames || 0) + " frames";
  $("preview").hidden =
    !data.session_id || !!s?.synthetic || !$("preview").getAttribute("src");
  $("camera-empty").hidden = !$("preview").hidden;
  $("camera-title").textContent = s?.synthetic
    ? "Synthetic monitoring replay"
    : data.session_id
      ? "Looking for your camera…"
      : "A clear view starts here.";
  $("camera-description").textContent = s?.synthetic
    ? "30 seconds: eye closure, mouth opening, head turn, and tracking loss."
    : data.session_id
      ? "Check camera permissions and keep your face in view."
      : "Start your webcam, or explore a synthetic replay.";
  $("camera-tag").textContent = s?.synthetic
    ? "ARTIFICIAL DATA · NO CAMERA IN USE"
    : "LOCAL VIDEO · NEVER UPLOADED";
  if (!data.session_id) {
    $("preview").removeAttribute("src");
    $("preview").hidden = true;
    $("camera-empty").hidden = false;
  }
  const alert = data.alert;
  $("alert-banner").hidden = !alert || alert.transition === "resolved";
  if (alert) {
    $("alert-title").textContent = titleCase(alert.severity) + " fatigue alert";
    $("alert-message").textContent = alert.acknowledged
      ? "Acknowledged. Condition remains active."
      : alert.message;
  }
  const calibration = s?.calibration || {};
  $("calibration-progress").value = calibration.valid_seconds || 0;
  $("calibration-status").textContent = calibration.active
    ? `${calibration.valid_seconds} / 20 valid seconds collected.`
    : calibration.reason || "No calibration in progress.";
  if (s && s.frame_id !== lastFrame) {
    lastFrame = s.frame_id;
    trend.push(s.score);
    if (trend.length > 120) trend.shift();
    drawTrend();
  }
  const errors = Object.values(data.errors || {});
  if (errors.length) notice(errors.join(" · "));
}
function drawTrend() {
  // Separate polyline segments prevent interpolating across unavailable measurements.
  const svg = $("trend");
  for (const el of svg.querySelectorAll(".trend-segment")) el.remove();
  $("trend-line").setAttribute("points", "");
  let points = [];
  function flush() {
    if (points.length) {
      const line = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "polyline",
      );
      line.setAttribute("points", points.join(" "));
      line.setAttribute("class", "trend-segment");
      line.setAttribute("stroke", "#71d8cf");
      line.setAttribute("stroke-width", "2");
      line.setAttribute("fill", "none");
      svg.append(line);
      points = [];
    }
  }
  trend.forEach((v, i) => {
    if (v === null) {
      flush();
      return;
    }
    points.push(`${(i * 700) / Math.max(1, trend.length - 1)},${110 - v}`);
  });
  flush();
}
async function refreshHealth() {
  render(await api("/health"));
}
function connect(path, onData, connection = false) {
  let retry = 500;
  function open() {
    const socket = new WebSocket(
      `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${path}`,
    );
    socket.onopen = () => {
      retry = 500;
      if (connection) {
        $("connection").textContent = "Connected locally";
        $("connection").classList.remove("offline");
      }
    };
    socket.onmessage = (event) => {
      try {
        onData(JSON.parse(event.data));
        if (connection) lastMessage = Date.now();
      } catch {
        notice("A monitoring update could not be displayed.");
      }
    };
    socket.onclose = () => {
      if (connection) {
        $("connection").textContent = "Disconnected · reconnecting";
        $("connection").classList.add("offline");
        markStale();
      }
      setTimeout(open, retry);
      retry = Math.min(10000, retry * 2);
    };
    socket.onerror = () => socket.close();
  }
  open();
}
function markStale() {
  if (health.session_id) {
    $("state").textContent = "Unavailable";
    $("score").textContent = "—";
    for (const id of ["ear", "closure", "perclos", "yaw", "mar"])
      $("metric-" + id).textContent = "—";
    $("tracking-label").textContent = "Connection lost · readings unavailable";
    $("preview").hidden = true;
    $("camera-empty").hidden = false;
  }
}
connect("/ws/v1/telemetry", render, true);
connect("/ws/v1/preview", (message) => {
  if (health.session_id && !health.snapshot?.synthetic) {
    $("preview").src = "data:image/jpeg;base64," + message.image;
    $("preview").hidden = false;
    $("camera-empty").hidden = true;
  }
});
setInterval(() => {
  if (lastMessage && Date.now() - lastMessage > 3000) markStale();
}, 1000);
async function loadSetup() {
  const data = await api("/setup");
  const list = $("setup-checks");
  list.replaceChildren();
  for (const [key, value] of Object.entries(data)) {
    const row = node("div", undefined, "check-row");
    row.append(node("strong", titleCase(key)));
    const detail = node("div", titleCase(value.status));
    if (value.action) detail.append(node("small", value.action));
    row.append(detail);
    list.append(row);
  }
}
$("refresh-setup").onclick = () => busy($("refresh-setup"), loadSetup);
$("audio-test").onclick = () =>
  busy($("audio-test"), async () => {
    await api("/audio/test", { method: "POST" });
    notice("Local speech queued. Confirm that you can hear the test.");
  });
$("calibrate").onclick = () =>
  busy($("calibrate"), async () => {
    await api("/calibration", { method: "POST" });
    notice(
      "Calibration started. Keep your face frontal and your eyes naturally open.",
    );
  });
async function loadAssistantStatus() {
  const status = await api("/assistant/status");
  $("assistant-status").textContent =
    `${status.configured ? "Key loaded" : "Local guidance available; key not loaded"} · ${status.model} · ${status.requests_today}/${status.daily_cap} attempts today · ${status.knowledge}`;
}
$("reload-assistant").onclick = () =>
  busy($("reload-assistant"), async () => {
    await api("/assistant/reload", { method: "POST" });
    await loadAssistantStatus();
    notice(
      "Assistant settings reloaded. Your monitoring session is unchanged.",
    );
  });
function answer(response) {
  $("assistant-answer").textContent = response.answer;
  $("answer-source").textContent = (response.source || "local").toUpperCase();
  $("assistant-limitation").textContent = response.limitation || "";
  loadAssistantStatus().catch(() => {});
  const citations = $("citations");
  citations.replaceChildren();
  for (const citation of response.citations || []) {
    const details = node("details");
    details.append(node("summary", citation.title + " · " + citation.section));
    details.addEventListener("toggle", async () => {
      if (!details.open || details.dataset.loaded) return;
      try {
        const data = await api("/knowledge/" + encodeURIComponent(citation.id));
        details.append(
          node("p", data.content),
          node("small", "Source: " + data.source),
        );
        details.dataset.loaded = "true";
      } catch (e) {
        notice(e.message);
      }
    });
    citations.append(details);
  }
}
$("local-status").onclick = () =>
  busy($("local-status"), async () =>
    answer(
      await api("/assistant", {
        method: "POST",
        body: JSON.stringify({ mode: "local" }),
      }),
    ),
  );
$("local-form").onsubmit = (event) => {
  event.preventDefault();
  const text = $("local-question").value.trim();
  if (!text) {
    notice("Enter a local question.");
    $("local-question").focus();
    return;
  }
  busy(event.submitter, async () =>
    answer(
      await api("/voice/local", {
        method: "POST",
        body: JSON.stringify({ text }),
      }),
    ),
  );
};
$("cloud-form").onsubmit = (event) => {
  event.preventDefault();
  busy(event.submitter, async () => {
    answer({ answer: "Retrieving supporting knowledge…", source: "local" });
    answer(
      await api("/assistant", {
        method: "POST",
        body: JSON.stringify({
          mode: $("cloud-mode").value,
          topic: $("cloud-topic").value,
          scenario: $("cloud-scenario").value,
        }),
      }),
    );
  });
};
function cloudMode() {
  const synthetic = $("cloud-mode").value === "synthetic";
  $("cloud-topic").disabled = synthetic;
  $("cloud-scenario").disabled = !synthetic;
}
$("cloud-mode").onchange = cloudMode;
cloudMode();
$("record-voice").onclick = () =>
  busy($("record-voice"), async () => {
    $("voice-status").textContent =
      "Recording locally for five seconds, then transcribing…";
    try {
      const result = await api("/voice/listen", { method: "POST" });
      $("voice-status").textContent = result.transcript
        ? "Heard locally: " + result.transcript
        : "No speech recognized.";
      answer(result);
    } catch (e) {
      $("voice-status").textContent =
        "Microphone is off. Typed questions remain available.";
      throw e;
    }
  });
async function loadHistory() {
  const sessions = await api("/sessions");
  const list = $("history-list");
  list.replaceChildren();
  if (!sessions.length) {
    list.append(
      node(
        "p",
        "No sessions yet. Start a webcam session or replay.",
        "caption",
      ),
    );
    return;
  }
  for (const session of sessions) {
    const row = node("div", undefined, "history-row");
    const detail = node("div");
    detail.append(
      node("strong", new Date(session.start * 1000).toLocaleString()),
      node(
        "p",
        `${session.mode === "replay" ? "Synthetic replay" : titleCase(session.mode)} · ${session.end ? "Saved" : "In progress"}${session.interrupted ? " · interrupted" : ""}`,
      ),
    );
    const controls = node("div", undefined, "inline-buttons");
    const open = node("button", "Open report", "secondary");
    open.onclick = () => busy(open, () => openReport(session.id));
    const del = node("button", "Delete", "quiet");
    del.disabled = !session.end;
    del.setAttribute(
      "aria-label",
      "Delete session from " + new Date(session.start * 1000).toLocaleString(),
    );
    del.onclick = () => {
      pendingDelete = session.id;
      deleteTrigger = del;
      $("delete-dialog").showModal();
      $("cancel-delete").focus();
    };
    controls.append(open, del);
    row.append(detail, controls);
    list.append(row);
  }
}
$("refresh-history").onclick = () => busy($("refresh-history"), loadHistory);
$("cancel-delete").onclick = () => {
  $("delete-dialog").close();
  deleteTrigger?.focus();
};
$("delete-dialog").addEventListener("close", () => deleteTrigger?.focus());
$("confirm-delete").onclick = () =>
  busy($("confirm-delete"), async () => {
    await api("/sessions/" + encodeURIComponent(pendingDelete), {
      method: "DELETE",
    });
    $("delete-dialog").close();
    await loadHistory();
    notice("Session and its measurements deleted.");
    $("refresh-history").focus();
  });
function reportChart(metrics) {
  const figure = node("figure");
  figure.style.margin = "24px 0";
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 700 120");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Recorded score trend");
  svg.style.width = "100%";
  svg.style.height = "120px";
  let points = [];
  const flush = () => {
    if (!points.length) return;
    const line = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "polyline",
    );
    line.setAttribute("points", points.join(" "));
    line.setAttribute("stroke", "#71d8cf");
    line.setAttribute("stroke-width", "2");
    line.setAttribute("fill", "none");
    svg.append(line);
    points = [];
  };
  metrics.forEach((m, i) => {
    if (m.score === null) {
      flush();
      return;
    }
    points.push(
      `${(i * 700) / Math.max(1, metrics.length - 1)},${110 - m.score}`,
    );
  });
  flush();
  figure.append(
    svg,
    node(
      "figcaption",
      "Recorded score / 0–100. Gaps indicate unavailable measurements.",
      "caption",
    ),
  );
  return figure;
}
async function openReport(id) {
  const data = await api("/sessions/" + encodeURIComponent(id) + "/report");
  route("report");
  const content = $("report-content");
  content.replaceChildren();
  content.append(
    node(
      "p",
      data.session.mode === "replay"
        ? "SYNTHETIC REPLAY · Artificial input"
        : "LOCAL WEBCAM SESSION",
      "eyebrow",
    ),
  );
  const summary = node("div", undefined, "report-summary");
  for (const [label, value] of [
    ["Duration", Math.round(data.summary.duration_s) + " s"],
    [
      "Tracking coverage",
      measurement(
        data.summary.tracking_coverage === null
          ? null
          : data.summary.tracking_coverage * 100,
        0,
        "%",
      ),
    ],
    ["Alerts", data.summary.alert_count],
  ]) {
    const card = node("div");
    card.append(node("span", label), node("strong", String(value)));
    summary.append(card);
  }
  content.append(
    summary,
    reportChart(data.metrics),
    node("h2", "Event timeline"),
  );
  const timeline = node("ol", undefined, "timeline");
  for (const event of data.events)
    timeline.append(
      node(
        "li",
        `${new Date(event.timestamp * 1000).toLocaleTimeString()} · ${event.severity || event.component || event.kind} · ${event.transition || event.status} · ${event.cause || ""}`,
      ),
    );
  if (!data.events.length)
    timeline.append(node("li", "No alert transitions recorded."));
  content.append(timeline);
  const samples = node("details");
  samples.append(node("summary", "Measured score samples"));
  const sampleList = node(
    "p",
    data.metrics
      .map(
        (m) =>
          `${new Date(m.captured_at * 1000).toLocaleTimeString()} · ${m.score === null ? "unavailable" : m.score}`,
      )
      .join(" / "),
    "caption",
  );
  samples.append(sampleList);
  content.append(samples);
  const exports = $("report-exports");
  exports.replaceChildren();
  for (const format of ["json", "csv", "html"]) {
    const a = node("a", "Export " + format.toUpperCase(), "export-link");
    a.href =
      "/api/v1/sessions/" + encodeURIComponent(id) + "/export?format=" + format;
    a.download = "sentinel-session." + format;
    exports.append(a);
  }
}
$("print-report").onclick = () => window.print();
async function loadSettings() {
  const controls = [...$("settings-form").querySelectorAll("input,button")];
  controls.forEach((el) => (el.disabled = true));
  try {
    const data = await api("/settings");
    settingsFields = Object.keys(data);
    for (const [key, value] of Object.entries(data)) {
      if (typeof value === "boolean") $(key).checked = value;
      else $(key).value = value;
    }
  } finally {
    controls.forEach((el) => (el.disabled = false));
  }
}
$("settings-form").onsubmit = (event) => {
  event.preventDefault();
  busy(event.submitter, async () => {
    const data = {};
    for (const key of settingsFields)
      data[key] =
        $(key).type === "checkbox" ? $(key).checked : Number($(key).value);
    try {
      await api("/settings", { method: "PATCH", body: JSON.stringify(data) });
      for (const el of $("settings-form").querySelectorAll("input")) {
        el.removeAttribute("aria-invalid");
        el.removeAttribute("aria-describedby");
      }
      $("settings-message").textContent =
        "Settings saved. They apply to your next session.";
      notice("Settings saved.");
    } catch (e) {
      $("settings-message").textContent = e.message;
      const invalid =
        Number($("fatigue_crit_threshold").value) <
        Number($("fatigue_warn_threshold").value) + 5
          ? $("fatigue_crit_threshold")
          : [...$("settings-form").querySelectorAll("input")].find(
              (el) => !el.validity.valid,
            );
      if (invalid) {
        invalid.setAttribute("aria-invalid", "true");
        invalid.setAttribute("aria-describedby", "settings-message");
        invalid.focus();
      }
      throw e;
    }
  });
};
// Controls are disabled in static markup until their external handlers are bound.
for (const el of document.querySelectorAll("button[disabled]"))
  el.disabled = false;
syncButtons();
route(location.hash.slice(1) || "live", false);
refreshHealth().catch((error) => notice(error.message));
