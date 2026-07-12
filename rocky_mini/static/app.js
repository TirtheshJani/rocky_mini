// Rocky Mini settings UI logic. Vanilla JS, no external deps.
const $ = (id) => document.getElementById(id);
const LATENCY_BUDGET = 2.5; // seconds, p50 budget
const METER_MAX = 5.0;

async function jsonFetch(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

function renderState(s) {
  $("chip-model").textContent = `model: ${s.model}`;
  $("chip-words").textContent = `knows ${s.known_words} words`;
  $("chip-stage").textContent = `stage ${s.stage}`;
  const sleep = $("chip-sleep");
  sleep.hidden = !s.sleep_watch;
  if ($("model-select").dataset.init !== "1") {
    $("model-select").value = s.model;
    $("url-input").value = s.llm_base_url || "";
    $("model-select").dataset.init = "1";
  }
}

function addMessage(who, text, meta) {
  const div = document.createElement("div");
  div.className = `msg ${who}`;
  if (meta && meta.leaked) div.classList.add("leak");
  div.textContent = text;
  if (meta && (meta.emotes?.length || meta.sfx?.length)) {
    const tags = document.createElement("span");
    tags.className = "tags";
    const bits = [];
    if (meta.emotes?.length) bits.push(`emote: ${meta.emotes.join(", ")}`);
    if (meta.sfx?.length) bits.push(`chord: ${meta.sfx.join(", ")}`);
    if (meta.leaked) bits.push("naivety: leak flagged");
    tags.textContent = bits.join("  ·  ");
    div.appendChild(tags);
  }
  const t = $("transcript");
  t.appendChild(div);
  t.scrollTop = t.scrollHeight;
}

function renderMetrics(m) {
  if (!m) return;
  $("m-ack").textContent = m.ack_ms ?? "—";
  $("m-ttfa").textContent = m.time_to_first_audio_s ?? "—";
  $("m-total").textContent = m.total_s ?? "—";
  $("m-prompt").textContent = m.prompt_eval_count ?? "—";
  $("m-eval").textContent = m.eval_count ?? "—";
  $("m-tools").textContent = `${m.tool_calls ?? 0}/${m.tool_errors ?? 0}`;
  const ttfa = m.time_to_first_audio_s;
  const fill = $("meter-fill");
  const status = $("meter-status");
  if (typeof ttfa === "number") {
    const pct = Math.min(100, (ttfa / METER_MAX) * 100);
    fill.style.width = pct + "%";
    const over = ttfa > LATENCY_BUDGET;
    fill.classList.toggle("warn", over);
    status.textContent = over ? "over budget" : "under budget";
    status.className = "meter-status " + (over ? "warn" : "good");
  }
}

async function refreshFacts() {
  const { facts } = await jsonFetch("/api/facts");
  const body = $("facts-body");
  body.innerHTML = "";
  if (!facts.length) {
    body.innerHTML = '<tr class="empty-row"><td colspan="5">Nothing yet. Teach Rocky something.</td></tr>';
    return;
  }
  for (const f of facts) {
    const tr = document.createElement("tr");
    tr.dataset.factId = f.id;
    tr.innerHTML = `
      <td>${escapeHtml(f.text)}</td>
      <td>${escapeHtml(f.category)}</td>
      <td><span class="conf ${f.confidence}">${f.confidence}</span></td>
      <td>${f.heard_count}</td>
      <td class="row-actions">
        <button class="ghost act-confirm" data-id="${f.id}">confirm</button>
        <button class="ghost act-delete" data-id="${f.id}">delete</button>
      </td>`;
    body.appendChild(tr);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function refreshState() {
  renderState(await jsonFetch("/api/state"));
}

// -- events ----------------------------------------------------------------
$("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("chat-input");
  const text = input.value.trim();
  if (!text) return;
  addMessage("user", text);
  input.value = "";
  try {
    const r = await jsonFetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const spoken = r.spoken || r.reply;
    const emotes = [].concat(...r.chunks.map((c) => c.emotes));
    const sfx = [].concat(...r.chunks.map((c) => c.sfx));
    addMessage("rocky", spoken, { emotes, sfx, leaked: r.leaked });
    renderMetrics(r.metrics);
    renderState(r.state);
    await refreshFacts();
  } catch (err) {
    addMessage("rocky", "Signal bad. Wait, wait, wait. (" + err.message + ")");
  }
});

$("barge-btn").addEventListener("click", async () => {
  await jsonFetch("/api/barge_in", { method: "POST" });
  addMessage("rocky", "…");
});

document.querySelectorAll(".emote-btn").forEach((btn) => {
  btn.addEventListener("click", async () => {
    await jsonFetch("/api/emote", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: btn.dataset.emote }),
    });
    btn.animate([{ transform: "scale(1)" }, { transform: "scale(0.94)" }, { transform: "scale(1)" }], { duration: 180 });
  });
});

$("facts-body").addEventListener("click", async (e) => {
  const id = e.target?.dataset?.id;
  if (!id) return;
  if (e.target.classList.contains("act-confirm")) {
    await jsonFetch(`/api/facts/${id}/confirm`, { method: "POST" });
  } else if (e.target.classList.contains("act-delete")) {
    await jsonFetch(`/api/facts/${id}`, { method: "DELETE" });
  }
  await refreshFacts();
  await refreshState();
});

$("settings-save").addEventListener("click", async () => {
  const s = await jsonFetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: $("model-select").value, llm_base_url: $("url-input").value }),
  });
  renderState(s);
  $("settings-save").textContent = "Saved";
  setTimeout(() => ($("settings-save").textContent = "Save settings"), 1200);
});

// SSE metrics stream (live updates when turns happen elsewhere).
function connectMetrics() {
  try {
    const es = new EventSource("/api/metrics/stream");
    es.onmessage = (ev) => {
      try { renderMetrics(JSON.parse(ev.data).metrics); } catch (_) {}
    };
    es.onerror = () => es.close();
  } catch (_) {}
}

// -- init ------------------------------------------------------------------
(async function init() {
  await refreshState();
  await refreshFacts();
  connectMetrics();
})();
