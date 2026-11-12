// 回复判断面板：通过 window.AstrBotPluginView 与插件后端通信。
const bridge = window.AstrBotPluginView || window.AstrBotPluginPage;

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
  overview: null,
  config: null, // 服务器上的配置
  personas: [],
  providers: { decision: [], chat: [], default_decision: "" },
  people: { users: [], groups: [] },
  dirty: false,
};

// ---------------------------------------------------------------- 工具
function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function shortId(id) {
  const s = String(id || "");
  return s.length > 14 ? `${s.slice(0, 6)}…${s.slice(-4)}` : s;
}

function ago(ts) {
  if (!ts) return "—";
  const sec = Math.max(0, Date.now() / 1000 - ts);
  if (sec < 60) return "刚刚";
  if (sec < 3600) return `${Math.floor(sec / 60)} 分钟前`;
  if (sec < 86400) return `${Math.floor(sec / 3600)} 小时前`;
  return `${Math.floor(sec / 86400)} 天前`;
}

function splitNames(text) {
  return String(text || "")
    .split(/[,，、\n]+/)
    .map((s) => s.trim())
    .filter((s, i, arr) => s && arr.indexOf(s) === i);
}

let toastTimer = null;
function toast(message, isError = false) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.toggle("err", isError);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, isError ? 4500 : 2200);
}

function unwrap(result) {
  if (result && typeof result === "object" && result.status === "error") {
    throw new Error(result.message || "请求失败");
  }
  if (result && typeof result === "object" && result.status === "ok" && "data" in result) {
    return result.data;
  }
  return result;
}

async function apiGet(endpoint, params) {
  return unwrap(await bridge.apiGet(endpoint, params));
}

async function apiPost(endpoint, body) {
  return unwrap(await bridge.apiPost(endpoint, body));
}

// 面板在 AstrBot 的 iframe 里，浏览器可能禁止 confirm() 弹窗（会直接返回「取消」）。
// 所以危险操作改成：第一次点按钮变成「再点一次确认」，3 秒内再点才执行。
const pendingConfirm = new WeakMap();
function confirmTwice(button, label = "再点一次确认") {
  if (pendingConfirm.has(button)) {
    clearTimeout(pendingConfirm.get(button).timer);
    button.textContent = pendingConfirm.get(button).original;
    pendingConfirm.delete(button);
    return true;
  }
  const original = button.textContent;
  button.textContent = label;
  const timer = setTimeout(() => {
    button.textContent = original;
    pendingConfirm.delete(button);
  }, 3000);
  pendingConfirm.set(button, { original, timer });
  return false;
}

async function guarded(fn) {
  try {
    return await fn();
  } catch (err) {
    console.error(err);
    toast(err?.message || String(err), true);
    return undefined;
  }
}

// ---------------------------------------------------------------- 标签页
function showTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $$(".pane").forEach((p) => p.classList.toggle("active", p.dataset.pane === name));
  if (name === "lists") guarded(loadPeople);
  if (name === "overview") guarded(loadOverview);
  if (name === "debug") guarded(loadDebug);
}

// ---------------------------------------------------------------- 概览
async function loadOverview() {
  const data = await apiGet("overview");
  state.overview = data;
  state.providers = data.providers || state.providers;

  const enabled = !!data.enabled;
  $("#run-dot").className = `dot ${enabled ? "on" : "off"}`;
  $("#master-switch").checked = enabled;
  $("#master-label").textContent = enabled ? "已启用" : "已停用";

  const decisionId = data.decision_provider_id || data.providers?.default_decision || "";
  let judge = "不使用模型";
  if (data.judge_enabled) {
    judge = data.use_decision_model
      ? (data.providers?.decision?.length ? `判断模型 ${decisionId || "默认"}` : "对话模型（未配置判断模型）")
      : "对话模型";
  }
  $("#subtitle").textContent = enabled ? `${judge} · 阈值 ${Number(data.threshold).toFixed(2)}` : "插件已停用，机器人不会在群里主动接话";

  $("#warnings").innerHTML = (data.warnings || [])
    .map((w) => `<div class="warning">⚠ ${esc(w)}</div>`)
    .join("");
  setDebugIndicator(!!data.debug_enabled);

  const stats = data.stats || {};
  const stages = stats.stages || {};
  const blocked = (stages["用户黑名单"] || 0) + (stages["群不在白名单"] || 0) + (stages["群在黑名单"] || 0);
  const filtered = (stages["没有实质内容"] || 0) + (stages["冷却中"] || 0);
  const cards = [
    { num: stats.total ?? 0, label: "24 小时内判断" },
    { num: stats.replied ?? 0, label: "决定回复" },
    { num: stats.at_blocked ?? 0, label: "被 @ 时判断后不回" },
    { num: blocked, label: "名单拦下" },
    { num: filtered, label: "规则过滤（太短 / 冷却）" },
    { num: stages["判断失败"] || 0, label: "判断失败" },
  ];
  $("#stat-cards").innerHTML = cards
    .map((c) => `<div class="card stat"><div class="num">${esc(c.num)}</div><div class="label">${esc(c.label)}</div></div>`)
    .join("");

  renderHistory();
}

function verdictTag(item) {
  if (item.stage === "判断失败") return `<span class="tag err">${item.reply ? "回复" : "不回"}</span>`;
  return item.reply ? '<span class="tag yes">回复</span>' : '<span class="tag no">不回</span>';
}

function probCell(item) {
  if (item.scores) {
    // 主动接话：显示判断出的类别，三个分数放在悬停提示里
    const ctx = typeof item.context_count === "number" ? `；前文 ${item.context_count} 条` : "";
    return `<span class="stage-cell" title="${esc(item.detail + ctx)}">${esc(item.stage)}</span>`;
  }
  if (typeof item.probability !== "number") {
    return `<span class="muted">${esc(item.stage)}${item.detail && item.stage !== "判断模型" ? `：${esc(item.detail)}` : ""}</span>`;
  }
  const th = state.overview?.threshold ?? 0.5;
  const pct = Math.round(item.probability * 100);
  return `<span class="prob" title="${esc(item.detail)}">
    <span class="prob-bar"><i style="width:${pct}%"></i><b style="left:${Math.round(th * 100)}%"></b></span>
    <span class="mono">${item.probability.toFixed(2)}</span></span>`;
}

function renderHistory() {
  const filter = $("#history-filter").value;
  const blacklist = new Set((state.config?.user_blacklist || []).map((x) => x.toLowerCase()));
  let rows = state.overview?.history || [];
  if (filter === "reply") rows = rows.filter((r) => r.reply);
  if (filter === "model") rows = rows.filter((r) => typeof r.probability === "number" || r.stage === "对话模型");
  $("#history-empty").hidden = rows.length > 0;
  $("#history-table tbody").innerHTML = rows
    .map((r) => {
      const black = r.sender_id && blacklist.has(String(r.sender_id).toLowerCase());
      let action = r.sender_id && !black
        ? `<button class="btn ghost small danger" data-black="${esc(r.sender_id)}" title="加入黑名单">拉黑</button>`
        : black ? '<span class="tag black">黑名单</span>' : "";
      if (r.has_debug && r.id != null) {
        action = `<button class="btn ghost small" data-debug-id="${esc(r.id)}">详情</button> ${action}`;
      }
      const atTag = r.path === "被@时判断" ? '<span class="tag at" title="这条消息 @ 了机器人，同时也 @ 了别人">被@</span> ' : "";
      return `<tr>
        <td class="nowrap mono">${esc(r.time)}</td>
        <td class="nowrap" title="${esc(r.sender_id)}">${esc(r.speaker)}</td>
        <td class="text">${atTag}${esc(r.text)}</td>
        <td>${verdictTag(r)}</td>
        <td>${probCell(r)}</td>
        <td>${action}</td>
      </tr>`;
    })
    .join("");
}

// ---------------------------------------------------------------- 名单
async function loadPeople() {
  const data = await apiGet("people");
  state.people = data;
  if (state.config) {
    state.config.user_blacklist = data.users.filter((u) => u.blacklisted).map((u) => u.id);
    state.config.group_whitelist = data.group_whitelist || [];
    state.config.group_blacklist = data.group_blacklist || [];
  }
  renderPeople();
}

function renderPeople() {
  const { users = [], groups = [] } = state.people;

  const black = users.filter((u) => u.blacklisted);
  $("#blacklist-chips").innerHTML = black.length
    ? black
        .map((u) => `<span class="chip">${esc(u.name || "（最近没出现）")}
          <span class="mono">${esc(shortId(u.id))}</span>
          <button title="移出黑名单" data-unblack="${esc(u.id)}">×</button></span>`)
        .join("")
    : '<span class="muted">黑名单是空的。</span>';

  const q = $("#user-search").value.trim().toLowerCase();
  const seen = users.filter((u) => u.count > 0)
    .filter((u) => !q || u.name.toLowerCase().includes(q) || u.id.toLowerCase().includes(q));
  $("#user-empty").hidden = seen.length > 0;
  $("#user-table tbody").innerHTML = seen
    .map((u) => `<tr>
      <td>${esc(u.name || "—")}</td>
      <td class="mono" title="${esc(u.id)}">${esc(u.id)}</td>
      <td>${esc(u.count)}</td>
      <td class="nowrap">${esc(ago(u.last_seen))}</td>
      <td>${u.blacklisted
        ? `<button class="btn ghost small" data-unblack="${esc(u.id)}">移出黑名单</button>`
        : `<button class="btn ghost small danger" data-black="${esc(u.id)}">拉黑</button>`}</td>
    </tr>`)
    .join("");

  const wl = state.people.group_whitelist || [];
  $("#group-mode").textContent = wl.length
    ? `白名单模式：只在勾选白名单的 ${wl.length} 个群里主动接话`
    : "白名单为空：所有群都生效（黑名单里的除外）";
  $("#group-empty").hidden = groups.length > 0;
  $("#group-table tbody").innerHTML = groups
    .map((g) => `<tr>
      <td class="mono" title="${esc(g.umo)}">${esc(shortId(g.id))}</td>
      <td>${esc(g.persona || "—")}</td>
      <td>${esc(g.count)}</td>
      <td class="nowrap">${esc(ago(g.last_seen))}</td>
      <td><input type="checkbox" data-group-list="group_white" data-id="${esc(g.id)}" ${g.whitelisted ? "checked" : ""}></td>
      <td><input type="checkbox" data-group-list="group_black" data-id="${esc(g.id)}" ${g.blacklisted ? "checked" : ""}></td>
    </tr>`)
    .join("");
}

async function toggleList(list, id, on) {
  const res = await apiPost("list/toggle", { list, id, on });
  const key = res.list;
  if (state.config && key) state.config[key] = res.items;
  return res;
}

async function setBlacklist(id, on) {
  await guarded(async () => {
    await toggleList("user", id, on);
    toast(on ? "已加入黑名单" : "已移出黑名单");
    await loadPeople();
    renderHistory();
  });
}

// ---------------------------------------------------------------- 判断设置
function providerOptions(list, emptyLabel, current) {
  const ids = list.map((p) => p.id);
  const opts = [`<option value="">${esc(emptyLabel)}</option>`];
  list.forEach((p) => {
    const extra = p.model ? `（${p.model}）` : "";
    opts.push(`<option value="${esc(p.id)}">${esc(p.id + extra)}</option>`);
  });
  if (current && !ids.includes(current)) {
    opts.push(`<option value="${esc(current)}">${esc(current)}（已不存在）</option>`);
  }
  return opts.join("");
}

async function loadConfig() {
  const data = await apiGet("config");
  state.config = data.config;
  state.personas = data.personas || [];
  state.providers = data.providers || state.providers;
  fillForm();
  fillTestPersonas();
}

function fillForm() {
  const cfg = state.config;
  const form = $("#settings-form");
  const prov = state.providers;

  const defaultDecision = prov.default_decision ? `AstrBot 默认（当前是 ${prov.default_decision}）` : "AstrBot 默认";
  $("#decision-select").innerHTML = prov.decision.length
    ? providerOptions(prov.decision, defaultDecision, cfg.decision_provider_id)
    : '<option value="">还没有配置判断模型</option>';
  $("#llm-select").innerHTML = providerOptions(prov.chat, "当前会话的对话模型", cfg.llm_provider_id);

  // 人格别称：列出所有人格，以及配置里有、但人格已被删掉的
  const aliases = cfg.persona_aliases || {};
  const names = [...state.personas];
  Object.keys(aliases).forEach((n) => { if (!names.includes(n)) names.push(n); });
  $("#persona-rows").innerHTML = names.length
    ? names
        .map((n) => `<div class="persona-row">
          <div class="persona-name">${esc(n)}${state.personas.includes(n) ? "" : "<small>人格已不存在</small>"}</div>
          <input type="text" data-persona="${esc(n)}" placeholder="别称，多个用逗号隔开，例如 小${esc(n.slice(0, 2))}"
            value="${esc((aliases[n] || []).join("，"))}" />
        </div>`)
        .join("")
    : '<p class="muted">没有读到人格。可以在 AstrBot 的「人格」里新建。</p>';

  for (const el of form.elements) {
    if (!el.name || !(el.name in cfg)) continue;
    const value = cfg[el.name];
    if (el.type === "checkbox") el.checked = !!value;
    else if (Array.isArray(value)) el.value = value.join("，");
    else el.value = value ?? "";
  }
  updateRangeLabels();
  updateDependent();
  setDirty(false);
}

function readForm() {
  const form = $("#settings-form");
  const out = {};
  for (const el of form.elements) {
    if (!el.name || !(el.name in state.config)) continue;
    if (el.type === "checkbox") out[el.name] = el.checked;
    else if (el.type === "number" || el.type === "range") out[el.name] = Number(el.value);
    else if (Array.isArray(state.config[el.name])) out[el.name] = splitNames(el.value);
    else out[el.name] = el.value;
  }
  const aliases = {};
  $$("[data-persona]").forEach((el) => {
    const list = splitNames(el.value);
    if (list.length) aliases[el.dataset.persona] = list;
  });
  out.persona_aliases = aliases;
  return out;
}

function updateRangeLabels() {
  const form = $("#settings-form");
  $("#threshold-value").textContent = Number(form.elements.threshold.value).toFixed(2);
  $("#open-threshold-value").textContent = Number(form.elements.open_threshold.value).toFixed(2);
  $("#open-chance-value").textContent = `${Math.round(Number(form.elements.open_question_chance.value) * 100)}%`;
  $("#prob-value").textContent = Number(form.elements.probability_without_judge.value).toFixed(2);
}

function updateDependent() {
  const form = $("#settings-form");
  const judge = form.elements.judge_enabled.checked;
  const useDecision = form.elements.use_decision_model.checked;
  const fallback = form.elements.llm_fallback_enabled.checked;
  form.elements.use_decision_model.disabled = !judge;
  form.elements.decision_provider_id.disabled = !judge || !useDecision;
  form.elements.threshold.disabled = !judge;
  form.elements.open_threshold.disabled = !judge;
  form.elements.open_question_chance.disabled = !judge;
  form.elements.llm_fallback_enabled.disabled = !judge;
  form.elements.llm_provider_id.disabled = !judge || !fallback;
  form.elements.image_use_llm.disabled = !judge || !fallback;
  form.elements.probability_without_judge.disabled = judge;
}

function setDirty(dirty) {
  state.dirty = dirty;
  $(".savebar").classList.toggle("dirty", dirty);
  $("#save-status").textContent = dirty ? "有未保存的改动" : "没有改动";
  $("#btn-save").disabled = !dirty;
  $("#btn-reset").disabled = !dirty;
}

async function saveConfig(event) {
  event.preventDefault();
  await guarded(async () => {
    $("#btn-save").disabled = true;
    const res = await apiPost("config/save", { config: readForm() });
    state.config = res.config;
    fillForm();
    toast(res.changed?.length ? "已保存，立即生效" : "没有需要保存的改动");
    await loadOverview();
  });
  if (state.dirty) $("#btn-save").disabled = false;
}

// ---------------------------------------------------------------- 总开关
async function toggleMaster() {
  const on = $("#master-switch").checked;
  await guarded(async () => {
    const res = await apiPost("config/save", { config: { enabled: on } });
    if (state.config) state.config.enabled = res.config.enabled;
    toast(on ? "已启用" : "已停用：机器人不会在群里主动接话");
    await loadOverview();
  });
  if (state.config) {
    const box = $("#settings-form").elements.enabled;
    if (box) box.checked = state.config.enabled;
  }
}

// ---------------------------------------------------------------- 调试
function setDebugIndicator(on) {
  $("#debug-badge").hidden = !on;
  $("#debug-switch").checked = on;
  $("#debug-label").textContent = on ? "调试模式已打开" : "调试模式已关闭";
}

function renderContext(lines) {
  if (!Array.isArray(lines) || !lines.length) return '<p class="muted">（没有前文）</p>';
  return `<ul class="ctx">${lines
    .map((l) => `<li class="${l.speaker === "机器人自己" ? "bot" : ""}"><span class="who">${esc(l.speaker)}</span>${esc(l.text)}</li>`)
    .join("")}</ul>`;
}

const SCORE_LABELS = { to_bot: "对机器人说", to_other: "对别人说", open: "对全群说", new_topic: "新话题" };

function renderScores(scores, threshold, openThreshold) {
  if (!scores || typeof scores !== "object") return "";
  const ot = Number.isFinite(openThreshold) ? openThreshold : 0.6;
  const marks = { to_bot: threshold, to_other: 0.4, open: ot, new_topic: ot };
  return `<div class="scores">${Object.entries(SCORE_LABELS)
    .filter(([key]) => typeof scores[key] === "number")
    .map(([key, label]) => {
      const v = scores[key];
      return `<div class="score-row"><span>${label}</span>
        <div class="meter small"><i style="width:${Math.round(v * 100)}%"></i><b style="left:${Math.round((marks[key] ?? 0.5) * 100)}%"></b></div>
        <span class="mono">${v.toFixed(2)}</span></div>`;
    })
    .join("")}</div>`;
}

function renderDebugItem(r) {
  const prob = typeof r.probability === "number"
    ? `<span class="mono">${r.probability.toFixed(2)} / 阈值 ${Number(r.threshold ?? 0).toFixed(2)}</span>`
    : "";
  const atTag = r.path === "被@时判断" ? '<span class="tag at">被@</span>' : "";
  const changed = r.raw_text && r.judge_text && r.raw_text !== r.judge_text;
  const rows = [
    ["时间", r.time],
    ["群", r.group_id],
    ["谁说的", `${r.speaker}（${r.sender_id}）`],
    ["人格 / 名字", `${r.persona || "无"} → ${(r.names || []).join("、")}`],
    ["收到的原文", r.raw_text],
    changed ? ["判断用的文字", r.judge_text] : null,
    ["@ 了谁", (r.mentions || []).length ? r.mentions.join("、") : "没有 @ 别人"],
    ["@ 了机器人", r.at_bot ? "是" : "否"],
    r.quote ? ["引用了", `${r.quote.speaker}：${r.quote.text}`] : null,
    r.images ? ["图片", `${r.images} 张${r.images_sent ? `，${r.images_sent} 张发给了对话模型` : "，模型没有看到图片内容"}`] : null,
    r.named != null ? ["叫了机器人名字", r.named ? "是" : "否"] : null,
    r.model_input ? ["交给模型的前文", `${(r.context || []).length} 条`] : null,
    ["结果", `${r.reply ? "回复" : "不回"}（${r.stage}${r.detail ? "，" + r.detail : ""}）`],
    ["用的模型", r.model || "没有调用模型"],
    r.elapsed_ms != null ? ["用时", `${r.elapsed_ms} 毫秒`] : null,
    (r.failures || []).length ? ["失败原因", r.failures.join("；")] : null,
    r.llm_raw ? ["对话模型原始回答", r.llm_raw] : null,
  ].filter(Boolean);
  return `<details class="debug-item" id="debug-${esc(r.id)}">
    <summary>
      <span class="mono muted">${esc(r.time)}</span>
      ${atTag}
      <span>${esc(r.speaker)}</span>
      <span class="sum-text">${esc(r.judge_text || r.raw_text)}</span>
      ${r.reply ? '<span class="tag yes">回复</span>' : '<span class="tag no">不回</span>'}
      <span class="muted">${esc(r.stage)}</span>
      ${prob}
    </summary>
    <div class="debug-body">
      <dl class="kv">${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>
      ${renderScores(r.scores, Number(r.threshold ?? 0.5), Number(r.open_threshold ?? 0.75))}
      ${r.model_input ? `<div class="muted">交给模型的前文</div>${renderContext(r.context)}` : '<p class="muted">没有走到模型这一步，在前面的规则就决定了。</p>'}
      ${r.question ? `<details class="raw"><summary>判断标准（交给模型的问题）</summary><pre class="json">${esc(r.question)}</pre></details>` : ""}
      ${r.model_input ? `<details class="raw"><summary>交给模型的完整输入（JSON）</summary><pre class="json">${esc(JSON.stringify(r.model_input, null, 2))}</pre></details>` : ""}
    </div>
  </details>`;
}

async function loadDebug() {
  const data = await apiGet("debug", { limit: 500 });
  setDebugIndicator(!!data.enabled);
  const records = data.records || [];
  $("#debug-summary").textContent = data.enabled
    ? `正在记录，共 ${data.count} 条（最多保存 ${data.max_records} 条，超过后删掉最早的）。`
    : `调试模式已关闭，不再记录新的判断。已保存 ${data.count} 条。`;
  $("#debug-empty").hidden = records.length > 0;
  $("#debug-list").innerHTML = records.map(renderDebugItem).join("");
}

async function openDebug(id) {
  showTab("debug");
  await guarded(loadDebug);
  const el = document.getElementById(`debug-${id}`);
  if (!el) { toast("这条记录的调试详情已经被清理了", true); return; }
  el.open = true;
  el.classList.add("flash");
  el.scrollIntoView({ block: "center", behavior: "smooth" });
  setTimeout(() => el.classList.remove("flash"), 1600);
}

async function toggleDebug() {
  const on = $("#debug-switch").checked;
  await guarded(async () => {
    const res = await apiPost("config/save", { config: { debug_enabled: on } });
    if (state.config) state.config.debug_enabled = res.config.debug_enabled;
    const box = $("#settings-form").elements.debug_enabled;
    if (box) box.checked = res.config.debug_enabled;
    toast(on ? "调试模式已打开，之后的判断会保存完整过程" : "调试模式已关闭，已有记录保留");
    await loadDebug();
  });
}

// ---------------------------------------------------------------- 试一试
const EXAMPLES = {
  other: {
    speaker: "小红",
    context: "小明：周末一起去爬山吗\n机器人：听起来不错，记得多带点水。",
    text: "小明，那你周六早上来接我吧",
  },
  follow: {
    speaker: "甲",
    context: "甲：你一般几点下班\n机器人：晚上十点。",
    text: "这么晚啊，那我晚点再来找你",
  },
  chat: {
    speaker: "甲",
    context: "甲：晚上打球吗\n乙：几点",
    text: "七点老地方，你记得带球",
  },
  at: {
    speaker: "甲",
    context: "",
    // 机器人的名字取当前选中的人格
    text: () => `@小明，和@${$("#test-persona").value || "机器人"} 说句话`,
  },
};

function fillTestPersonas() {
  const groups = state.people.groups || [];
  const used = groups.map((g) => g.persona).find(Boolean);
  const list = state.personas.length ? state.personas : [""];
  $("#test-persona").innerHTML = list
    .map((n) => `<option value="${esc(n)}" ${n === used ? "selected" : ""}>${esc(n || "（无人格）")}</option>`)
    .join("");
}

function parseContext(text) {
  const botWords = new Set(["机器人", "bot", "我", $("#test-persona").value].filter(Boolean));
  return String(text || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const m = line.match(/^([^:：]{1,20})[:：]\s*(.+)$/);
      if (!m) return { speaker: "群友", text: line, is_bot: false };
      const speaker = m[1].trim();
      return { speaker, text: m[2], is_bot: botWords.has(speaker) };
    });
}

async function runTest() {
  const text = $("#test-text").value.trim();
  if (!text) { toast("先填要判断的这句话", true); return; }
  const box = $("#test-result");
  box.hidden = false;
  box.innerHTML = '<span class="muted">判断中…</span>';
  $("#btn-test").disabled = true;
  try {
    const res = await apiPost("test", {
      text,
      speaker: $("#test-speaker").value,
      persona: $("#test-persona").value,
      context: parseContext($("#test-context").value),
    });
    const prob = typeof res.probability === "number" ? res.probability : null;
    const meter = res.scores ? renderScores(res.scores, Number(res.threshold), Number(res.open_threshold)) : prob === null ? "" : `
      <div class="meter"><i style="width:${Math.round(prob * 100)}%"></i><b style="left:${Math.round(res.threshold * 100)}%"></b></div>
      <div class="muted">概率 ${prob.toFixed(2)} · 阈值 ${Number(res.threshold).toFixed(2)}（竖线）</div>`;
    const atNote = res.path === "被@时判断"
      ? `<p><span class="tag at">被@</span> 这句话同时 @ 了机器人和 ${esc((res.mentions || []).join("、"))}，按「被 @ 时判断」的标准。</p>`
      : (res.mentions || []).length ? `<p class="muted">@ 了：${esc(res.mentions.join("、"))}</p>` : "";
    box.innerHTML = `
      <div class="big ${res.reply ? "yes" : "no"}">${res.reply ? "会回复" : "不回复"}</div>
      ${meter}
      ${atNote}
      <p>依据：${esc(res.stage)}${res.detail ? `，${esc(res.detail)}` : ""} · 用时 ${esc(res.elapsed_ms)} 毫秒</p>
      ${res.stage === "开放问题" || res.stage === "新话题" ? `<p class="muted">${res.stage === "新话题" ? "这句话开了个新话题" : "这是对全群的开放问题"}，实际在群里会按 ${Math.round(Number(res.open_question_chance) * 100)}% 的概率接话。</p>` : ""}
      <p class="muted">当作机器人名字的词：${esc((res.names || []).join("、"))}${res.named ? "（这句话里叫了名字）" : ""}</p>
      ${res.model_input ? `<details class="raw"><summary>交给模型的完整输入</summary><pre class="json">${esc(JSON.stringify(res.model_input, null, 2))}</pre></details>` : ""}`;
  } catch (err) {
    box.innerHTML = `<span class="tag err">出错</span> ${esc(err?.message || err)}`;
  } finally {
    $("#btn-test").disabled = false;
  }
}

// ---------------------------------------------------------------- 事件
function bindEvents() {
  $("#tabs").addEventListener("click", (e) => {
    const tab = e.target.closest(".tab");
    if (tab) showTab(tab.dataset.tab);
  });
  $("#btn-refresh").addEventListener("click", () => guarded(refreshAll));
  $("#master-switch").addEventListener("change", toggleMaster);
  $("#history-filter").addEventListener("change", renderHistory);
  $("#btn-clear-history").addEventListener("click", (e) => guarded(async () => {
    if (!confirmTwice(e.currentTarget, "再点一次确认清空")) {
      toast("清空所有判断记录？名单和设置不受影响。3 秒内再点一次确认。");
      return;
    }
    await apiPost("history/clear", {});
    toast("已清空");
    await loadOverview();
  }));

  document.addEventListener("click", (e) => {
    const black = e.target.closest("[data-black]");
    if (black) setBlacklist(black.dataset.black, true);
    const unblack = e.target.closest("[data-unblack]");
    if (unblack) setBlacklist(unblack.dataset.unblack, false);
    const example = e.target.closest("[data-example]");
    if (example) {
      const ex = EXAMPLES[example.dataset.example];
      $("#test-speaker").value = ex.speaker;
      $("#test-context").value = ex.context;
      $("#test-text").value = typeof ex.text === "function" ? ex.text() : ex.text;
      runTest();
    }
    const detail = e.target.closest("[data-debug-id]");
    if (detail) openDebug(detail.dataset.debugId);
  });

  $("#debug-switch").addEventListener("change", toggleDebug);
  $("#btn-debug-refresh").addEventListener("click", () => guarded(loadDebug));
  $("#btn-debug-export").addEventListener("click", () => guarded(async () => {
    await bridge.download("debug/export", {}, "reply-gate-debug.json");
    toast("已导出");
  }));
  $("#btn-debug-clear").addEventListener("click", (e) => guarded(async () => {
    if (!confirmTwice(e.currentTarget, "再点一次确认清空")) {
      toast("清空所有调试记录？判断记录、名单和设置不受影响。3 秒内再点一次确认。");
      return;
    }
    await apiPost("debug/clear", {});
    toast("调试记录已清空");
    await loadDebug();
  }));

  $("#blacklist-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const id = $("#blacklist-input").value.trim();
    if (!id) return;
    const submit = e.currentTarget.querySelector('button[type="submit"]');
    if (/^\d{5,11}$/.test(id) && !confirmTwice(submit, "仍然加入")) {
      toast("这看起来像 QQ 号。QQ 官方机器人用的是 32 位 openid，QQ 号不会生效。确定要加，3 秒内再点一次。", true);
      return;
    }
    $("#blacklist-input").value = "";
    setBlacklist(id, true);
  });
  $("#user-search").addEventListener("input", renderPeople);

  $("#group-table").addEventListener("change", (e) => {
    const box = e.target.closest("[data-group-list]");
    if (!box) return;
    guarded(async () => {
      await toggleList(box.dataset.groupList, box.dataset.id, box.checked);
      toast("已保存");
      await loadPeople();
    });
  });

  const form = $("#settings-form");
  form.addEventListener("input", () => { updateRangeLabels(); updateDependent(); setDirty(true); });
  form.addEventListener("change", () => { updateDependent(); setDirty(true); });
  form.addEventListener("submit", saveConfig);
  $("#btn-reset").addEventListener("click", () => fillForm());

  $("#btn-test").addEventListener("click", runTest);
  $("#test-text").addEventListener("keydown", (e) => { if (e.key === "Enter") runTest(); });
}

async function refreshAll() {
  await loadConfig();
  await Promise.all([loadOverview(), loadPeople()]);
  fillTestPersonas();
}

async function main() {
  bindEvents();
  if (!bridge) {
    $("#subtitle").textContent = "请在 AstrBot 管理面板的插件页里打开这个面板。";
    return;
  }
  try {
    await bridge.ready();
  } catch (_) { /* 旧版本没有 ready，直接继续 */ }
  await guarded(refreshAll);
}

main();
