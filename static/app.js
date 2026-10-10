(() => {
  "use strict";

  /* ───────────────────────── Setup ───────────────────────── */
  const $ = (id) => document.getElementById(id);
  const startBtn = $("startBtn"), endBtn = $("endBtn");
  const stateBadge = $("stateBadge"), stateLabel = $("stateLabel");
  const liveTranscript = $("liveTranscript");
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  const STATE_LABELS = { idle: "Idle", listening: "Listening", thinking: "Thinking", speaking: "Speaking" };

  let state = "idle";
  let callActive = false;
  let callId = 0;           // bumps on every call start/end so stale async work can be ignored
  let sessionId = null;
  let recognition = null;
  let voice = null;
  let interimBubble = null;
  let silentRestarts = 0;

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  /* ───────────────────── A. State machine + UI ───────────────────── */
  function setState(next) {
    state = next;
    stateBadge.className = "state-badge state-" + next;
    stateLabel.textContent = STATE_LABELS[next];
    startBtn.disabled = callActive;
    endBtn.disabled = !callActive;
  }

  function showError(msg) {
    $("errorText").textContent = msg;
    $("errorBanner").hidden = false;
  }
  function hideError() { $("errorBanner").hidden = true; }
  $("errorClose").addEventListener("click", hideError);

  /* ───────────────────── E. Rendering ───────────────────── */
  function makeBubble(speaker, text, extraClass) {
    const b = el("div", `bubble ${speaker}${extraClass ? " " + extraClass : ""}`);
    b.appendChild(el("span", "who", speaker === "agent" ? "Aria" : "You"));
    b.appendChild(el("span", "", text));
    return b;
  }

  function addBubble(speaker, text) {
    liveTranscript.appendChild(makeBubble(speaker, text));
    liveTranscript.scrollTop = liveTranscript.scrollHeight;
  }

  function showInterim(text) {
    if (!text) return;
    if (!interimBubble) {
      interimBubble = makeBubble("customer", text, "interim");
      liveTranscript.appendChild(interimBubble);
    } else {
      interimBubble.lastChild.textContent = text;
    }
    liveTranscript.scrollTop = liveTranscript.scrollHeight;
  }

  function clearInterim() {
    if (interimBubble) { interimBubble.remove(); interimBubble = null; }
  }

  function chipFor(status) {
    const cls = status === "Delivered" ? "chip-delivered"
              : status === "Processing" ? "chip-processing" : "chip-out";
    return el("span", `chip ${cls}`, status);
  }

  async function copyText(text, btn) {
    try {
      await navigator.clipboard.writeText(text);
      if (btn) { const old = btn.textContent; btn.textContent = "Copied"; setTimeout(() => (btn.textContent = old), 1200); }
    } catch { /* clipboard may be blocked; ignore */ }
  }

  function renderOrders(orders) {
    const wrap = $("ordersCard");
    wrap.textContent = "";
    const table = el("table", "orders");
    const head = el("tr");
    ["Order", "Customer", "Product", "Value", "Status", "Notes", ""].forEach((h) => head.appendChild(el("th", "", h)));
    table.appendChild(head);
    orders.forEach((o) => {
      const tr = el("tr");
      tr.appendChild(el("td", "", o.order_id));
      tr.appendChild(el("td", "", o.customer_name));
      tr.appendChild(el("td", "", o.product));
      tr.appendChild(el("td", "", "₹" + o.value_inr));
      const st = el("td"); st.appendChild(chipFor(o.status)); tr.appendChild(st);
      const notes = [o.carrier && o.tracking_id ? `${o.carrier} ${o.tracking_id}` : null, o.notes].filter(Boolean).join(" · ");
      tr.appendChild(el("td", "", notes));
      const act = el("td");
      const btn = el("button", "copy-btn", "Copy ID");
      btn.type = "button";
      btn.addEventListener("click", () => copyText(o.order_id, btn));
      act.appendChild(btn); tr.appendChild(act);
      table.appendChild(tr);
    });
    wrap.appendChild(table);
  }

  function renderSummary(data) {
    const panel = $("summaryPanel");
    const box = $("summaryTranscript");
    box.textContent = "";
    (data.transcript || []).forEach((t) => {
      const b = makeBubble(t.speaker, t.text);
      if (t.ts) b.querySelector(".who").textContent +=
        " · " + new Date(t.ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
      box.appendChild(b);
    });
    $("outcomeJson").textContent = data.outcome
      ? JSON.stringify(data.outcome, null, 2)
      : (data.error || "No summary available.");
    panel.hidden = false;
    panel.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* ───────────────────── D. API layer ───────────────────── */
  async function api(path, body) {
    const res = await fetch("/api" + path, {
      method: body ? "POST" : "GET",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    return res.json();
  }

  async function loadOrders() {
    try { renderOrders((await api("/orders")).orders); }
    catch { $("ordersCard").textContent = "Could not load test orders."; }
  }

  /* ───────────────────── C. Speech synthesis (TTS) ───────────────────── */
  function pickVoice() {
    const voices = window.speechSynthesis.getVoices();
    if (!voices.length) return;
    voice =
      voices.find((v) => v.lang === "en-IN") ||
      voices.find((v) => /india/i.test(v.name)) ||
      voices.find((v) => v.lang === "hi-IN") ||
      voices.find((v) => v.lang.startsWith("en")) ||
      null;
  }
  if ("speechSynthesis" in window) {
    pickVoice();
    window.speechSynthesis.addEventListener("voiceschanged", pickVoice);
  }

  function splitSentences(text) {
    return (text.match(/[^.!?]+[.!?]*\s*/g) || [text]).map((s) => s.trim()).filter(Boolean);
  }

  // Speaks sentence by sentence (avoids Chrome cutting off long utterances), then calls onDone.
  function speak(text, onDone) {
    const synth = window.speechSynthesis;
    const myCall = callId;
    synth.cancel();
    setState("speaking");
    const parts = splitSentences(text);
    let i = 0;
    const next = () => {
      if (myCall !== callId || !callActive) return; // call ended: stop quietly
      if (i >= parts.length) { onDone && onDone(); return; }
      const u = new SpeechSynthesisUtterance(parts[i++]);
      if (voice) u.voice = voice;
      u.lang = voice ? voice.lang : "en-IN";
      u.rate = 1.03;
      u.onend = next;
      u.onerror = next;
      synth.speak(u);
    };
    next();
  }

  /* ───────────────────── B. Speech recognition (STT) ───────────────────── */
  function listen() {
    if (!callActive) return;
    setState("listening");

    const rec = new SpeechRecognition();
    recognition = rec;
    let gotFinal = false;
    rec.lang = "en-IN";
    rec.interimResults = true;
    rec.continuous = false;
    rec.maxAlternatives = 1;

    rec.onresult = (e) => {
      let interim = "", final = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i];
        if (r.isFinal) final += r[0].transcript; else interim += r[0].transcript;
      }
      if (final.trim()) { gotFinal = true; handleFinal(final.trim()); }
      else showInterim(interim);
    };

    rec.onerror = (e) => {
      if (e.error === "not-allowed" || e.error === "service-not-allowed") {
        showError("Microphone access is blocked. Allow it in your browser's site settings and start the call again.");
        endCall();
      } else if (e.error === "network") {
        showError("Speech recognition lost its network connection. Check your internet and try again.");
      } else if (e.error === "no-speech") {
        silentRestarts++;
      }
      // "aborted" and others fall through to onend
    };

    rec.onend = () => {
      if (recognition !== rec) return; // superseded, or the turn was handed off
      if (callActive && state === "listening" && !gotFinal) {
        clearInterim();
        if (silentRestarts >= 3) {
          silentRestarts = 0;
          const nudge = "I'm still here whenever you're ready.";
          addBubble("agent", nudge);
          speak(nudge, listen);
          return;
        }
        setTimeout(() => { if (callActive && state === "listening" && recognition === rec) listen(); }, 250);
      }
    };

    try { rec.start(); } catch { /* already started */ }
  }

  async function handleFinal(text) {
    const myCall = callId;
    recognition = null; // stops onend from restarting; mic stays off until the agent finishes
    silentRestarts = 0;
    clearInterim();
    addBubble("customer", text);
    setState("thinking");

    let reply;
    try {
      reply = (await api("/chat", { session_id: sessionId, message: text })).reply;
    } catch {
      showError("Could not reach the server.");
      reply = "Sorry, I'm having trouble connecting. Could you please say that again?";
    }
    if (myCall !== callId || !callActive) return;
    addBubble("agent", reply);
    speak(reply, listen);
  }

  /* ───────────────────── Call lifecycle ───────────────────── */
  async function startCall() {
    hideError();
    if (!SpeechRecognition) {
      showError("This browser doesn't support speech recognition. Please use Chrome or Edge.");
      return;
    }
    startBtn.disabled = true;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((t) => t.stop()); // permission prompt only
    } catch {
      showError("Microphone access was denied. Allow it in your browser and try again.");
      startBtn.disabled = false;
      return;
    }

    let session;
    try { session = await api("/session", {}); }
    catch {
      showError("Could not reach the server. Please try again in a moment.");
      startBtn.disabled = false;
      return;
    }

    liveTranscript.textContent = "";
    $("summaryPanel").hidden = true;
    sessionId = session.session_id;
    callActive = true;
    callId++;
    silentRestarts = 0;
    addBubble("agent", session.greeting);
    speak(session.greeting, listen);
  }

  async function endCall() {
    if (!callActive) return;
    const sid = sessionId;
    callActive = false;
    callId++;
    try { recognition && recognition.abort(); } catch {}
    recognition = null;
    window.speechSynthesis.cancel();
    clearInterim();
    setState("idle");

    $("summaryTranscript").textContent = "";
    $("outcomeJson").textContent = "Generating summary…";
    $("summaryPanel").hidden = false;
    try { renderSummary(await api("/summarize", { session_id: sid })); }
    catch {
      $("outcomeJson").textContent = "Could not generate the summary.";
      showError("Could not generate the call summary.");
    }
  }

  /* ───────────────────── Wiring ───────────────────── */
  startBtn.addEventListener("click", startCall);
  endBtn.addEventListener("click", endCall);

  // Hints: click to copy, then say it aloud during the call
  document.querySelectorAll("#hints .hint-btn").forEach((b) =>
    b.addEventListener("click", () => copyText(b.textContent, null)));

  window.addEventListener("beforeunload", () => {
    try { recognition && recognition.abort(); } catch {}
    window.speechSynthesis.cancel();
  });

  setState("idle");
  loadOrders();
})();
