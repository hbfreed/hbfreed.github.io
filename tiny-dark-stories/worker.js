// The storyteller's engine, off the page's main thread: downloads a model (with progress, kept in the
// browser's Cache Storage so later visits skip the download), sets it up in ONNX Runtime and streams
// token ids back to the page. Setting up the 74 MB Big model takes seconds of solid work; here that no
// longer freezes the page.
//
// Messages in:  {type: "init", wasm}  wasm = URL of ONNX Runtime's binary, or null to let it fetch its own
//               {type: "generate", run, name, files, cfg, minute, temperature}
//               {type: "stop"}
// Messages out: {type: "progress", name, loaded, total}  {type: "preparing", name}
//               {type: "token", run, id, ms}  {type: "done", run}  {type: "error", run, message}

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.bundle.min.mjs";
import { generate } from "./tds.js";

ort.env.wasm.numThreads = 1;
const CACHE = "tiny-dark-stories-models-v1";
const sessions = {};          // name -> Promise<InferenceSession>, so a model never downloads twice
let runtime = null, current = null;
let writing = Promise.resolve();   // one story at a time: a new one waits for the old one to stop

// A zero-delay yield to the worker's event loop, so a "stop" can arrive between two tokens.
const yieldNow = (() => {
  const ch = new MessageChannel(), waiting = [];
  ch.port1.onmessage = () => waiting.shift()?.();
  return () => new Promise((resolve) => { waiting.push(resolve); ch.port2.postMessage(0); });
})();

async function cached(url) {
  try {
    const hit = await (await caches.open(CACHE)).match(url);
    return hit ? new Uint8Array(await hit.arrayBuffer()) : null;
  } catch {
    return null;                // no Cache Storage here (e.g. a sandboxed frame): just download
  }
}

async function remember(url, bytes) {
  try { await (await caches.open(CACHE)).put(url, new Response(bytes)); } catch { /* best effort */ }
}

async function download(name, files) {
  const progress = (loaded, total) => postMessage({ type: "progress", name, loaded, total });
  if (files[0].endsWith(".json")) {        // base64 chunks (the Claude artifact build)
    let b64 = "";
    for (let k = 0; k < files.length; k++) {
      b64 += (await (await fetch(files[k])).json()).base64;
      progress(k + 1, files.length);
    }
    const bin = atob(b64), out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  const url = files[0];
  const hit = await cached(url);
  if (hit) return hit;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed with HTTP ${res.status}`);
  const total = +res.headers.get("content-length") || 0, reader = res.body.getReader(), parts = [];
  let loaded = 0, last = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    if (performance.now() - last > 100) { progress(loaded, total); last = performance.now(); }
  }
  const bytes = new Uint8Array(loaded);
  let at = 0;
  for (const p of parts) { bytes.set(p, at); at += p.length; }
  remember(url, bytes);
  return bytes;
}

function load(name, files) {
  sessions[name] ??= (async () => {
    await runtime;
    const bytes = await download(name, files);
    postMessage({ type: "preparing", name });
    return ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  })().catch((err) => { delete sessions[name]; throw err; });    // let a later attempt retry
  return sessions[name];
}

onmessage = async ({ data: m }) => {
  if (m.type === "init") {
    runtime = m.wasm ? fetch(m.wasm).then((r) => r.arrayBuffer()).then((b) => { ort.env.wasm.wasmBinary = b; }) : Promise.resolve();
  } else if (m.type === "stop") {
    current = null;
  } else if (m.type === "generate") {
    current = m.run;
    let release;
    const previous = writing;
    writing = new Promise((resolve) => { release = resolve; });
    try {
      const session = await load(m.name, m.files);
      await previous;
      if (current !== m.run) return;
      const tokens = generate(ort, session, m.cfg, m.minute, { temperature: m.temperature });
      for (;;) {
        const t = performance.now();
        const { value: id, done } = await tokens.next();
        if (done) break;
        if (current !== m.run) { await tokens.return(); return; }
        postMessage({ type: "token", run: m.run, id, ms: performance.now() - t });
        await yieldNow();
      }
      if (current === m.run) postMessage({ type: "done", run: m.run });
    } catch (err) {
      postMessage({ type: "error", run: m.run, message: err.message || String(err) });
    } finally {
      release();
    }
  }
};
