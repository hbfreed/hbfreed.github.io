// Story generation for the Tiny Dark Stories demo, shared by the page (onnxruntime-web) and the
// Node test (onnxruntime-node): both pass in their `ort` module.
//
// The models only ever see "Time: HH:MM\n" as a prompt, and their tokenizer splits digits one per
// token, so the header is built here directly instead of shipping a BPE tokenizer: "Time: " is
// [77, 438, 51, 246], digit d is 41 + d, ":" is 51 and the newline 224. Decoding uses the vocab
// (byte-level BPE, GPT-2's byte-to-unicode mapping).

export const EOS = 0;
const PREFIX = [77, 438, 51, 246], COLON = 51, NEWLINE = 224, DIGIT0 = 41;

export function headerIds(minute) {
  const hh = String(Math.floor(minute / 60)).padStart(2, "0"), mm = String(minute % 60).padStart(2, "0");
  return [...PREFIX, DIGIT0 + +hh[0], DIGIT0 + +hh[1], COLON, DIGIT0 + +mm[0], DIGIT0 + +mm[1], NEWLINE];
}

export function headerText(minute) {
  return `Time: ${String(Math.floor(minute / 60)).padStart(2, "0")}:${String(minute % 60).padStart(2, "0")}`;
}

// Night for these models is 18:00-05:59 (trained with the wraparound rule).
export const isNight = (minute) => minute >= 18 * 60 || minute < 6 * 60;

function byteDecoder() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); }
  const map = new Map();
  bs.forEach((b, i) => map.set(String.fromCharCode(cs[i]), b));
  return map;
}

// Streaming detokenizer: push token ids, get back the text they complete (UTF-8 may span tokens).
export function makeDetokenizer(vocab) {
  const bytesOf = byteDecoder();
  const tokenBytes = vocab.map((t) => Uint8Array.from([...t].map((c) => bytesOf.get(c) ?? 63)));
  const td = new TextDecoder("utf-8");
  return (id) => td.decode(tokenBytes[id], { stream: true });
}

// Top-k sampling. One pass keeps the k largest logits in order (k is small), instead of sorting the
// whole vocabulary every token.
function sample(logits, temperature, topK, rand) {
  const idx = new Int32Array(topK), val = new Float32Array(topK).fill(-Infinity);
  for (let i = 0; i < logits.length; i++) {
    const x = logits[i];
    if (x <= val[topK - 1]) continue;
    let j = topK - 1;
    for (; j > 0 && val[j - 1] < x; j--) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; }
    val[j] = x;
    idx[j] = i;
  }
  const w = new Float64Array(topK);
  let sum = 0;
  for (let k = 0; k < topK; k++) sum += (w[k] = Math.exp((val[k] - val[0]) / temperature));
  let r = rand() * sum;
  for (let k = 0; k < topK; k++) if ((r -= w[k]) <= 0) return idx[k];
  return idx[topK - 1];
}

// Yields token ids one at a time until the model ends the story (EOS) or maxTokens.
// cfg: {layers, kvHeads, headDim} of the exported Qwen3 model.
export async function* generate(ort, session, cfg, minute, { temperature = 0.5, topK = 40, maxTokens = 400, rand = Math.random } = {}) {
  const big = (xs) => BigInt64Array.from(xs, BigInt);
  const ones = new BigInt64Array(maxTokens + 16).fill(1n);    // attention mask, sliced each step
  let past = {};
  for (let l = 0; l < cfg.layers; l++)
    for (const kv of ["key", "value"])
      past[`past_key_values.${l}.${kv}`] = new ort.Tensor("float32", new Float32Array(0), [1, cfg.kvHeads, 0, cfg.headDim]);
  let ids = headerIds(minute), pos = 0;
  const wantsPositions = session.inputNames.includes("position_ids");
  for (let step = 0; step < maxTokens; step++) {
    const feed = {
      input_ids: new ort.Tensor("int64", big(ids), [1, ids.length]),
      attention_mask: new ort.Tensor("int64", ones.subarray(0, pos + ids.length), [1, pos + ids.length]),
      ...past,
    };
    if (wantsPositions)
      feed.position_ids = new ort.Tensor("int64", big(ids.map((_, i) => pos + i)), [1, ids.length]);
    const out = await session.run(feed);
    const logits = out.logits, V = logits.dims[2];
    const last = logits.data.subarray((ids.length - 1) * V, ids.length * V);
    pos += ids.length;
    const next = sample(last, temperature, topK, rand);
    if (next === EOS) return;
    yield next;
    ids = [next];
    const fresh = {};
    for (const name of Object.keys(past)) fresh[name] = out[name.replace("past_key_values", "present")];
    past = fresh;
  }
}
