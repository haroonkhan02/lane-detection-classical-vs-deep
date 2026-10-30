// Run the browser demo's pipeline (web/lanes.js + onnxruntime-web WASM) on frames dumped by
// scripts/web_parity.py and write the lanes it finds.
//
// Usage: node parity.mjs <dump dir> <model.onnx>

import fs from "node:fs";
import path from "node:path";
import * as ort from "onnxruntime-web";

import { extractLanes, INPUT_H, INPUT_W, resizeArea, toInput } from "../../web/lanes.js";

const [dir, modelPath] = process.argv.slice(2);
const meta = JSON.parse(fs.readFileSync(path.join(dir, "meta.json"), "utf8"));
const session = await ort.InferenceSession.create(fs.readFileSync(modelPath),
  { executionProviders: ["wasm"] });

const results = [];
let total = 0;
for (const item of meta.images) {
  const rgb = fs.readFileSync(path.join(dir, item.file));
  const start = performance.now();
  const input = toInput(resizeArea(rgb, item.width, item.height, 3));
  const out = await session.run({
    image: new ort.Tensor("float32", input, [1, 3, INPUT_H, INPUT_W]),
  });
  const probs = out.probs;
  const lanes = extractLanes(probs.data, probs.dims[1], INPUT_H, INPUT_W, item.width,
    item.height);
  total += performance.now() - start;
  results.push({ file: item.file, lanes });
}
fs.writeFileSync(path.join(dir, "js_lanes.json"), JSON.stringify({
  ort_version: ort.env.versions.web, ms_per_frame: total / results.length, results,
}));
console.log(`${results.length} frames, ${(total / results.length).toFixed(1)} ms/frame (WASM)`);
