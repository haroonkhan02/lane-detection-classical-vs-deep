import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";

import { extractLanes, INPUT_H, INPUT_W, resizeArea, toInput } from "./lanes.js";

const MODEL_URL = "https://huggingface.co/haroonx02/lane-detection-unet-resnet34/resolve/main/"
  + "onnx/lane_unet_int8.onnx";
const EXAMPLES = ["1_multilane.jpg", "2_curve.jpg", "3_truck_occlusion.jpg",
  "4_concrete_patch.jpg", "5_udacity_other_camera.jpg"];
const COLORS = { left_left: "#00a0ff", left: "#ffdc00", right: "#ffdc00", right_right: "#00a0ff" };

const view = document.getElementById("view");
const ctx = view.getContext("2d");
const status = document.getElementById("status");
let session = null;

async function fetchWithProgress(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`model download failed: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    status.textContent = total
      ? `Downloading the model… ${Math.round((100 * got) / total)} %`
      : `Downloading the model… ${(got / 1e6).toFixed(1)} MB`;
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return buf;
}

function drawLanes(lanes) {
  if (lanes.left && lanes.right) { // fill only rows both ego lanes cover, as in viz.draw_lanes
    const l = lanes.left, r = lanes.right;
    const y0 = Math.max(l[0][1], r[0][1]);
    const y1 = Math.min(l[l.length - 1][1], r[r.length - 1][1]);
    if (y1 - y0 >= 10) {
      const inside = (pts) => pts.filter(([, y]) => y >= y0 && y <= y1);
      const poly = [...inside(l), ...inside(r).reverse()];
      ctx.fillStyle = "rgba(0, 200, 0, 0.3)";
      ctx.beginPath();
      poly.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.fill();
    }
  }
  ctx.lineWidth = Math.max(3, view.width / 210);
  ctx.lineJoin = "round";
  for (const [slot, pts] of Object.entries(lanes)) {
    ctx.strokeStyle = COLORS[slot];
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.stroke();
  }
}

async function detect(img) {
  view.width = img.width;
  view.height = img.height;
  ctx.drawImage(img, 0, 0);
  if (!session) return;
  const { data, width, height } = ctx.getImageData(0, 0, view.width, view.height);
  const t0 = performance.now();
  const input = toInput(resizeArea(data, width, height, 4));
  const t1 = performance.now();
  const out = await session.run({ image: new ort.Tensor("float32", input,
    [1, 3, INPUT_H, INPUT_W]) });
  const t2 = performance.now();
  const lanes = extractLanes(out.probs.data, out.probs.dims[1], INPUT_H, INPUT_W, width, height);
  const t3 = performance.now();
  drawLanes(lanes);
  const n = Object.keys(lanes).length;
  status.textContent = `${n} lane${n === 1 ? "" : "s"} · model ${(t2 - t1).toFixed(0)} ms `
    + `(WASM, ${ort.env.wasm.numThreads} thread${ort.env.wasm.numThreads === 1 ? "" : "s"}) · `
    + `pre ${(t1 - t0).toFixed(0)} ms · post ${(t3 - t2).toFixed(0)} ms`;
  window.lastLanes = lanes; // for automated checks against the Python pipeline
}

// Decode without colour management: OpenCV ignores embedded ICC profiles, and the model was
// trained on the raw pixel values.
async function loadImage(src) {
  const res = await fetch(src);
  if (!res.ok) throw new Error(`could not load ${src}`);
  return createImageBitmap(await res.blob(), { colorSpaceConversion: "none" });
}

async function show(src, button) {
  document.querySelectorAll(".examples button").forEach((b) =>
    b.setAttribute("aria-pressed", String(b === button)));
  try {
    await detect(await loadImage(src));
  } catch (err) {
    status.textContent = String(err.message || err);
  }
}

const strip = document.getElementById("examples");
for (const name of EXAMPLES) {
  const b = document.createElement("button");
  b.type = "button";
  b.title = name.replace(/^\d_|\.jpg$/g, "").replaceAll("_", " ");
  b.innerHTML = `<img src="examples/${name}" alt="${b.title}">`;
  b.addEventListener("click", () => show(`examples/${name}`, b));
  strip.appendChild(b);
}
document.getElementById("file").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) show(URL.createObjectURL(file), null);
});

window.detectExample = (name) => show(`examples/${name}`,
  [...strip.children][EXAMPLES.indexOf(name)]);

(async () => {
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
  try {
    const model = await fetchWithProgress(MODEL_URL);
    status.textContent = "Starting the model…";
    session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
    await show(`examples/${EXAMPLES[0]}`, strip.children[0]);
  } catch (err) {
    status.textContent = `Could not start the model: ${err.message || err}`;
  }
})();
