// Pre- and post-processing of the lane U-Net, ported from src/lanedet (no DOM, so Node can
// run it too: scripts/web_parity checks it against the Python pipeline).

export const INPUT_W = 512;
export const INPUT_H = 288;
export const SLOTS = ["left_left", "left", "right", "right_right"];

// OpenCV INTER_AREA downscale of interleaved 8-bit pixels (RGB or RGBA) to RGB, rounded to
// uint8 like cv2.resize does. Each output pixel averages the source rectangle it covers,
// with fractional weights at the edges.
function areaTable(src, dst) {
  const scale = src / dst;
  const tab = [];
  for (let d = 0; d < dst; d++) {
    const f0 = d * scale;
    const f1 = f0 + scale;
    const s0 = Math.ceil(f0);
    const s1 = Math.floor(f1);
    const taps = [];
    if (s0 - f0 > 1e-3) taps.push([s0 - 1, (s0 - f0) / scale]);
    for (let s = s0; s < s1; s++) taps.push([s, 1 / scale]);
    if (f1 - s1 > 1e-3 && s1 < src) taps.push([s1, Math.min(1, f1 - s1) / scale]);
    tab.push(taps);
  }
  return tab;
}

export function resizeArea(pixels, srcW, srcH, channels, dstW = INPUT_W, dstH = INPUT_H) {
  const xt = areaTable(srcW, dstW);
  const yt = areaTable(srcH, dstH);
  const rows = new Float64Array(dstW * 3);
  const out = new Uint8Array(dstW * dstH * 3);
  for (let dy = 0; dy < dstH; dy++) {
    const acc = new Float64Array(dstW * 3);
    for (const [sy, wy] of yt[dy]) {
      rows.fill(0);
      const base = sy * srcW * channels;
      for (let dx = 0; dx < dstW; dx++) {
        let r = 0, g = 0, b = 0;
        for (const [sx, wx] of xt[dx]) {
          const i = base + sx * channels;
          r += pixels[i] * wx;
          g += pixels[i + 1] * wx;
          b += pixels[i + 2] * wx;
        }
        rows[dx * 3] = r;
        rows[dx * 3 + 1] = g;
        rows[dx * 3 + 2] = b;
      }
      for (let k = 0; k < acc.length; k++) acc[k] += rows[k] * wy;
    }
    for (let k = 0; k < acc.length; k++) {
      out[dy * dstW * 3 + k] = Math.min(255, Math.max(0, Math.round(acc[k])));
    }
  }
  return out;
}

// Resized RGB uint8 (h, w, 3) -> CHW float32 in [0, 1]: the exported graph's input.
export function toInput(rgb, w = INPUT_W, h = INPUT_H) {
  const plane = w * h;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    out[i] = rgb[i * 3] / 255;
    out[plane + i] = rgb[i * 3 + 1] / 255;
    out[2 * plane + i] = rgb[i * 3 + 2] / 255;
  }
  return out;
}

// Least-squares polynomial x = f(y) (same solution as numpy.polyfit). y is rescaled to
// [-1, 1] first so the normal equations stay well conditioned.
export function polyfit(ys, xs, degree) {
  const lo = Math.min(...ys);
  const hi = Math.max(...ys);
  const mid = (lo + hi) / 2;
  const half = (hi - lo) / 2 || 1;
  const n = degree + 1;
  const a = Array.from({ length: n }, () => new Float64Array(n + 1));
  for (let i = 0; i < ys.length; i++) {
    const t = (ys[i] - mid) / half;
    const pow = [1];
    for (let k = 1; k < 2 * n; k++) pow.push(pow[k - 1] * t);
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) a[r][c] += pow[r + c];
      a[r][n] += pow[r] * xs[i];
    }
  }
  for (let c = 0; c < n; c++) { // Gaussian elimination with partial pivoting
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
    [a[c], a[p]] = [a[p], a[c]];
    for (let r = c + 1; r < n; r++) {
      const f = a[r][c] / a[c][c];
      for (let k = c; k <= n; k++) a[r][k] -= f * a[c][k];
    }
  }
  const coef = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = a[r][n];
    for (let k = r + 1; k < n; k++) s -= a[r][k] * coef[k];
    coef[r] = s / a[r][r];
  }
  return (y) => {
    const t = (y - mid) / half;
    let v = 0;
    for (let k = n - 1; k >= 0; k--) v = v * t + coef[k];
    return v;
  };
}

// Port of lanedet.deep.model.extract_lanes: per lane slot and row, the probability-weighted
// x around the row's peak; rows below `thresh` are skipped; a low-order polynomial removes
// pixel noise and is resampled at 72 rows over the detected span.
export function extractLanes(probs, numClasses, h, w, outW, outH,
                             { thresh = 0.4, minRows = 8, window = 6, degree = 2 } = {}) {
  const sx = outW / w;
  const sy = outH / h;
  const lanes = {};
  for (let cls = 1; cls < numClasses; cls++) {
    const off = cls * h * w;
    const px = [];
    const py = [];
    for (let y = 0; y < h; y++) {
      const row = off + y * w;
      let peak = 0;
      for (let x = 1; x < w; x++) if (probs[row + x] > probs[row + peak]) peak = x;
      if (probs[row + peak] < thresh) continue;
      const lo = Math.max(peak - window, 0);
      const hi = Math.min(peak + window + 1, w);
      let num = 0, den = 0;
      for (let x = lo; x < hi; x++) {
        num += x * probs[row + x];
        den += probs[row + x];
      }
      px.push(num / den);
      py.push(y);
    }
    if (px.length < minRows) continue;
    const f = polyfit(py, px, Math.min(degree, px.length - 1));
    const y0 = Math.min(...py);
    const y1 = Math.max(...py);
    lanes[SLOTS[cls - 1]] = Array.from({ length: 72 }, (_, i) => {
      const y = y0 + ((y1 - y0) * i) / 71;
      return [f(y) * sx, y * sy];
    });
  }
  return lanes;
}
