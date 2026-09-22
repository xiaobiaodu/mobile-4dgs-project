# WebGL-GS

This is a WebGL implementation of a real-time renderer with the only first-order Spherical Harmonics for 3DGS.
We significantly reduce GS storage by quantization and decode them in the `main.js` file.
All decoding and rendering functions are in `main.js` file.

## Dynamic scenes

The shared WebGL viewer supports dynamic Mobile-GS2 exports in addition to
static scenes. When a model contains `dynamic_enabled`, it decodes the
reference project's VQ-compressed `dynamic_code`, `dynamic_index`, and
`dynamic_htable` fields (or the legacy raw `velocity`, `acceleration`, `time`,
and `duration` fields).

For each Gaussian, the viewer evaluates the same FreeTimeGS motion model as
the training code:

```text
x(t) = x + velocity * (t - canonical_time)
       + 0.5 * acceleration * (t - canonical_time)^2
opacity(t) = opacity * exp(-0.5 * ((t - canonical_time) / exp(log_duration))^2)
```

### Static and dynamic Gaussians

Mobile-GS2 commits a hard per-Gaussian gate, so a trained model holds two
streams: time-invariant Gaussians and animated ones. The viewer reads the
exported `dynamic_gate_bits`/`dynamic_gate_count` metadata and loads both
streams separately.

- Static Gaussians keep their canonical position and opacity at every time
  step, which reproduces the reference renderer's `gate * trajectory`
  displacement and `(1 - gate) + gate * temporal` opacity exactly.
- Their projection and depth sort are cached per camera. A playback frame with
  an unchanged camera only re-projects, re-sorts and merges the animated
  subset, the same static-cache strategy as
  `render.py --exact_static_cache`. Changing the camera direction invalidates
  the cache and re-sorts both streams.
- Each stream is sorted with the reference renderer's 16 bit counting sort, and
  the two depth-ordered streams are merged with depth ties broken by the
  original Gaussian row id, so the single draw order stays a valid
  back-to-front order.
- The attribute-conditioned appearance residual is baked into the explicit SH
  coefficients once at load, exactly like the reference decoder
  (`gaussian_model.py:decode`), so no network is evaluated during playback or
  during sorting. The shader only advances the motion state.

The `#info` panel reports the `static / dynamic` Gaussian counts of a loaded
gated model, and each sorted frame carries a `sortStats` record
(`staticPoints`, `dynamicPoints`, `staticResorted`, `dynamicResorted`,
`merged`). Models without gate metadata keep the previous behaviour: every
Gaussian of a dynamic model is animated, and a split with only one non-empty
stream skips the merge pass entirely.

Dynamic scenes automatically show play/pause and timeline controls. Playback
loops over normalized time `[0, 1]`; use `?time=0.5` to choose the initial
time, `?dynamicAutoplay=false` to start paused, and
`?dynamicLoopSeconds=8` to set the loop length. If a loaded camera JSON has an
optional `time` field, selecting that camera pauses playback at its frame time.

`window.FLUX_GS_CONFIG` may also set `dynamicTime`, `dynamicAutoplay`,
`dynamicLoopSeconds`, `dynamicSortFps` (default `30`; use `0` for uncapped), and `cameraUrl` for a
scene page. `cameraUrl` should point to the Mobile-GS2 `cameras.json` that was
exported with that model; its intrinsics are rescaled to the actual WebGL
framebuffer before calculating Gaussian footprints.

Dynamic scenes advance GPU motion at display refresh rate by default. Set
`smoothDynamicPlayback: false` to restore worker-synchronized motion.

### Fill rate

`renderScale` (or `?renderScale=`) sets the drawing buffer as a fraction of the
size the viewer would otherwise pick, in `(0, 2]`; the browser then scales that
buffer up to the CSS box. A value below one trades sharpness for fragment
throughput without changing the geometry: the intrinsics, the projection and the
Gaussian footprints are all derived from `renderWidth`/`renderHeight`, so a
smaller buffer renders the same image, only softer. The default of `1` keeps the
previous behaviour, and the resolution actually in use is shown next to the FPS
counter.

Because Gaussian splatting is fill rate bound on mobile, this is usually a
larger lever than any sorting change: halving both axes quarters the shaded
pixels. `setRenderScale(0.5)` from the console applies a new factor in place,
without reloading the model, which makes a quick sweep practical:

```js
for (const scale of [1, 0.75, 0.5, 0.35, 0.25]) {
    setRenderScale(scale);
    await new Promise((done) => setTimeout(done, 3000));
}
```

### Adaptive sort scheduling

A playback frame only has to re-sort when the animated Gaussians can actually
reorder the draw list, so the worker bounds their travel through the view depth
axis and keeps the committed order while that movement is harmless. A reused
frame skips the re-projection, both counting sorts, the merge, the worker
message and the GPU index upload.

`adaptiveSortBudget` is the depth drift the viewer accepts, in mean Gaussian
radii (default `0.5`): a swap inside that drift only exchanges Gaussians that
already overlap on screen, and the drift of any pair that can end up swapped is
bounded by twice the budget. A budget of `0` selects the exact policy instead,
which reuses the order only when no neighbouring pair with an animated side can
even reach a tie. The exact policy costs one extra pass over the draw list per
commit and rarely fires on dense models, because the 16 bit counting sort
already resolves about four depth keys per bucket. `adaptiveSort: false`
restores fixed-rate sorting.

`tools/adaptive_sort_bench.js` measures the scheduler against fixed-rate sorting
on a synthetic gated model (200,000 Gaussians, 8% animated, 4 second clip, one
camera, requests at 30 Hz):

```text
                config    req    sorts    reuse       ms    ms/s   MB/s     vis%
           fixed 30 Hz    114      114        0      301    75.2   22.8     0.00
   adaptive r=0.25 @30    114       56       58      136    33.9   11.2     0.00
    adaptive r=0.5 @30    114       30       84       78    19.5    6.0     0.22
      adaptive r=1 @30    114       16       98       42    10.4    3.2     1.01
      adaptive r=0 @30    114      113        1      394    98.5   22.6     0.00
```

`ms/s` is worker sort time per second of playback and `MB/s` is the draw-order
payload crossing the worker boundary, which is also what the main thread no
longer uploads; `vis%` is the share of adjacent draw pairs that are out of depth
order by more than a quarter of a Gaussian radius.

### Measuring on a phone

`tools/bench_viewer/` is a page that runs the real viewer and measures what the
scheduler changes on the device. It wraps `requestAnimationFrame`, the WebGL
index upload and the worker messages before `render_shared/main.js` starts, so
the samples come from the shipping code path, and it reports presented frame
intervals, main-thread upload time, the worker's reply delay, long tasks, heap
growth and battery level. Measurements accumulate in a table with CSV and JSON
export, and the same table is mirrored to the console for remote debugging.

Serve the repository and open the page on the phone in landscape:

```sh
python -m http.server 8000
# http://<pc-ip>:8000/tools/bench_viewer/
```

| query | meaning |
| --- | --- |
| `model`, `camera` | model file under `modelBaseUrl`, and an optional `cameras.json`, both relative to the page |
| `cam` | `fixed` pins one camera before measuring (default); `carousel` keeps the orbiting demo camera |
| `sortFps` | `dynamicSortFps`, where `0` is uncapped |
| `adaptive`, `budget` | `adaptiveSort` and `adaptiveSortBudget` |
| `renderScale` | drawing buffer fraction, `1` is full size (fills the `scale` column) |
| `smooth`, `loop`, `sec`, `warm` | smooth playback, clip length, measured window, warmup |
| `auto` | start measuring as soon as the model renders |
| `panel` | `0` hides the panel; drive the bench from `window.BenchView` in the console instead |
| `sweep`, `si` | run a list of configurations, one reload each |

The Sweep button (or `sweep=1`) walks `fixed 30 Hz`, `adaptive r=0.25/0.5/1` and
`fixed 10 Hz` from the same camera and clip. A custom list goes in the URL as
`?sweep=label:adaptive:budget:sortFps|...`, and every step appends to the same
exported table. `?auto=1` runs a single configuration hands-free.

`tools/bench_viewer/smoke.html` checks the harness itself: it replays synthetic
worker traffic through the same hooks and reports `SMOKE PASS` or the checks
that failed, so a new device or browser can be validated without a model.

A frozen camera is the case where the scheduler can act: with the orbiting
camera the view changes every frame and the worker has to re-sort anyway, so both
configurations measure the same. The pinned camera is the viewer's own first
camera, the framing the page opens with, unless the page has already been looked
around in, in which case the view that was chosen is kept; that keeps a sweep
comparable across its reloads. What the measurements show on a device is the
work the scheduler removes, `up/s`, `upMB/s` and `upMs/s` for the index traffic a
reused order skips and `reuse%` for how often it fires. The frame rate only has
to move when the main thread was the bottleneck, because the depth sort itself
runs on a worker thread.

For temporally subsampled training data, set `dynamicLoopSeconds` from the
source sequence rather than the training sample count. For example, the N3DV
scene pages use `1200 / 30` seconds: their 300 training frames are interpolated
over the original 1,200-frame, 30 FPS timeline.
The on-screen FPS counter measures presented animation frames and is therefore
capped by the browser/display refresh rate.
The drag-and-drop loader distinguishes a camera JSON array from a Mobile-GS2
binary `.json` model, so locally exported dynamic `comp.json` files load as
models rather than being mistaken for camera metadata.


You can [try it out here](https://antimatter15.com/splat/).



https://github.com/antimatter15/splat/assets/30054/878d5d34-e0a7-4336-85df-111ff22daf4b



## 👍 **Acknowledgement**
This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!
* [splat](https://github.com/antimatter15/splat)
# mobile-gs2-project
