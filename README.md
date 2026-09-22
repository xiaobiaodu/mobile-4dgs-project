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


You can [try it out here](https://antimatter15.com/splat/).



https://github.com/antimatter15/splat/assets/30054/878d5d34-e0a7-4336-85df-111ff22daf4b



## 👍 **Acknowledgement**
This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!
* [splat](https://github.com/antimatter15/splat)
# mobile-gs2-project
