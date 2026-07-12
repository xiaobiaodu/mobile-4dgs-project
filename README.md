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

Dynamic scenes automatically show play/pause and timeline controls. Playback
loops over normalized time `[0, 1]`; use `?time=0.5` to choose the initial
time, `?dynamicAutoplay=false` to start paused, and
`?dynamicLoopSeconds=8` to set the loop length. If a loaded camera JSON has an
optional `time` field, selecting that camera pauses playback at its frame time.

`window.FLUX_GS_CONFIG` may also set `dynamicTime`, `dynamicAutoplay`,
`dynamicLoopSeconds`, `dynamicSortFps` (default `30`), and `cameraUrl` for a
scene page. `cameraUrl` should point to the Mobile-GS2 `cameras.json` that was
exported with that model; its intrinsics are rescaled to the actual WebGL
framebuffer before calculating Gaussian footprints.
The drag-and-drop loader distinguishes a camera JSON array from a Mobile-GS2
binary `.json` model, so locally exported dynamic `comp.json` files load as
models rather than being mistaken for camera metadata.


You can [try it out here](https://antimatter15.com/splat/).



https://github.com/antimatter15/splat/assets/30054/878d5d34-e0a7-4336-85df-111ff22daf4b



## 👍 **Acknowledgement**
This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!
* [splat](https://github.com/antimatter15/splat)
