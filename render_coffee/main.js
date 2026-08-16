window.FLUX_GS_CONFIG = {
    defaultModel: "coffee.json",
    // N3DV has 1,200 source frames at 30 FPS. The 4DGS was trained on
    // 300 uniformly sampled frames; smooth playback evaluates the learned
    // motion continuously to reconstruct the three frames between samples.
    dynamicLoopSeconds: 1200/30/4,
    dynamicSortFps: 30,
    smoothDynamicPlayback: true,
    dynamicShDuringPlayback: false,
};
