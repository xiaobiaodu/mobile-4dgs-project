window.FLUX_GS_CONFIG = {
    defaultModel: "chicken.json",
    // HyperNeRF's chicken capture contains 164 frames recorded at 15 FPS.
    dynamicLoopSeconds: 164 / 15,
    // Motion stays GPU-smooth while expensive depth/appearance corrections
    // run at the capture rate in the background.
    dynamicSortFps: 15,
    smoothDynamicPlayback: true,
    dynamicShDuringPlayback: false,
};
