// Browser side of the adaptive-sort bench: it loads the real viewer, wraps the
// few entry points the scheduler goes through, and collects the samples that
// metrics.js aggregates.
//
// The script order in index.html matters.  This file runs before
// render_shared/main.js so that
//   * the viewer starts with the configuration described by the URL,
//   * Worker, requestAnimationFrame and WebGL are already wrapped when the
//     viewer first touches them.
//
// The page itself only provides the stylesheet and the panel below: the viewer
// injects its own DOM through ensureViewerDom().
(function () {
    "use strict";

    const M = window.BenchMetrics;
    const config = M.parseConfig(location.search);

    window.FLUX_GS_CONFIG = {
        defaultModel: config.model,
        dynamicSortFps: config.sortFps,
        smoothDynamicPlayback: config.smooth === 1,
        dynamicLoopSeconds: config.loop,
        adaptiveSort: config.adaptive === 1,
        adaptiveSortBudget: config.adaptive === 1 ? config.budget : 0,
        dynamicAutoplay: true,
    };
    if (config.camera) window.FLUX_GS_CONFIG.cameraUrl = config.camera;

    const STORE_KEY = "benchViewer.sweep";
    const SWEEP_SPEC_KEY = "sweep";

    const state = {
        samples: null,      // active sample bucket, null while idle
        busy: false,
        ready: false,
        phase: "loading",
        sawCommit: false,
        gl: null,
        battery: { manager: null, charging: null },
        batteryTimer: null,
        results: [],
        stopRequested: false,
        scene: {
            vertexCount: 0,
            staticCount: null,
            dynamicCount: null,
            dynamicEnabled: null,
            adaptiveRequested: null,
            adaptiveBudget: null,
        },
    };

    function readStore() {
        const empty = { spec: "", entries: [], index: 0, reports: [], hash: "" };
        try {
            const raw = sessionStorage.getItem(STORE_KEY);
            if (!raw) return empty;
            const store = JSON.parse(raw);
            return Object.assign(empty, store);
        } catch (error) {
            return empty;
        }
    }

    function writeStore(store) {
        try {
            sessionStorage.setItem(STORE_KEY, JSON.stringify(store));
        } catch (error) {
            // Private browsing modes can refuse sessionStorage; the run still
            // works, only the sweep cannot carry its table across reloads.
        }
    }

    function sweepSpec() {
        return new URLSearchParams(location.search).get(SWEEP_SPEC_KEY) || "";
    }

    // Polling rather than setTimeout so that Stop can cut a measurement short
    // without leaving a timer behind that would end the next one.
    function waitFor(milliseconds) {
        return new Promise((resolve) => {
            const deadline = performance.now() + milliseconds;
            const tick = () => {
                if (state.stopRequested || performance.now() >= deadline) resolve();
                else setTimeout(tick, 100);
            };
            tick();
        });
    }

    function heapBytes() {
        return performance.memory ? performance.memory.usedJSHeapSize : 0;
    }

    function renderSize() {
        const gl = state.gl;
        if (!gl) return "";
        return gl.drawingBufferWidth + "x" + gl.drawingBufferHeight;
    }

    function glRenderer() {
        const gl = state.gl;
        if (!gl) return "";
        try {
            const info = gl.getExtension("WEBGL_debug_renderer_info");
            if (info) return String(gl.getParameter(info.UNMASKED_RENDERER_WEBGL));
            return String(gl.getParameter(gl.VERSION));
        } catch (error) {
            return "";
        }
    }

    // `viewMatrix` is a top level binding of render_shared/main.js.  Classic
    // scripts share the global lexical scope, so it can be read from here once
    // the viewer has run, which is how a measurement pins its camera.
    function currentViewMatrix() {
        try {
            if (typeof viewMatrix !== "undefined" &&
                Array.isArray(viewMatrix) && viewMatrix.length === 16) {
                return viewMatrix.slice();
            }
        } catch (error) {
            // Still in the temporal dead zone, i.e. the viewer has not loaded.
        }
        return null;
    }

    // The viewer opens on an orbiting demo camera, so pinning "whatever the
    // camera shows right now" would give every reload, and every step of a
    // sweep, a different viewpoint.  A page that the user has already looked
    // around in is a choice of view and is kept; otherwise the viewer's own
    // first camera is used, which is the framing the page opens with.
    let userMovedCamera = false;
    for (const type of ["keydown", "pointerdown", "touchstart", "wheel"]) {
        window.addEventListener(type, () => {
            userMovedCamera = true;
        }, { capture: true, passive: true });
    }

    // `cameras` and `getViewMatrix` are top level bindings of
    // render_shared/main.js, reachable from here because classic scripts share
    // one global scope.  Both only exist once that script has run.
    function defaultCameraMatrix() {
        try {
            if (typeof cameras !== "undefined" && cameras.length > 0 &&
                typeof getViewMatrix === "function") {
                return getViewMatrix(cameras[0]);
            }
        } catch (error) {
            // Still in the temporal dead zone, i.e. the viewer has not loaded.
        }
        return null;
    }

    // The viewer parses location.hash as a view matrix, and that path also
    // stops its orbiting demo camera, so pinning a view is a fragment change.
    function freezeCamera() {
        if (config.cam !== "fixed") return;
        if (location.hash.length > 1) return;
        const chosen = userMovedCamera ? currentViewMatrix() : null;
        const matrix = chosen || defaultCameraMatrix();
        if (!matrix) return;
        location.replace("#" + encodeURIComponent(JSON.stringify(matrix)));
    }

    // ---------------------------------------------------------------- hooks

    // Presented frames and the time the viewer's own frame callback spends on
    // the main thread.  The timestamp handed to the callback is the frame's
    // presentation time, so the intervals between them are the frame pacing.
    const nativeRequestFrame = window.requestAnimationFrame.bind(window);
    const nativeCancelFrame = window.cancelAnimationFrame.bind(window);
    const frameCallbacks = new Map();

    window.requestAnimationFrame = function (callback) {
        if (typeof callback !== "function") return nativeRequestFrame(callback);
        let handle = 0;
        const wrapped = (timestamp) => {
            frameCallbacks.delete(handle);
            const samples = state.samples;
            if (!samples) return callback(timestamp);
            const startedAt = performance.now();
            try {
                return callback(timestamp);
            } finally {
                samples.frameTimes.push(timestamp);
                samples.frameBusy.push(performance.now() - startedAt);
            }
        };
        handle = nativeRequestFrame(wrapped);
        frameCallbacks.set(handle, callback);
        return handle;
    };

    window.cancelAnimationFrame = function (handle) {
        frameCallbacks.delete(handle);
        return nativeCancelFrame(handle);
    };

    // The viewer uploads the draw order as one DYNAMIC_DRAW ARRAY_BUFFER per
    // accepted depth commit.  That upload is exactly what a reused order skips,
    // so timing it measures the main thread cost the scheduler removes.
    const ARRAY_BUFFER = 0x8892;
    const DYNAMIC_DRAW = 0x88e8;

    function wrapBufferData(prototype) {
        if (!prototype || !prototype.bufferData) return;
        const original = prototype.bufferData;
        prototype.bufferData = function (target, data, usage) {
            const samples = state.samples;
            if (!samples || target !== ARRAY_BUFFER || usage !== DYNAMIC_DRAW) {
                return original.apply(this, arguments);
            }
            const startedAt = performance.now();
            const result = original.apply(this, arguments);
            samples.uploads.push({
                bytes: data && data.byteLength ? data.byteLength : 0,
                ms: performance.now() - startedAt,
            });
            return result;
        };
    }

    wrapBufferData(window.WebGL2RenderingContext && window.WebGL2RenderingContext.prototype);
    wrapBufferData(window.WebGLRenderingContext && window.WebGLRenderingContext.prototype);

    const nativeGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function () {
        const context = nativeGetContext.apply(this, arguments);
        if (context && context.getParameter && !state.gl) state.gl = context;
        return context;
    };

    // Worker traffic.  A reply that carries depthIndex re-sorted and re-uploaded
    // the draw order, a reusedOrder reply proves the committed order is still
    // valid, and the gap between the request and the reply is how far behind
    // the depth correction is.
    const MAX_LATENCY_MS = 5000;
    // Dynamic depth requests in flight.  The viewer also posts the camera every
    // display frame, which would make a plain "time since the last post" read
    // as a fraction of a millisecond, so replies are matched against the oldest
    // unanswered time request instead.  With coalescing that is an upper bound
    // on the delay of the request the worker just answered, and a persistently
    // growing queue is exactly the backlog this metric is meant to expose.
    const pendingTimePosts = [];
    const MAX_PENDING_POSTS = 32;

    function postKind(message) {
        if (message.time !== undefined) return "time";
        if (message.view) return "view";
        if (message.adaptiveSort) return "policy";
        if (message.mobilegs || message.ply) return "model";
        return "other";
    }

    function messageKind(data) {
        if (data.depthIndex !== undefined) return "sort";
        if (data.reusedOrder) return "reuse";
        if (data.dynamic) return "scene";
        if (data.texdata || data.texdata_sh || data.texdata_dynamic) return "texture";
        if (data.progress || data.error) return "status";
        return "other";
    }

    function observe(data) {
        if (!data || typeof data !== "object") return;
        if (data.dynamic && typeof data.dynamic === "object") {
            state.scene.dynamicEnabled = Boolean(data.dynamic.enabled);
            state.scene.staticCount = Number(data.dynamic.staticCount) || 0;
            state.scene.dynamicCount = Number(data.dynamic.dynamicCount) || 0;
            state.scene.adaptiveRequested = data.dynamic.adaptiveSort === true;
            state.scene.adaptiveBudget = data.dynamic.adaptiveSortBudget;
        }
        if (Number.isFinite(data.vertexCount) && data.vertexCount > 0) {
            state.scene.vertexCount = data.vertexCount;
        }
        const kind = messageKind(data);
        if (kind === "sort" || kind === "reuse") state.sawCommit = true;
        const samples = state.samples;
        if (!samples) return;
        const now = performance.now();
        let latency = -1;
        if ((kind === "sort" || kind === "reuse") && pendingTimePosts.length > 0) {
            latency = now - pendingTimePosts.shift();
            if (!(latency > 0) || latency >= MAX_LATENCY_MS) latency = -1;
        }
        samples.messages.push({
            t: now,
            kind,
            latency,
        });
    }

    const workerPrototype = window.Worker.prototype;
    const nativePostMessage = workerPrototype.postMessage;
    const nativeOnMessage = Object.getOwnPropertyDescriptor(workerPrototype, "onmessage");

    // Bookkeeping for one request the viewer sends to the worker.  It is also
    // the input seam used by smoke.html, which replays traffic without a model.
    function noteRequest(message) {
        if (!message || typeof message !== "object") return;
        const kind = postKind(message);
        if (kind === "time") {
            pendingTimePosts.push(performance.now());
            if (pendingTimePosts.length > MAX_PENDING_POSTS) pendingTimePosts.shift();
        }
        const samples = state.samples;
        if (samples && (kind === "time" || kind === "view")) {
            samples.posts.push({ t: performance.now(), kind });
        }
    }

    workerPrototype.postMessage = function (message) {
        noteRequest(message);
        return nativePostMessage.apply(this, arguments);
    };

    Object.defineProperty(workerPrototype, "onmessage", {
        configurable: true,
        enumerable: nativeOnMessage ? nativeOnMessage.enumerable : true,
        get() {
            return nativeOnMessage.get.call(this);
        },
        set(handler) {
            const worker = this;
            // Assigning the same handler twice must not stack two wrappers.
            if (worker.benchHandler === handler) {
                nativeOnMessage.set.call(worker, worker.benchWrapped);
                return;
            }
            const wrapped = (event) => {
                observe(event.data);
                return handler.call(worker, event);
            };
            worker.benchHandler = handler;
            worker.benchWrapped = wrapped;
            nativeOnMessage.set.call(worker, wrapped);
        },
    });

    // Long tasks are Chromium only; everywhere else the frame intervals already
    // show the same stalls.
    try {
        const observer = new PerformanceObserver((list) => {
            const samples = state.samples;
            if (!samples) return;
            for (const entry of list.getEntries()) samples.longTasks.push({ ms: entry.duration });
        });
        observer.observe({ entryTypes: ["longtask"] });
    } catch (error) {
        // Unsupported entry type.
    }

    function sampleBattery() {
        const manager = state.battery.manager;
        if (!manager) return;
        state.battery.charging = manager.charging;
        const samples = state.samples;
        if (!samples) return;
        samples.battery.series.push({
            t: performance.now(),
            level: manager.level,
            charging: manager.charging,
        });
        samples.battery.levelEnd = manager.level;
    }

    if (navigator.getBattery) {
        navigator.getBattery().then((manager) => {
            state.battery.manager = manager;
            state.battery.charging = manager.charging;
            manager.addEventListener("levelchange", sampleBattery);
            manager.addEventListener("chargingchange", sampleBattery);
            sampleBattery();
        }).catch(() => { });
    }

    document.addEventListener("visibilitychange", () => {
        if (document.hidden && state.samples) state.samples.hidden = true;
    });

    // ------------------------------------------------------------ run control

    function meta() {
        const rate = config.sortFps === 0 ? "uncapped" : config.sortFps + "Hz";
        return {
            label: config.label || (config.adaptive === 1
                ? "adaptive r=" + config.budget + " @" + rate
                : "fixed " + rate),
            model: config.model,
            camMode: config.cam,
            sortFps: config.sortFps,
            adaptive: config.adaptive,
            budget: config.adaptive === 1 ? config.budget : 0,
            smooth: config.smooth,
            loopSeconds: config.loop,
            renderSize: renderSize(),
            dpr: window.devicePixelRatio,
            glRenderer: glRenderer(),
            hardwareConcurrency: navigator.hardwareConcurrency || "",
            deviceMemory: navigator.deviceMemory || "",
            mobile: navigator.userAgentData ? navigator.userAgentData.mobile : null,
            userAgent: navigator.userAgent,
            staticCount: state.scene.staticCount,
            dynamicCount: state.scene.dynamicCount,
            vertexCount: state.scene.vertexCount,
        };
    }

    function beginSamples() {
        const manager = state.battery.manager;
        const samples = {
            startedAt: performance.now(),
            stoppedAt: 0,
            frameTimes: [],
            frameBusy: [],
            uploads: [],
            messages: [],
            posts: [],
            longTasks: [],
            heap: { start: heapBytes(), end: 0 },
            battery: {
                charging: manager ? manager.charging : null,
                levelStart: manager ? manager.level : undefined,
                levelEnd: undefined,
                series: [],
            },
            hidden: document.hidden,
            canvasResized: false,
            renderSizeStart: renderSize(),
        };
        pendingTimePosts.length = 0;
        state.samples = samples;
        sampleBattery();
        state.batteryTimer = setInterval(sampleBattery, 5000);
        return samples;
    }

    function endSamples() {
        const samples = state.samples;
        if (!samples) return null;
        samples.stoppedAt = performance.now();
        samples.heap.end = heapBytes();
        sampleBattery();
        if (state.batteryTimer) {
            clearInterval(state.batteryTimer);
            state.batteryTimer = null;
        }
        samples.canvasResized = renderSize() !== samples.renderSizeStart;
        state.samples = null;
        return samples;
    }

    function persistResults() {
        const store = readStore();
        store.reports = state.results;
        writeStore(store);
    }

    async function runOnce() {
        if (state.busy || !state.ready) return false;
        state.busy = true;
        state.stopRequested = false;
        try {
            freezeCamera();
            if (config.warm > 0) {
                setPhase("warmup " + config.warm + "s");
                await waitFor(config.warm * 1000);
            }
            if (state.stopRequested) return false;
            setPhase("measuring " + config.sec + "s");
            beginSamples();
            updateLive();
            await waitFor(config.sec * 1000);
            const samples = endSamples();
            updateLive();
            if (samples && samples.frameTimes.length > 0) {
                const report = M.buildReport(samples, meta());
                state.results.push(report);
                persistResults();
                renderTable();
                setPhase("done: " + report.framesPerSec + " fps, " + report.reusePct + "% reuse");
                console.log(M.formatTable(state.results));
            } else {
                setPhase("no frames captured");
            }
        } finally {
            state.busy = false;
            state.samples = null;
            state.stopRequested = false;
        }
        if (config.sweep === 1) advanceSweep();
        return true;
    }

    function stopRun() {
        state.stopRequested = true;
    }

    function startRun() {
        runOnce().catch((error) => {
            setPhase("error: " + (error && error.message ? error.message : error));
            console.error(error);
        });
    }

    // ---------------------------------------------------------------- sweep

    function sweepUrl(store, index) {
        const entry = store.entries[index];
        const next = M.applySweepEntry(config, entry);
        next.sweep = 1;
        next.si = index;
        next.auto = 1;
        next.label = entry.label;
        return location.pathname + "?" + M.configToSearch(next) + (store.hash || location.hash);
    }

    function startSweep() {
        const spec = sweepSpec();
        const store = {
            spec,
            entries: M.parseSweep(spec),
            index: 0,
            reports: [],
            hash: "",
        };
        state.results = [];
        writeStore(store);
        setPhase("sweep 1/" + store.entries.length + ": " + store.entries[0].label);
        location.replace(sweepUrl(store, 0));
    }

    function advanceSweep() {
        const store = readStore();
        if (!store.entries || store.entries.length === 0) return;
        const next = store.index + 1;
        if (next >= store.entries.length) {
            setPhase("sweep complete: " + store.entries.length + " configs");
            return;
        }
        // The first step pins the camera; every later step reuses that hash so
        // all configurations are measured from the same viewpoint.
        if (!store.hash) store.hash = location.hash;
        store.index = next;
        writeStore(store);
        setPhase("next " + (next + 1) + "/" + store.entries.length + ": " + store.entries[next].label);
        setTimeout(() => location.replace(sweepUrl(store, next)), 1500);
    }

    // ------------------------------------------------------------------- ui

    const CONTROLS = [
        { key: "cam", label: "camera", values: [["fixed", "frozen"], ["carousel", "orbiting"]] },
        { key: "adaptive", label: "adaptive", values: [["1", "on"], ["0", "off"]] },
        { key: "budget", label: "budget r", values: [["0", "0 exact"], ["0.25", "0.25"], ["0.5", "0.5"], ["1", "1"], ["2", "2"]] },
        { key: "sortFps", label: "sort Hz", values: [["0", "uncapped"], ["10", "10"], ["30", "30"], ["60", "60"]] },
        { key: "sec", label: "measure s", values: [["5", "5"], ["10", "10"], ["20", "20"], ["30", "30"]] },
        { key: "warm", label: "warmup s", values: [["2", "2"], ["3", "3"], ["5", "5"]] },
        { key: "model", label: "model", values: [["coffee.json", "coffee"], ["salmon.json", "salmon"], ["flame_steak.json", "steak"], ["garden.json", "garden (static)"]] },
    ];

    const STRING_CONTROLS = ["model", "cam"];
    const panel = document.getElementById("bench-panel");
    const statusEl = document.getElementById("bench-status");
    const liveEl = document.getElementById("bench-live");
    const tableEl = document.getElementById("bench-table");
    const hintEl = document.getElementById("bench-hint");
    const exportEl = document.getElementById("bench-export");

    let pending = Object.assign({}, config);

    function setPhase(text) {
        state.phase = text;
        if (statusEl) statusEl.textContent = text;
    }

    function sumOf(entries, pick) {
        let sum = 0;
        for (const entry of entries) sum += pick(entry);
        return sum;
    }

    function updateLive() {
        const samples = state.samples;
        if (!samples || !liveEl) return;
        const seconds = Math.max(0.001, (performance.now() - samples.startedAt) / 1000);
        const measured = M.frameIntervals(samples.frameTimes);
        const frames = M.summarize(measured.intervals);
        const uploads = samples.uploads;
        const messages = samples.messages;
        const sorts = messages.filter((entry) => entry.kind === "sort").length;
        const reuses = messages.filter((entry) => entry.kind === "reuse").length;
        const latency = M.summarize(
            messages.map((entry) => entry.latency).filter((value) => value >= 0),
        );
        liveEl.textContent = [
            (samples.frameTimes.length / seconds).toFixed(1) + " fps  " +
                "p50 " + frames.p50.toFixed(1) + "  p95 " + frames.p95.toFixed(1) + " ms  " +
                "jank " + (M.jankShare(measured.intervals) * 100).toFixed(1) + "%",
            "upload " + (uploads.length / seconds).toFixed(1) + "/s  " +
                (sumOf(uploads, (entry) => entry.bytes) / 1048576 / seconds).toFixed(2) + " MB/s  " +
                (sumOf(uploads, (entry) => entry.ms) / seconds).toFixed(2) + " ms/s",
            "sorts " + (sorts / seconds).toFixed(1) + "/s  reuse " +
                (sorts + reuses > 0 ? (100 * reuses / (sorts + reuses)).toFixed(0) : "0") + "%  " +
                "reply p95 " + latency.p95.toFixed(1) + " ms  " +
                "long " + samples.longTasks.length,
        ].join("\n");
    }

    function renderTable() {
        if (!tableEl) return;
        tableEl.textContent = state.results.length > 0
            ? M.formatTable(state.results)
            : "no measurements yet";
        if (tableEl.scrollWidth > 0) tableEl.scrollLeft = tableEl.scrollWidth;
    }

    function setHint() {
        if (!hintEl) return;
        const scene = state.scene;
        const parts = [
            config.model,
            config.cam === "fixed" ? "frozen camera" : "orbiting camera",
            config.adaptive === 1 ? "adaptive r=" + config.budget : "fixed-rate sorting",
            config.sortFps === 0 ? "uncapped sort" : config.sortFps + " Hz sort",
        ];
        if (Number.isFinite(scene.staticCount) && Number.isFinite(scene.dynamicCount)) {
            parts.push(scene.staticCount + " static / " + scene.dynamicCount + " dynamic");
        }
        const warnings = [];
        if (config.sweep === 1) {
            const index = Number(new URLSearchParams(location.search).get("si")) || 0;
            warnings.push("sweep step " + (index + 1));
        }
        if (config.adaptive === 1 && config.cam === "carousel") {
            warnings.push("an orbiting camera changes the view every frame, so there is no order to reuse");
        }
        if (scene.dynamicEnabled === false) warnings.push("this model has no animated Gaussians");
        if (config.sec < 5) warnings.push("short windows are noisy on a phone");
        hintEl.textContent = parts.join("  |  ") +
            (warnings.length > 0 ? "\n" + warnings.join("\n") : "");
    }

    function buildControls() {
        const host = document.getElementById("bench-controls");
        if (!host) return;
        for (const control of CONTROLS) {
            const label = document.createElement("label");
            label.appendChild(document.createTextNode(control.label + " "));
            const select = document.createElement("select");
            for (const [value, text] of control.values) {
                const option = document.createElement("option");
                option.value = value;
                option.textContent = text;
                select.appendChild(option);
            }
            const wanted = String(pending[control.key]);
            select.value = wanted;
            if (select.selectedIndex < 0) {
                // A value that is not in the list (?model=..., ?budget=3) stays
                // selectable, so applying the panel never silently rewrites it.
                const option = document.createElement("option");
                option.value = wanted;
                option.textContent = wanted;
                select.appendChild(option);
                select.value = wanted;
            }
            select.addEventListener("change", () => {
                pending[control.key] = STRING_CONTROLS.includes(control.key)
                    ? select.value
                    : Number(select.value);
                setHint();
            });
            label.appendChild(select);
            host.appendChild(label);
        }
    }

    function applyConfig() {
        // The panel describes a single measurement; sweep state must not leak
        // into a configuration the user picked by hand.
        pending.sweep = 0;
        pending.si = 0;
        pending.auto = 0;
        pending.label = "";
        location.replace(location.pathname + "?" + M.configToSearch(pending) + location.hash);
    }

    function showExport(format) {
        if (!exportEl) return;
        exportEl.value = state.results.length > 0
            ? (format === "json" ? M.toJson(state.results) : M.toCsv(state.results))
            : "no results yet";
        exportEl.focus();
        exportEl.select();
    }

    async function copyExport() {
        if (!exportEl) return;
        if (!exportEl.value) showExport("csv");
        try {
            await navigator.clipboard.writeText(exportEl.value);
            setPhase("copied " + exportEl.value.length + " characters");
        } catch (error) {
            exportEl.focus();
            exportEl.select();
            document.execCommand("copy");
            setPhase("copied (clipboard fallback)");
        }
    }

    function clearResults() {
        state.results = [];
        const store = readStore();
        store.reports = [];
        writeStore(store);
        if (exportEl) exportEl.value = "";
        renderTable();
        setPhase("cleared");
    }

    function checkReady() {
        if (state.ready) return;
        if (!state.sawCommit || !(state.scene.vertexCount > 0)) return;
        state.ready = true;
        clearInterval(readyTimer);
        setPhase("ready");
        setHint();
        document.getElementById("bench-run").disabled = false;
        if (config.auto === 1) startRun();
    }

    const readyTimer = setInterval(checkReady, 250);


    function init() {
        state.results = readStore().reports || [];
        buildControls();
        renderTable();
        setHint();
        setPhase("loading model");
        if (config.panel === 0 && panel) panel.style.display = "none";
        document.getElementById("bench-run").addEventListener("click", startRun);
        document.getElementById("bench-stop").addEventListener("click", stopRun);
        document.getElementById("bench-sweep").addEventListener("click", startSweep);
        document.getElementById("bench-apply").addEventListener("click", applyConfig);
        document.getElementById("bench-clear").addEventListener("click", clearResults);
        document.getElementById("bench-csv").addEventListener("click", () => showExport("csv"));
        document.getElementById("bench-json").addEventListener("click", () => showExport("json"));
        document.getElementById("bench-copy").addEventListener("click", copyExport);
        document.getElementById("bench-toggle").addEventListener("click", (event) => {
            const body = document.getElementById("bench-body");
            const hidden = body.style.display === "none";
            body.style.display = hidden ? "" : "none";
            event.target.textContent = hidden ? "hide" : "show";
        });
        setInterval(updateLive, 250);
    }

    window.BenchView = {
        config,
        metrics: M,
        state,
        // The two inputs the harness normally gets from the wrapped browser
        // objects, exposed so a console or smoke.html can replay traffic.
        observe,
        post: noteRequest,
        start: runOnce,
        stop: stopRun,
        sweep: startSweep,
        clear: clearResults,
        results: () => state.results.slice(),
        table: () => M.formatTable(state.results),
        csv: () => M.toCsv(state.results),
        json: () => M.toJson(state.results),
    };

    init();
})();
