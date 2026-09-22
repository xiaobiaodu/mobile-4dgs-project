// Compares fixed-rate playback sorting with the adaptive scheduler in
// render_shared/main.js.
//
// The decode path is replaced with a synthetic scene of the requested size,
// because decoding a million Gaussians costs minutes and is not what is being
// measured.  Everything that runs per playback frame is the shipped code: the
// worker's runSort, the per-stream counting sort, the stream merge, the gap
// measurement and the reuse decision.  Every configuration replays the same
// timeline, so the columns are directly comparable.
//
//   node tools/adaptive_sort_bench.js [--count=200000] [--dynamic=0.08]
//        [--extent=60] [--speed=2] [--seconds=6] [--seed=7]

"use strict";

const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const MAIN_JS = path.join(__dirname, "..", "render_shared", "main.js");
const DEPTH_KEY_SCALE = 4096;
const DISPLAY_HZ = 60;

// The bench only needs a fixed camera looking down the -z axis; the depth row
// of the resulting view-projection matrix is what the scheduler projects onto.
const VIEW_PROJECTION = [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
];

function parseArgs(argv) {
    const options = {
        count: 200000,
        dynamic: 0.08,
        extent: 60,
        speed: 2,
        seconds: 6,
        seed: 7,
        requestHz: 30,
    };
    for (const argument of argv) {
        const match = argument.match(/^--([a-z]+)=(.+)$/i);
        if (!match) continue;
        const value = Number(match[2]);
        if (Number.isFinite(value)) options[match[1]] = value;
    }
    return options;
}

function mulberry32(seed) {
    let state = seed >>> 0;
    return () => {
        state = (state + 0x6d2b79f5) >>> 0;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// A box of Gaussians with a time-invariant majority and an animated minority,
// the shape a gated mobile 4DGS export has.  Velocities are drawn from a range
// so a few rows move much faster than the rest.
function buildScene({ count, dynamic, extent, speed, seed }) {
    const random = mulberry32(seed);
    const positions = new Float32Array(count * 3);
    const gate = new Uint8Array(count);
    const attributes = new Float32Array(count * 8);
    for (let i = 0; i < count; i++) {
        positions[3 * i + 0] = (random() - 0.5) * extent;
        positions[3 * i + 1] = (random() - 0.5) * extent * 0.35;
        positions[3 * i + 2] = (random() - 0.5) * extent;
        if (random() >= dynamic) continue;
        gate[i] = 1;
        const magnitude = speed * (0.2 + 0.8 * random());
        const theta = random() * Math.PI * 2;
        const phi = Math.acos(2 * random() - 1);
        const dx = Math.sin(phi) * Math.cos(theta);
        const dy = Math.cos(phi) * 0.35;
        const dz = Math.sin(phi) * Math.sin(theta);
        attributes[8 * i + 0] = dx * magnitude;
        attributes[8 * i + 1] = dy * magnitude;
        attributes[8 * i + 2] = dz * magnitude;
        // A small acceleration keeps the bound's quadratic term alive.
        attributes[8 * i + 3] = dx * magnitude * 0.3;
        attributes[8 * i + 4] = dy * magnitude * 0.3;
        attributes[8 * i + 5] = dz * magnitude * 0.3;
        attributes[8 * i + 6] = random();
        attributes[8 * i + 7] = Math.log(0.15 + 0.5 * random());
    }
    return { count, positions, gate, attributes };
}

// Loads the worker source with two hooks: the sort is timed, and the decoded
// model is replaced by a synthetic scene.
function loadWorker() {
    const source = fs.readFileSync(MAIN_JS, "utf8");
    const start = source.indexOf("function createWorker(self) {");
    const end = source.indexOf("\nconst vertexShaderSource");
    if (start < 0 || end <= start) throw new Error("createWorker not found");
    let body = source.slice(start, end).trimEnd();

    const sortCall = "            runSort(lastView);";
    if (!body.includes(sortCall)) throw new Error("the throttled sort call changed shape");
    body = body.replace(sortCall, "            globalThis.__timedSort(lastView);");

    const hook = [
        "    globalThis.__timedSort = (view) => {",
        "        const started = globalThis.__now();",
        "        try {",
        "            runSort(view);",
        "        } finally {",
        "            globalThis.__sortMs += globalThis.__now() - started;",
        "            globalThis.__sortCalls += 1;",
        "        }",
        "    };",
        "    globalThis.__installScene = (scene) => {",
        "        const count = scene.count;",
        "        vertexCount = count;",
        "        buffer = new ArrayBuffer(count * 32);",
        "        const packed = new Float32Array(buffer);",
        "        for (let i = 0; i < count; i++) {",
        "            packed[8 * i + 0] = scene.positions[3 * i + 0];",
        "            packed[8 * i + 1] = scene.positions[3 * i + 1];",
        "            packed[8 * i + 2] = scene.positions[3 * i + 2];",
        "            packed[8 * i + 3] = scene.scale;",
        "            packed[8 * i + 4] = scene.scale;",
        "            packed[8 * i + 5] = scene.scale;",
        "        }",
        "        covarianceBuffer = new Float32Array(count * 6);",
        "        for (let i = 0; i < count; i++) {",
        "            covarianceBuffer[6 * i + 0] = scene.scale * scene.scale;",
        "            covarianceBuffer[6 * i + 3] = scene.scale * scene.scale;",
        "            covarianceBuffer[6 * i + 5] = scene.scale * scene.scale;",
        "        }",
        "        shBuffer = new ArrayBuffer(count * 48);",
        "        dynamicBuffer = new Float32Array(count * 8);",
        "        dynamicBuffer.set(scene.attributes);",
        "        dynamicGate = scene.gate;",
        "        dynamicTime = 0;",
        "        lastVertexCount = 0;",
        "        buildGaussianSplit(dynamicGate, true, count);",
        "        materializeStaticTemporalRows(dynamicGate, count);",
        "        refreshAdaptiveMotionBounds();",
        "        refreshAdaptiveScale();",
        "    };",
    ].join("\n");
    const messageHook = "    self.onmessage = async (e) => {";
    if (!body.includes(messageHook)) throw new Error("the worker message hook changed shape");
    body = body.replace(messageHook, hook + "\n" + messageHook);

    const messages = [];
    const sandbox = {
        console: { log() {}, warn() {}, error() {}, time() {}, timeEnd() {} },
        TextDecoder,
        setTimeout,
        clearTimeout,
        __now: () => Number(process.hrtime.bigint()) / 1e6,
        __sortMs: 0,
        __sortCalls: 0,
        __collect: (message) => messages.push(message),
    };
    const context = vm.createContext(sandbox);
    vm.runInContext(
        "globalThis.self = globalThis;\n" +
        "globalThis.TMC3_URL = 'about:blank';\n" +
        "globalThis.importScripts = function () {};\n" +
        "globalThis.postMessage = function (message) { globalThis.__collect(message); };\n",
        context,
    );
    vm.runInContext("(" + body + ")(self);", context);
    return { context, messages };
}

function createHarness() {
    const { context, messages } = loadWorker();
    const onMessage = vm.runInContext("self.onmessage", context);
    const installScene = vm.runInContext("globalThis.__installScene", context);

    const counters = () => vm.runInContext(
        "({ ms: globalThis.__sortMs, calls: globalThis.__sortCalls })",
        context,
    );
    const resetCounters = () => vm.runInContext(
        "globalThis.__sortMs = 0; globalThis.__sortCalls = 0;",
        context,
    );

    async function pump() {
        for (let idle = 0; idle < 3;) {
            const before = messages.length;
            await new Promise((resolve) => setTimeout(resolve, 0));
            idle = messages.length === before ? idle + 1 : 0;
        }
    }

    return {
        messages,
        counters,
        resetCounters,
        installScene,
        async send(data) {
            await onMessage({ data });
            await pump();
        },
    };
}

function depthKeysAt(scene, time) {
    const keys = new Float64Array(scene.count);
    const { positions, attributes } = scene;
    for (let i = 0; i < scene.count; i++) {
        const dt = time - attributes[8 * i + 6];
        const half = 0.5 * dt * dt;
        const x = positions[3 * i + 0] +
            attributes[8 * i + 0] * dt + attributes[8 * i + 3] * half;
        const y = positions[3 * i + 1] +
            attributes[8 * i + 1] * dt + attributes[8 * i + 4] * half;
        const z = positions[3 * i + 2] +
            attributes[8 * i + 2] * dt + attributes[8 * i + 5] * half;
        keys[i] = (VIEW_PROJECTION[2] * x + VIEW_PROJECTION[6] * y +
            VIEW_PROJECTION[10] * z) * DEPTH_KEY_SCALE | 0;
    }
    return keys;
}

// How far the presented order is from the exact depth order at this instant:
// the share of neighbouring pairs that are out of order, and the largest depth
// violation any of them has, measured in Gaussian radii.
function orderError(order, keys, footprintKeys) {
    let inversions = 0;
    let visible = 0;
    let worst = 0;
    for (let k = 1; k < order.length; k++) {
        const previous = keys[order[k - 1]];
        const current = keys[order[k]];
        if (current >= previous) continue;
        inversions++;
        if (previous - current > 0.25 * footprintKeys) visible++;
        if (previous - current > worst) worst = previous - current;
    }
    return {
        inversions,
        fraction: order.length > 1 ? inversions / (order.length - 1) : 0,
        visibleFraction: order.length > 1 ? visible / (order.length - 1) : 0,
        worstRadii: footprintKeys > 0 ? worst / footprintKeys : 0,
    };
}

async function runConfig(harness, scene, options, config) {
    const { messages } = harness;
    const animated = scene.gate.reduce((sum, value) => sum + value, 0);
    messages.length = 0;
    await harness.send({
        adaptiveSort: { enabled: config.budget !== null, budget: config.budget ?? 0 },
    });
    await harness.send({ view: VIEW_PROJECTION });
    await harness.send({ time: 0, dynamicRequestId: 0 });
    harness.resetCounters();
    messages.length = 0;

    const footprintKeys = Math.exp(-2) * DEPTH_KEY_SCALE;
    const frames = Math.round(options.seconds * DISPLAY_HZ);
    const requestInterval = config.requestHz > 0 ? 1 / config.requestHz : 0;
    const sampleEvery = 5;
    let nextRequestAt = 0;
    let requestId = 1;
    let requests = 0;
    let reuses = 0;
    let sorts = 0;
    let gapMin;
    let depthBytes = 0;
    let order = null;
    let commitTime = 0;
    let errorSum = 0;
    let errorMax = 0;
    let visibleSum = 0;
    let visibleMax = 0;
    let worstRadii = 0;
    let ageSum = 0;
    let samples = 0;

    for (let frame = 0; frame < frames; frame++) {
        const playbackSeconds = frame / DISPLAY_HZ;
        const time = frame / (frames - 1);
        if (playbackSeconds >= nextRequestAt) {
            nextRequestAt = playbackSeconds + requestInterval;
            requests++;
            const before = messages.length;
            await harness.send({ time, dynamicRequestId: requestId++ });
            for (let k = before; k < messages.length; k++) {
                const message = messages[k];
                if (message.reusedOrder) {
                    reuses++;
                } else if (message.depthIndex) {
                    order = message.depthIndex;
                    commitTime = time;
                    sorts++;
                    gapMin = message.sortStats ? message.sortStats.adaptiveGapMin : undefined;
                    depthBytes += message.depthIndex.byteLength;
                }
            }
        }
        if (order && frame % sampleEvery === 0) {
            const measured = orderError(order, depthKeysAt(scene, time), footprintKeys);
            errorSum += measured.fraction;
            errorMax = Math.max(errorMax, measured.fraction);
            visibleSum += measured.visibleFraction;
            visibleMax = Math.max(visibleMax, measured.visibleFraction);
            worstRadii = Math.max(worstRadii, measured.worstRadii);
            ageSum += time - commitTime;
            samples++;
        }
    }
    const counters = harness.counters();
    return {
        name: config.name,
        requests,
        sorts,
        reuses,
        sortMs: counters.ms,
        perSecondMs: counters.ms / options.seconds,
        msPerSort: sorts > 0 ? counters.ms / sorts : 0,
        depthBytes,
        perSecondBytes: depthBytes / options.seconds,
        meanInversions: samples > 0 ? errorSum / samples : 0,
        maxInversions: errorMax,
        meanVisible: samples > 0 ? visibleSum / samples : 0,
        maxVisible: visibleMax,
        worstRadii,
        meanAge: samples > 0 ? ageSum / samples : 0,
        gapMin,
        animated,
    };
}

function formatRow(columns, widths) {
    return columns.map((value, index) =>
        String(value).padStart(widths[index])).join("  ");
}

async function main() {
    const options = parseArgs(process.argv.slice(2));
    const scene = buildScene(options);
    scene.scale = Math.exp(-2);
    const harness = createHarness();
    harness.installScene(scene);
    await harness.send({ view: VIEW_PROJECTION });
    await harness.send({ time: 0, dynamicRequestId: 0 });
    const animated = scene.gate.reduce((sum, value) => sum + value, 0);

    const keys = depthKeysAt(scene, 0);
    const order = harness.messages.filter((message) => message.depthIndex).pop().depthIndex;
    let movable = 0;
    let tight = 0;
    for (let k = 1; k < order.length; k++) {
        const previous = order[k - 1];
        const current = order[k];
        if (!scene.gate[previous] && !scene.gate[current]) continue;
        movable++;
        const gap = keys[current] - keys[previous];
        if (gap <= 2) tight++;
    }
    const keysPerSecond = options.speed * DEPTH_KEY_SCALE *
        Math.hypot(VIEW_PROJECTION[2], VIEW_PROJECTION[6], VIEW_PROJECTION[10]);
    const movementPerSort = keysPerSecond * (1 / options.requestHz) / options.seconds;
    let keyMin = Infinity;
    let keyMax = -Infinity;
    for (let i = 0; i < keys.length; i++) {
        if (keys[i] < keyMin) keyMin = keys[i];
        if (keys[i] > keyMax) keyMax = keys[i];
    }

    process.stdout.write([
        `scene      ${options.count.toLocaleString()} Gaussians, ` +
        `${animated.toLocaleString()} animated (${(100 * animated / options.count).toFixed(1)}%)`,
        `footprint  ${(Math.exp(-2)).toFixed(4)} world units = ` +
        `${(Math.exp(-2) * DEPTH_KEY_SCALE).toFixed(0)} depth keys`,
        `playback   ${options.seconds}s clip over ${options.seconds}s wall time, ` +
        `${DISPLAY_HZ} Hz display, requests at ${options.requestHz} Hz`,
        `motion     about ${movementPerSort.toFixed(1)} depth keys between two requests`,
        `gaps       ${(100 * tight / movable).toFixed(1)}% of the ` +
        `${movable.toLocaleString()} movable adjacent pairs are within 2 depth keys ` +
        `(the 16 bit counting sort itself resolves about ` +
        `${((keyMax - keyMin) / 65536).toFixed(1)} keys per bucket)`,
        "",
    ].join("\n"));

    const configs = [
        { name: "fixed 30 Hz", budget: null, requestHz: options.requestHz },
        { name: "fixed 10 Hz", budget: null, requestHz: 10 },
        { name: "fixed 3 Hz", budget: null, requestHz: 3 },
        { name: "adaptive r=0 @30", budget: 0, requestHz: options.requestHz },
        { name: "adaptive r=0.25 @30", budget: 0.25, requestHz: options.requestHz },
        { name: "adaptive r=0.5 @30", budget: 0.5, requestHz: options.requestHz },
        { name: "adaptive r=1 @30", budget: 1, requestHz: options.requestHz },
        { name: "adaptive r=0.5 @60", budget: 0.5, requestHz: 60 },
    ];
    const results = [];
    for (const config of configs) {
        results.push(await runConfig(harness, scene, options, config));
    }

    const header = [
        "config", "req", "sorts", "reuse", "ms", "ms/s", "ms/sort",
        "MB/s", "inv%", "vis%", "visMax%", "drift(r)", "age",
    ];
    const widths = [22, 5, 7, 7, 7, 7, 8, 7, 7, 7, 8, 8, 7];
    process.stdout.write(formatRow(header, widths) + "\n");
    const rows = results.map((result) => [
        result.name,
        result.requests,
        result.sorts,
        result.reuses,
        result.sortMs.toFixed(0),
        result.perSecondMs.toFixed(1),
        result.msPerSort.toFixed(2),
        (result.perSecondBytes / 1e6).toFixed(1),
        (100 * result.meanInversions).toFixed(2),
        (100 * result.meanVisible).toFixed(2),
        (100 * result.maxVisible).toFixed(2),
        result.worstRadii.toFixed(2),
        result.meanAge.toFixed(3),
    ]);
    for (const row of rows) process.stdout.write(formatRow(row, widths) + "\n");

    const baseline = results[0];
    const adaptive = results.find((result) => result.name.includes("r=0.5"));
    const reduction = (key) =>
        (100 * (1 - adaptive[key] / baseline[key])).toFixed(0);
    process.stdout.write("\n" +
        `adaptive r=0.5 vs fixed ${options.requestHz} Hz: ` +
        `${reduction("sorts")}% fewer sorts, ` +
        `${reduction("sortMs")}% less worker sort time, ` +
        `${reduction("depthBytes")}% fewer index bytes\n` +
        `strict bound saw a tightest movable gap of ` +
        `${results.find((result) => result.name.includes("r=0 @")).gapMin} keys\n`);
}

main().catch((error) => {
    process.stderr.write(String(error && error.stack ? error.stack : error) + "\n");
    process.exitCode = 1;
});
