// Ablation for the depth-order certificate (DOC) of Sec. IV-D.
//
// The shipped worker code is loaded in a VM with a synthetic 4D scene, so the
// measured work is exactly what a playback frame runs: the per-stream
// projection, the counting sort, the stream merge, the gap measurement and the
// reuse decision.  Every policy replays the same timeline, hence the columns
// are directly comparable.
//
//   node tools/doc_ablation_bench.js [--count=200000] [--dynamic=0.08]
//        [--extent=60] [--speed=2] [--seconds=6] [--seed=7] [--requestHz=30]
//        [--mode=box|lattice]
//
// `box` is a dense scene with a time-invariant majority, the shape a real
// capture has: the tightest movable depth gap is small and the certificate is
// hard to satisfy.  `lattice` spreads the rows over a much wider depth range,
// so kappa is small and the budget branch of Eq. (41) admits nearly every
// frame -- a tolerance-limited contrast to the certificate-limited box.  (A
// lattice whose layers are more than one bin apart needs fewer than B = 2^16
// rows, so pass a smaller --count to make the certified branch itself fire.)
//
// Columns
//   reuse%   playback requests answered from the committed draw list
//   ms/s     worker time spent in projection + sort + merge per second
//   ms/req   the same, per request that had to be served
//   MB/s     draw-list traffic of the index buffer per second
//   red%     reduction of ms/s and MB/s against the two-stream baseline
//   exact    reused commits whose list is bit-identical to a fresh sort of the
//            shipped pipeline: the 16 bit counting sort of Eq. (32) per stream,
//            then the exact-key merge of Eq. (33)
//   inv%     mean share of adjacent pairs out of depth order while reusing
//   drift    largest depth violation of the presented order, in radii
//   mis%     share of rows out of place against an uncached frame
//   shift    worst displacement of a row against an uncached frame, in rows
//
// inv% and drift are measured against the key order of Eq. (31).  The counting
// sort of Eq. (32) reproduces it exactly, because q_i is the bucket index, so a
// bucket holds a single value of q and the stable pass is argsort(q_i, i); the
// baselines therefore carry no quantisation floor.  mis% and shift are measured
// against an uncached frame and are therefore the cost of the scheduling
// decision alone.

"use strict";

const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const MAIN_JS = path.join(__dirname, "..", "render_shared", "main.js");
// Eq. (31): B = 2^16 bins, so kappa = (B - 1) / (z_max - z_min).
const DEPTH_BUCKETS = 256 * 256;
const DEPTH_KEY_MAX = DEPTH_BUCKETS - 1;
const LATTICE_SPACING = 4 * Math.exp(-2);   // 4 mean Gaussian radii per layer
const DISPLAY_HZ = 60;
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
        mode: "box",
    };
    for (const argument of argv) {
        const text = argument.match(/^--(mode)=([a-z]+)$/i);
        if (text) {
            options[text[1]] = text[2];
            continue;
        }
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

// A box with a time-invariant majority and an animated minority, the shape a
// gated mobile 4DGS export has.  The same draw of velocities is used for the
// gated scene and for the monolithic reference, so the two differ only in the
// partition the renderer is given.
function buildScenes({ count, dynamic, extent, speed, seed, mode }) {
    const random = mulberry32(seed);
    const positions = new Float32Array(count * 3);
    const attributes = new Float32Array(count * 8);
    const gate = new Uint8Array(count);
    // Depth lattice: one Gaussian per layer, a few radii apart, shuffled so the
    // correct draw order is not the row order.  Adjacent gaps are then
    // thousands of keys wide instead of a handful, which is the regime the
    // certified branch needs.
    const lattice = new Float64Array(count);
    if (mode === "lattice") {
        for (let i = 0; i < count; i++) lattice[i] = i;
        for (let i = count - 1; i > 0; i--) {
            const j = Math.floor(random() * (i + 1));
            const swap = lattice[i];
            lattice[i] = lattice[j];
            lattice[j] = swap;
        }
    }
    for (let i = 0; i < count; i++) {
        positions[3 * i + 0] = (random() - 0.5) * extent;
        positions[3 * i + 1] = (random() - 0.5) * extent * 0.35;
        positions[3 * i + 2] = mode === "lattice"
            ? (lattice[i] - (count - 1) / 2) * LATTICE_SPACING
            : (random() - 0.5) * extent;
        const animated = random() < dynamic;
        const magnitude = speed * (0.2 + 0.8 * random());
        const theta = random() * Math.PI * 2;
        const phi = Math.acos(2 * random() - 1);
        const dx = Math.sin(phi) * Math.cos(theta);
        const dy = Math.cos(phi) * 0.35;
        const dz = Math.sin(phi) * Math.sin(theta);
        attributes[8 * i + 0] = animated ? dx * magnitude : 0;
        attributes[8 * i + 1] = animated ? dy * magnitude : 0;
        attributes[8 * i + 2] = animated ? dz * magnitude : 0;
        attributes[8 * i + 3] = animated ? dx * magnitude * 0.3 : 0;
        attributes[8 * i + 4] = animated ? dy * magnitude * 0.3 : 0;
        attributes[8 * i + 5] = animated ? dz * magnitude * 0.3 : 0;
        attributes[8 * i + 6] = random();
        attributes[8 * i + 7] = Math.log(0.15 + 0.5 * random());
        gate[i] = animated ? 1 : 0;
    }
    const monolithicGate = new Uint8Array(count).fill(1);
    return {
        count,
        positions,
        gate,
        monolithicGate,
        attributes,
    };
}

// Loads the worker source with the sort timed and the decoder replaced.
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
        // A new model is a new depth mapping, exactly like a real decode:
        // resetSortState clears the buffers and marks the interval dirty.
        "        resetSortState();",
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
        "    // What the scheduler knows when it commits: the movement bound of the",
        "    // committed list, the tightest movable gap it measured, and the drift",
        "    // the tolerance policy accepts.  The three are comparable, in keys.",
        "    globalThis.__docDiag = (time) => {",
        "        const movement = predictedDepthMovement(time);",
        "        return {",
        "            movement,",
        "            rate: predictedDepthRate(time),",
        "            gapMin: depthGapMin,",
        "            tolerance: adaptiveTolerance(),",
        "            certified: 2 * movement < depthGapMin,",
        "            budget: movement <= adaptiveTolerance(),",
        "            committedDepthTime,",
        "        };",
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

    async function pump() {
        for (let idle = 0; idle < 3;) {
            const before = messages.length;
            await new Promise((resolve) => setTimeout(resolve, 0));
            idle = messages.length === before ? idle + 1 : 0;
        }
    }

    return {
        messages,
        installScene,
        counters: () => vm.runInContext(
            "({ ms: globalThis.__sortMs, calls: globalThis.__sortCalls })",
            context,
        ),
        resetCounters: () => vm.runInContext(
            "globalThis.__sortMs = 0; globalThis.__sortCalls = 0;",
            context,
        ),
        diag: (time) => vm.runInContext("globalThis.__docDiag", context)(time),
        async send(data) {
            await onMessage({ data });
            await pump();
        },
    };
}

// Unit depth axis of Eq. (31): the view-projection row the quantiser reads.
// The benchmark view is the identity, so n = (0, 0, 1).
function depthAxisOf() {
    const norm = Math.hypot(VIEW_PROJECTION[2], VIEW_PROJECTION[6], VIEW_PROJECTION[10]);
    if (!(norm > 0)) return [0, 0, 0];
    return [
        VIEW_PROJECTION[2] / norm,
        VIEW_PROJECTION[6] / norm,
        VIEW_PROJECTION[10] / norm,
    ];
}

// [z_min, z_max] of Eq. (31) and kappa = (B - 1) / (z_max - z_min), mirrored
// from refreshDepthInterval: measured over the decoded rows of both streams,
// then widened by the worst-case travel of the animated subset over the clip so
// that an animated row can never leave the interval and clip onto a neighbour.
function depthIntervalOf(scene) {
    const { positions, attributes, gate } = scene;
    const count = scene.count;
    const axis = depthAxisOf();
    let low = Infinity;
    let high = -Infinity;
    for (let i = 0; i < count; i++) {
        const z = axis[0] * positions[3 * i + 0] +
            axis[1] * positions[3 * i + 1] + axis[2] * positions[3 * i + 2];
        if (z < low) low = z;
        if (z > high) high = z;
    }
    if (!(high > low)) {
        const single = Number.isFinite(low) ? low : 0;
        return { min: single, max: single, scale: 0 };
    }
    let speedBound = 0;
    let accelBound = 0;
    let timeMin = Infinity;
    let timeMax = -Infinity;
    for (let i = 0; i < count; i++) {
        if (!gate[i]) continue;
        const speed = Math.hypot(
            attributes[8 * i + 0], attributes[8 * i + 1], attributes[8 * i + 2]);
        if (speed > speedBound) speedBound = speed;
        const accel = Math.hypot(
            attributes[8 * i + 3], attributes[8 * i + 4], attributes[8 * i + 5]);
        if (accel > accelBound) accelBound = accel;
        const canonical = attributes[8 * i + 6];
        if (canonical < timeMin) timeMin = canonical;
        if (canonical > timeMax) timeMax = canonical;
    }
    if (!Number.isFinite(timeMin)) {
        timeMin = 0;
        timeMax = 0;
    }
    const tauMax = Math.max(0, timeMax, 1 - timeMin);
    const travel = speedBound * tauMax + 0.5 * accelBound * tauMax * tauMax;
    const min = low - travel;
    const max = high + travel;
    return { min, max, scale: DEPTH_KEY_MAX / (max - min) };
}

// Quantised depth of Eq. (31), q_i(t) = clip(floor(kappa * (z_i - z_min)), 0, B-1),
// for every row: displacement v*dt + 0.5*a*dt^2 past the canonical position,
// projected onto the unit depth axis.  This is exactly what the worker's
// quantizeKey produces, so the replica and the shipped sort share one key space.
function keysAt(scene, time) {
    const keys = new Int32Array(scene.count);
    const { positions, attributes, gate } = scene;
    const interval = depthIntervalOf(scene);
    const axis = depthAxisOf();
    for (let i = 0; i < scene.count; i++) {
        let x = positions[3 * i + 0];
        let y = positions[3 * i + 1];
        let z = positions[3 * i + 2];
        if (gate[i]) {
            const dt = time - attributes[8 * i + 6];
            const half = 0.5 * dt * dt;
            x += attributes[8 * i + 0] * dt + attributes[8 * i + 3] * half;
            y += attributes[8 * i + 1] * dt + attributes[8 * i + 4] * half;
            z += attributes[8 * i + 2] * dt + attributes[8 * i + 5] * half;
        }
        const q = (interval.scale * (
            axis[0] * x + axis[1] * y + axis[2] * z - interval.min)) | 0;
        keys[i] = q < 0 ? 0 : (q > DEPTH_KEY_MAX ? DEPTH_KEY_MAX : q);
    }
    return keys;
}

// Eq. (32)/(33): ascending q_i, ties broken by row id.  In the 2^16 key space
// every bucket holds exactly one value of q, so the worker's per-stream
// counting sort followed by the exact-key merge reproduces this order row for
// row; it is both the ideal order and the order a fresh frame ships.
function keyOrder(keys) {
    const count = keys.length;
    const order = new Int32Array(count);
    for (let i = 0; i < count; i++) order[i] = i;
    const sorted = Array.from(order).sort((a, b) => keys[a] - keys[b] || a - b);
    return Int32Array.from(sorted);
}

// The order a fresh playback frame produces: project both streams, count-sort
// each on q, then merge on (q, i) -- which is argsort(q_i, i) of Eq. (33).
function referenceOrder(scene, time) {
    return keyOrder(keysAt(scene, time));
}

// Position of every id inside a draw list, to measure how far a row moved.
function positionsOf(order, count) {
    const position = new Int32Array(count).fill(-1);
    for (let k = 0; k < order.length; k++) position[order[k]] = k;
    return position;
}

function compareOrders(presented, reference) {
    if (presented.length !== reference.length) {
        return { identical: false, mismatches: presented.length, maxShift: Infinity };
    }
    const position = positionsOf(reference, reference.length);
    let mismatches = 0;
    let maxShift = 0;
    for (let k = 0; k < presented.length; k++) {
        if (presented[k] === reference[k]) continue;
        mismatches++;
        const shift = Math.abs(k - position[presented[k]]);
        if (shift > maxShift) maxShift = shift;
    }
    return { identical: mismatches === 0, mismatches, maxShift };
}

function orderDrift(order, keys, footprintKeys) {
    let inversions = 0;
    let worst = 0;
    for (let k = 1; k < order.length; k++) {
        const previous = keys[order[k - 1]];
        const current = keys[order[k]];
        if (current >= previous) continue;
        inversions++;
        if (previous - current > worst) worst = previous - current;
    }
    return {
        fraction: order.length > 1 ? inversions / (order.length - 1) : 0,
        worstRadii: footprintKeys > 0 ? worst / footprintKeys : 0,
    };
}

async function runPolicy(harness, scene, options, policy) {
    const { messages } = harness;
    messages.length = 0;
    await harness.send({
        adaptiveSort: { enabled: policy.budget !== null, budget: policy.budget ?? 0 },
    });
    await harness.send({ minPixelRadius: 0 });
    await harness.send({
        view: VIEW_PROJECTION,
        focal: 800,
        viewDepthRow: [0, 0, 1, 0],
    });
    await harness.send({ time: 0, dynamicRequestId: 0 });
    harness.resetCounters();
    messages.length = 0;

    const footprintKeys = scene.scale * depthIntervalOf(scene).scale;
    const frames = Math.round(options.seconds * DISPLAY_HZ);
    const requestInterval = policy.requestHz > 0 ? 1 / policy.requestHz : Infinity;
    // The scheduler only decides when a request arrives, so drift and the cost
    // against an uncached frame are measured there.  About a dozen requests per
    // run, because every sample costs a fresh sort of the whole model.
    const sampleEvery = Math.max(
        1,
        Math.round(Math.max(1, options.seconds * policy.requestHz) / 12),
    );
    let requestSamples = 0;
    let nextRequestAt = 0;
    let requestId = 1;
    let requests = 0;
    let reuses = 0;
    let sorts = 0;
    let indexBytes = 0;
    let order = null;
    let driftSum = 0;
    let driftWorst = 0;
    let driftSamples = 0;
    let exactSamples = 0;
    let exactMismatches = 0;
    let exactMaxShift = 0;
    let reusedSampled = 0;
    let gapMin;
    let toleranceKeys = 0;
    let rateKeys = 0;
    let reuseHeadroomMin = Number.POSITIVE_INFINITY;
    let reuseHeadroomMax = Number.NEGATIVE_INFINITY;
    let refusalHeadroomMax = Number.NEGATIVE_INFINITY;
    let boundAtReuseMax = 0;
    let boundAtRefusalMin = Number.POSITIVE_INFINITY;
    let freshSampled = 0;
    let freshMismatches = 0;
    let freshMaxShift = 0;
    let bucketMismatches = 0;
    let bucketMaxShift = 0;
    let idealMismatches = 0;
    let idealMaxShift = 0;
    let deviationSum = 0;
    let deviationShift = 0;
    let deviationFrames = 0;
    let certifiedReuses = 0;
    let trivialReuses = 0;
    let certifiedSampled = 0;
    let certifiedExact = 0;
    let gapMinLow = Number.POSITIVE_INFINITY;
    let gapMinHigh = Number.NEGATIVE_INFINITY;
    let gapMinCommits = 0;
    let gapMinNonPositive = 0;
    // Every reuse is checked until the cap, not just the frames on the drift
    // grid: the certified branch fires rarely, so its evidence is its own.
    const exactCap = 25;
    const freshCap = 5;

    for (let frame = 0; frame < frames; frame++) {
        const playbackSeconds = frame / DISPLAY_HZ;
        const time = frames > 1 ? frame / (frames - 1) : 0;
        if (playbackSeconds >= nextRequestAt) {
            nextRequestAt = playbackSeconds + requestInterval;
            requests++;
            let reused = false;
            // Read the scheduler state before the request is served: a refused
            // request re-sorts and re-commits, which would overwrite the very
            // numbers the decision was made from.
            const diag = harness.diag(time);
            const movement = diag.movement;
            const before = messages.length;
            await harness.send({ time, dynamicRequestId: requestId++ });
            for (let k = before; k < messages.length; k++) {
                const message = messages[k];
                if (message.reusedOrder) {
                    reuses++;
                    reused = true;
                } else if (message.depthIndex) {
                    order = message.depthIndex;
                    sorts++;
                    indexBytes += message.depthIndex.byteLength;
                    if (message.sortStats) gapMin = message.sortStats.adaptiveGapMin;
                    // The quantity the certified test compares its movement
                    // bound against, over every commit of the run.
                    if (Number.isFinite(gapMin)) {
                        gapMinCommits++;
                        if (gapMin <= 0) gapMinNonPositive++;
                        gapMinLow = Math.min(gapMinLow, gapMin);
                        gapMinHigh = Math.max(gapMinHigh, gapMin);
                    }
                    // The replica is only usable as a reference if it
                    // reproduces the sort the worker itself posts.
                    if (freshSampled < freshCap) {
                        freshSampled++;
                        const fresh = compareOrders(
                            Int32Array.from(message.depthIndex),
                            referenceOrder(scene, time),
                        );
                        if (!fresh.identical) {
                            freshMismatches += fresh.mismatches;
                            freshMaxShift = Math.max(freshMaxShift, fresh.maxShift);
                        }
                        const ideal = compareOrders(
                            Int32Array.from(message.depthIndex),
                            keyOrder(keysAt(scene, time)),
                        );
                        bucketMismatches += ideal.mismatches;
                        bucketMaxShift = Math.max(bucketMaxShift, ideal.maxShift);
                    }
                }
            }
            // The bound this request was decided with, and whether the reuse was
            // certified by a non-zero movement or taken at the committed time.
            if (reused) {
                if (movement > 0) certifiedReuses++;
                else trivialReuses++;
            }
            // The presented list is the committed one; a fresh sort of the same
            // time is what the scheduler claims it can skip.
            if (reused && order && reusedSampled < exactCap) {
                reusedSampled++;
                const presented = Int32Array.from(order);
                const shipped = compareOrders(presented, referenceOrder(scene, time));
                if (shipped.identical) {
                    exactSamples++;
                    if (movement > 0) certifiedExact++;
                } else {
                    exactMismatches += shipped.mismatches;
                    exactMaxShift = Math.max(exactMaxShift, shipped.maxShift);
                }
                if (movement > 0) certifiedSampled++;
                const ideal = compareOrders(presented, keyOrder(keysAt(scene, time)));
                idealMismatches += ideal.mismatches;
                idealMaxShift = Math.max(idealMaxShift, ideal.maxShift);
            }
            if (Number.isFinite(movement)) {
                rateKeys = Math.max(rateKeys, diag.rate);
                toleranceKeys = Math.max(toleranceKeys, diag.tolerance);
                if (diag.gapMin > 0) {
                    const bound = Math.ceil(2 * movement);
                    const headroom = diag.gapMin - bound;
                    if (reused) {
                        reuseHeadroomMin = Math.min(reuseHeadroomMin, headroom);
                        reuseHeadroomMax = Math.max(reuseHeadroomMax, headroom);
                        boundAtReuseMax = Math.max(boundAtReuseMax, bound);
                    } else {
                        boundAtRefusalMin = Math.min(boundAtRefusalMin, bound);
                        if (policy.budget === 0) {
                            refusalHeadroomMax = Math.max(refusalHeadroomMax, headroom);
                        }
                    }
                }
            }
            requestSamples++;
            if (order && requestSamples % sampleEvery === 0) {
                const keys = keysAt(scene, time);
                const drift = orderDrift(order, keys, footprintKeys);
                driftSum += drift.fraction;
                driftWorst = Math.max(driftWorst, drift.worstRadii);
                driftSamples++;
                // What the decision costs against an uncached frame: how many
                // rows sit in the wrong place and how far the worst one moved.
                // A policy that always re-sorts has to score zero here.
                const uncached = referenceOrder(scene, time);
                const deviation = compareOrders(order, uncached);
                deviationFrames++;
                deviationSum += deviation.mismatches;
                deviationShift = Math.max(deviationShift, deviation.maxShift);
            }
        }
    }

    const counters = harness.counters();
    return {
        name: policy.name,
        requests,
        sorts,
        reuses,
        reuseFraction: requests > 0 ? reuses / requests : 0,
        sortMs: counters.ms,
        perSecondMs: counters.ms / options.seconds,
        msPerRequest: requests > 0 ? counters.ms / requests : 0,
        perSecondBytes: indexBytes / options.seconds,
        meanInversions: driftSamples > 0 ? driftSum / driftSamples : 0,
        worstRadii: driftWorst,
        reusedSampled,
        exactSamples,
        exactMismatches,
        exactMaxShift,
        gapMin,
        toleranceKeys,
        rateKeys,
        reuseHeadroomMin,
        reuseHeadroomMax,
        refusalHeadroomMax,
        boundAtReuseMax,
        boundAtRefusalMin,
        freshSampled,
        freshMismatches,
        freshMaxShift,
        bucketMismatches,
        bucketMaxShift,
        idealMismatches,
        idealMaxShift,
        deviationSum,
        deviationShift,
        deviationFrames,
        certifiedReuses,
        trivialReuses,
        certifiedSampled,
        certifiedExact,
        gapMinLow,
        gapMinHigh,
        gapMinCommits,
        gapMinNonPositive,
    };
}

function formatRow(columns, widths) {
    return columns.map((value, index) =>
        String(value).padStart(widths[index])).join("  ");
}

async function main() {
    const options = parseArgs(process.argv.slice(2));
    const scenes = buildScenes(options);
    scenes.scale = Math.exp(-2);
    const harness = createHarness();

    const policies = [
        { name: "monolithic 30 Hz", budget: null, requestHz: options.requestHz, monolithic: true },
        { name: "two-stream 30 Hz", budget: null, requestHz: options.requestHz },
        { name: "DOC r=0 30 Hz", budget: 0, requestHz: options.requestHz },
        { name: "DOC r=0.25 30 Hz", budget: 0.25, requestHz: options.requestHz },
        { name: "DOC r=0.5 30 Hz", budget: 0.5, requestHz: options.requestHz },
        { name: "DOC r=1 30 Hz", budget: 1, requestHz: options.requestHz },
        { name: "DOC r=0.5 60 Hz", budget: 0.5, requestHz: 60 },
    ];

    const results = [];
    for (const policy of policies) {
        const scene = {
            count: scenes.count,
            positions: scenes.positions,
            attributes: scenes.attributes,
            gate: policy.monolithic ? scenes.monolithicGate : scenes.gate,
            scale: scenes.scale,
        };
        harness.installScene(scene);
        await harness.send({ view: VIEW_PROJECTION });
        await harness.send({ time: 0, dynamicRequestId: 0 });
        results.push(await runPolicy(harness, scene, options, policy));
    }

    const animated = scenes.gate.reduce((sum, value) => sum + value, 0);
    const interval = depthIntervalOf(scenes);
    const keys = keysAt(scenes, 0);
    let keyMin = Infinity;
    let keyMax = -Infinity;
    for (let i = 0; i < keys.length; i++) {
        if (keys[i] < keyMin) keyMin = keys[i];
        if (keys[i] > keyMax) keyMax = keys[i];
    }

    process.stdout.write([
        `scene      ${options.mode}: ${options.count.toLocaleString()} Gaussians, ` +
        `${animated.toLocaleString()} animated (${(100 * animated / options.count).toFixed(1)}%)`,
        `footprint  a Gaussian radius is ${Math.exp(-2).toFixed(4)} world units = ` +
        `${(Math.exp(-2) * interval.scale).toFixed(3)} depth keys`,
        `playback   ${options.seconds}s clip over ${options.seconds}s wall time, ` +
        `${DISPLAY_HZ} Hz display, requests at ${options.requestHz} Hz`,
        `quantiser  B = 2^16 of Eq. (31), kappa = ${interval.scale.toFixed(3)} keys per ` +
        `world unit, rows spanning ${keyMin}..${keyMax} of the ${DEPTH_BUCKETS} bins at t = 0`,
        "",
    ].join("\n"));

    const header = [
        "policy", "req", "sorts", "reuse%", "ms/s", "ms/req", "MB/s",
        "red ms/s", "red MB/s", "inv%", "drift(r)", "mis%", "shift", "exact",
    ];
    const widths = [18, 4, 6, 7, 7, 7, 7, 9, 9, 7, 9, 7, 6, 9];
    process.stdout.write(formatRow(header, widths) + "\n");
    const baseline = results[1];
    for (const result of results) {
        process.stdout.write(formatRow([
            result.name,
            result.requests,
            result.sorts,
            (100 * result.reuseFraction).toFixed(1),
            result.perSecondMs.toFixed(1),
            result.msPerRequest.toFixed(2),
            (result.perSecondBytes / 1e6).toFixed(1),
            (100 * (1 - result.perSecondMs / baseline.perSecondMs)).toFixed(0) + "%",
            (100 * (1 - result.perSecondBytes / baseline.perSecondBytes)).toFixed(0) + "%",
            (100 * result.meanInversions).toFixed(2),
            result.worstRadii.toFixed(2),
            (100 * result.deviationSum / (result.deviationFrames || 1) / options.count).toFixed(3),
            result.deviationShift,
            result.reusedSampled > 0
                ? `${result.exactSamples}/${result.reusedSampled}`
                : "-",
        ], widths) + "\n");
    }

    const certified = results.find((result) => result.name.includes("r=0 "));
    const budget = results.find((result) => result.name.includes("r=0.5 30"));
    const monolithic = results[0];
    const line = (label, result) =>
        `${label}: ${(100 * result.reuseFraction).toFixed(1)}% of the requests reused, ` +
        `${result.perSecondMs.toFixed(1)} ms/s of worker sort time, ` +
        `${(result.perSecondBytes / 1e6).toFixed(1)} MB/s of index traffic`;
    process.stdout.write("\n" + [
        line("monolithic", monolithic),
        line("two-stream", baseline),
        line("DOC certified (r=0)", certified),
        line("DOC budget (r=0.5)", budget),
        "",
        `DOC vs two-stream (r=0):   ` +
        `${(100 * (1 - certified.perSecondMs / baseline.perSecondMs)).toFixed(0)}% less worker sort time, ` +
        `${(100 * (1 - certified.perSecondBytes / baseline.perSecondBytes)).toFixed(0)}% fewer index bytes`,
        `DOC vs two-stream (r=0.5): ` +
        `${(100 * (1 - budget.perSecondMs / baseline.perSecondMs)).toFixed(0)}% less worker sort time, ` +
        `${(100 * (1 - budget.perSecondBytes / baseline.perSecondBytes)).toFixed(0)}% fewer index bytes`,
        `separation + static cache vs monolithic: ` +
        `${(100 * (1 - baseline.sortMs / monolithic.sortMs)).toFixed(0)}% less worker sort time, ` +
        `${(100 * (1 - baseline.perSecondBytes / monolithic.perSecondBytes)).toFixed(0)}% fewer index bytes`,
        "",
        "certificate (budget r = 0)",
        `  g_min over the ${certified.gapMinCommits} commits of the run: ` +
        `${Number.isFinite(certified.gapMinLow) ? certified.gapMinLow : "n/a"}..` +
        `${Number.isFinite(certified.gapMinHigh) ? certified.gapMinHigh : "n/a"} keys, ` +
        `at or below zero on ${certified.gapMinNonPositive} of them`,
        `  animated rows sweep ${certified.rateKeys.toFixed(0)} keys per canonical second of the clip`,
        `  reuse granted while the bound 2*ceil(M(t)) reached ${certified.boundAtReuseMax} keys, ` +
        `refused once it passed ` +
        `${Number.isFinite(certified.boundAtRefusalMin)
            ? certified.boundAtRefusalMin + " keys"
            : "nothing"}`,
        `  headroom at reuse ${certified.reuseHeadroomMin}..${certified.reuseHeadroomMax} keys, ` +
        `shortfall at refusal ` +
        `${Number.isFinite(certified.refusalHeadroomMax) ? certified.refusalHeadroomMax : "n/a"} keys`,
        `  exactness ${certified.exactSamples}/${certified.reusedSampled} reused commits reproduce ` +
        `the shipped sort` +
        (certified.exactMismatches
            ? `, ${certified.exactMismatches} rows out of place, worst shift ${certified.exactMaxShift}`
            : " (row for row)") +
        `; the ideal key order differs in ${certified.idealMismatches} rows ` +
        `(worst shift ${certified.idealMaxShift})`,
        `  of ${certified.reuses} reuses, ${certified.trivialReuses} were taken at zero movement ` +
        `and ${certified.certifiedReuses} under a positive bound, ` +
        `${certified.certifiedExact}/${certified.certifiedSampled} of the checked ones exact`,
        "tolerance (budget r = 0.5)",
        `  accepts ${budget.toleranceKeys.toFixed(0)} keys of drift, ` +
        `${budget.exactSamples}/${budget.reusedSampled} reused commits reproduce the shipped sort`,
        `  mean adjacency error ${(100 * budget.meanInversions).toFixed(2)}%, ` +
        `worst depth violation ${budget.worstRadii.toFixed(2)} radii`,
        "reference",
        `  replica against the worker's own sort: ${baseline.freshSampled} fresh sorts, ` +
        (baseline.freshMismatches
            ? `${baseline.freshMismatches} rows out of place, worst shift ${baseline.freshMaxShift}`
            : "row for row"),
        `  the shipped counting sort against the key order of Eq. (31) on the same sorts: ` +
        `${baseline.bucketMismatches} rows out of place, worst shift ${baseline.bucketMaxShift}`,
        "",
    ].join("\n"));
}

main().catch((error) => {
    process.stderr.write(String(error && error.stack ? error.stack : error) + "\n");
    process.exitCode = 1;
});
