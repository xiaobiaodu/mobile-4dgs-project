// Regression tests for the Mobile-GS2 static/dynamic separation.
//
// The viewer decodes a committed dynamic gate, loads the time-invariant and
// animated Gaussians as two streams, and keeps the static projection and depth
// sort between frames.  These tests drive the real worker source (the same
// `createWorker` body the page hands to `new Worker`) inside a VM with a stubbed
// TMC3 decoder, so the decode, gate, sort and merge code paths all run.

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const MAIN_JS = path.join(__dirname, "..", "render_shared", "main.js");

const GRID = [-4, -1.5, 1, 3.5];
// The motion texture is float32, so the fixture mirrors that precision and the
// expected depth keys stay bit-comparable with the worker's arithmetic.
const DYNAMIC_VELOCITY = Math.fround(1.2);
const DYNAMIC_CANONICAL_TIME = Math.fround(0.5);
const DYNAMIC_LOG_DURATION = Math.fround(-1);
const STATIC_LOG_DURATION = Math.log(1e15);
// Both view directions are unit length so the worker's view-change test (the
// dot product of the third column) treats repeated frames as unchanged, and
// their dot product with each other is far enough from one to be detected.
const VIEW_PROJECTION = [0, 0, 0.6, 0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 1];
const ALT_VIEW_PROJECTION = [0, 0, 0.8, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 1];

// ---------------------------------------------------------------- IEEE half

function halfToFloat(h) {
    const s = (h & 0x8000) >> 15;
    const e = (h & 0x7c00) >> 10;
    const f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

const _f32 = new Float32Array(1);
const _i32 = new Int32Array(_f32.buffer);

function floatToHalfBits(value) {
    _f32[0] = value;
    const x = _i32[0];
    const sign = (x >>> 16) & 0x8000;
    let exponent = ((x >>> 23) & 0xff) - 127 + 15;
    const mantissa = x & 0x7fffff;
    if (exponent >= 0x1f) return sign | 0x7c00;
    if (exponent <= 0) {
        if (exponent < -10) return sign;
        const shifted = (mantissa | 0x800000) >> (1 - exponent);
        return sign | ((shifted + 0x1000 + ((shifted >> 13) & 1)) >> 13);
    }
    const rounded = (mantissa + 0x1000 + ((mantissa >> 13) & 1)) >> 13;
    if (rounded === 0x400) {
        exponent += 1;
        if (exponent >= 0x1f) return sign | 0x7c00;
        return sign | (exponent << 10);
    }
    return sign | (exponent << 10) | rounded;
}

// ------------------------------------------------------------------- model

function xyzToPlyInts([x, y, z]) {
    const ints = [];
    for (const value of [x, y, z]) {
        const bits = floatToHalfBits(value);
        assert.equal(halfToFloat(bits), value, `test position ${value} must be half-exact`);
        ints.push((bits ^ 0x8000) & 0xffff);
    }
    return ints;
}

function mortonOrder(ints) {
    return ints
        .map((t, index) => ({ z: t[2], y: t[1], x: t[0], index }))
        .sort((a, b) => a.z - b.z || a.y - b.y || a.x - b.x)
        .map((entry) => entry.index);
}

function buildPly(ints) {
    const header = "ply\n" +
        "format binary_little_endian 1.0\n" +
        "comment synthetic test fixture\n" +
        `element vertex ${ints.length}\n` +
        "property uint16 x\n" +
        "property uint16 y\n" +
        "property uint16 z\n" +
        "end_header\n";
    const headerBytes = Buffer.from(header, "ascii");
    const data = Buffer.alloc(ints.length * 6);
    ints.forEach((triple, index) => {
        data.writeUInt16LE(triple[0], index * 6 + 0);
        data.writeUInt16LE(triple[1], index * 6 + 2);
        data.writeUInt16LE(triple[2], index * 6 + 4);
    });
    return new Uint8Array(Buffer.concat([headerBytes, data]));
}

// The viewer's view-space depth: 4096 quantisation with int32 truncation.
function depthKey(viewProj, x, y, z) {
    return ((Math.fround(viewProj[2]) * x + Math.fround(viewProj[6]) * y +
        Math.fround(viewProj[10]) * z) * 4096) | 0;
}

function motionAt(attribute, time) {
    const dt = time - attribute[6];
    const halfDtSquared = 0.5 * dt * dt;
    return {
        x: attribute[0] * dt + attribute[3] * halfDtSquared,
        y: attribute[1] * dt + attribute[4] * halfDtSquared,
        z: attribute[2] * dt + attribute[5] * halfDtSquared,
    };
}

// Brute force reference: order by quantized depth, ties broken by row id.
function expectedOrder(positions, attributes, time, viewProj) {
    return positions
        .map((position, id) => {
            const offset = motionAt(attributes[id], time);
            return {
                id,
                key: depthKey(
                    viewProj,
                    position[0] + offset.x,
                    position[1] + offset.y,
                    position[2] + offset.z,
                ),
            };
        })
        .sort((a, b) => a.key - b.key || a.id - b.id)
        .map((entry) => entry.id);
}

// The pre-separation monolithic counting sort, kept verbatim as a reference so
// models without a usable gate still hash out to the same draw order.
function referenceMonolithicOrder(keys) {
    const count = keys.length;
    let maxDepth = -Infinity;
    let minDepth = Infinity;
    for (let i = 0; i < count; i++) {
        if (keys[i] > maxDepth) maxDepth = keys[i];
        if (keys[i] < minDepth) minDepth = keys[i];
    }
    const depthInv = (256 * 256 - 1) / (maxDepth - minDepth);
    const sizeList = new Int32Array(count);
    const counts = new Uint32Array(256 * 256);
    for (let i = 0; i < count; i++) {
        sizeList[i] = ((keys[i] - minDepth) * depthInv) | 0;
        counts[sizeList[i]]++;
    }
    const starts = new Uint32Array(256 * 256);
    for (let i = 1; i < 256 * 256; i++) starts[i] = starts[i - 1] + counts[i - 1];
    const order = new Array(count);
    for (let i = 0; i < count; i++) order[starts[sizeList[i]]++] = i;
    return order;
}

// ------------------------------------------------------- container builder

const rawBytes = (data) => ({ __kind: "bytes", data: Uint8Array.from(data) });
const tensor = (dtype, shape, values) => {
    let data;
    if (dtype === "float32") data = Float32Array.from(values);
    else if (dtype === "uint8") data = Uint8Array.from(values);
    else throw new Error(`unsupported test dtype ${dtype}`);
    return { __kind: "ndarray", dtype, shape, data };
};

function serializeContainer(dict) {
    const blobs = [];
    let offset = 0;
    const process = (value) => {
        if (value && value.__kind) {
            const source = Buffer.from(value.data.buffer, value.data.byteOffset, value.data.byteLength);
            const padding = (4 - (source.length % 4)) % 4;
            const meta = {
                _type: value.__kind,
                offset,
                length: source.length,
            };
            if (value.__kind === "ndarray") {
                meta.dtype = value.dtype;
                meta.shape = value.shape;
            }
            blobs.push(Buffer.concat([source, Buffer.alloc(padding)]));
            offset += source.length + padding;
            return meta;
        }
        if (Array.isArray(value)) return value.map(process);
        if (value && typeof value === "object") {
            const out = {};
            for (const key of Object.keys(value)) out[key] = process(value[key]);
            return out;
        }
        return value;
    };
    const metadata = process(dict);
    const json = Buffer.from(JSON.stringify(metadata), "utf8");
    const header = Buffer.alloc(4);
    header.writeUInt32LE(json.length, 0);
    return new Uint8Array(Buffer.concat([header, json, ...blobs]));
}

// Canonical prefix code for the synthetic dynamic stream: symbol 0 is the static
// one bit code, symbols 1 and 2 are the two opposing velocities.
const MULTI_SYMBOL_TABLE = { 0: [1, 0], 1: [2, 2], 2: [2, 3] };
const SINGLE_SYMBOL_TABLE = { 0: [1, 0] };

function huffmanBits(symbols) {
    const bytes = [];
    let current = 0;
    let bitCount = 0;
    for (const symbol of symbols) {
        const [length, value] = MULTI_SYMBOL_TABLE[symbol];
        for (let bit = length - 1; bit >= 0; bit--) {
            current = (current << 1) | ((value >> bit) & 1);
            bitCount++;
            if (bitCount === 8) {
                bytes.push(current);
                current = 0;
                bitCount = 0;
            }
        }
    }
    if (bitCount > 0) bytes.push(current << (8 - bitCount));
    return Uint8Array.from(bytes);
}

const zeroTensor = (dtype, shape, size) => tensor(dtype, shape, new Array(size).fill(0));

// A non-zero output bias makes the residual network produce a known constant
// offset, which lets a test prove that the offset was baked at load time.
function mlpOffsetTensor(sizes, shBias = 0) {
    return {
        "main.0.weight": zeroTensor("float32", [64, 23], sizes.main0Weight),
        "main.0.bias": zeroTensor("float32", [64], sizes.main0Bias),
        "main.2.weight": zeroTensor("float32", [64, 64], sizes.main2Weight),
        "main.2.bias": zeroTensor("float32", [64], sizes.main2Bias),
        "main.4.weight": zeroTensor("float32", [64, 64], sizes.main4Weight),
        "main.4.bias": zeroTensor("float32", [64], sizes.main4Bias),
        "shs_output.0.weight": zeroTensor("float32", [12, 64], sizes.shsWeight),
        "shs_output.0.bias": tensor("float32", [12], new Array(12).fill(shBias)),
    };
}

// Builds a Mobile-GS2 comp.json whose Gaussians are already stored in Morton
// order, exactly like the exporter writes them.
function buildModel({
    positions,
    dynamicMask,
    dynamicVelocity = DYNAMIC_VELOCITY,
    gateMetadata = "full",
    shBias = 0,
}) {
    const ints = positions.map(xyzToPlyInts);
    const order = mortonOrder(ints);
    const savedInts = order.map((index) => ints[index]);
    const savedPositions = order.map((index) => positions[index]);
    const savedDynamic = order.map((index) => dynamicMask[index]);
    const count = positions.length;

    // The animated rows alternate between two opposing velocities so the dynamic
    // stream genuinely re-orders as time advances instead of merely translating.
    const dynamicRow = [];
    let dynamicOrdinal = 0;
    for (const isDynamic of savedDynamic) {
        dynamicRow.push(isDynamic ? 1 + (dynamicOrdinal++ % 2) : 0);
    }

    const dynamicCodebook = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [dynamicVelocity, 0, 0, 0, 0, 0, DYNAMIC_CANONICAL_TIME, DYNAMIC_LOG_DURATION],
        [-dynamicVelocity, 0, 0, 0, 0, 0, DYNAMIC_CANONICAL_TIME, DYNAMIC_LOG_DURATION],
    ];

    const saveDict = {
        xyz: rawBytes([0, 0, 0, 0, 0, 0, 0, 0]),
        scale_code: [tensor("float32", [1, 3], [-2, -2, -2])],
        scale_index: [rawBytes(huffmanBits(new Array(count).fill(0)))],
        scale_htable: [SINGLE_SYMBOL_TABLE],
        rotation_code: [tensor("float32", [1, 4], [1, 0, 0, 0])],
        rotation_index: [rawBytes(huffmanBits(new Array(count).fill(0)))],
        rotation_htable: [SINGLE_SYMBOL_TABLE],
        app_code: [tensor("float32", [1, 6], [0, 0, 0, 0, 0, 0])],
        app_index: [rawBytes(huffmanBits(new Array(count).fill(0)))],
        app_htable: [SINGLE_SYMBOL_TABLE],
        dynamic_enabled: true,
        dynamic_code: [tensor("float32", [3, 8], dynamicCodebook.flat())],
        dynamic_index: [rawBytes(huffmanBits(dynamicRow))],
        dynamic_htable: [MULTI_SYMBOL_TABLE],
        dynamic_gate_count: count,
        dynamic_gate_bits: tensor("uint8", [Math.ceil(count / 8)], packGateBits(savedDynamic)),
        MLP_cont: zeroTensor("float32", [7168], 7168),
        MLP_opacity: zeroTensor("float32", [2048], 2048),
        MLP_dc: zeroTensor("float32", [2048], 2048),
        MLP_sh: zeroTensor("float32", [2048], 2048),
        MLP_offset: mlpOffsetTensor({
            main0Weight: 64 * 23,
            main0Bias: 64,
            main2Weight: 64 * 64,
            main2Bias: 64,
            main4Weight: 64 * 64,
            main4Bias: 64,
            shsWeight: 12 * 64,
            shsBias: 12,
        }, shBias),
    };

    if (gateMetadata === "none") {
        delete saveDict.dynamic_gate_count;
        delete saveDict.dynamic_gate_bits;
    } else if (gateMetadata === "mismatch") {
        saveDict.dynamic_gate_count = count + 1;
    }

    return {
        bytes: serializeContainer(saveDict),
        ply: buildPly(savedInts),
        positions: savedPositions,
        dynamicMask: savedDynamic,
        dynamicRow,
        count,
    };
}

function packGateBits(mask) {
    const bytes = new Uint8Array(Math.ceil(mask.length / 8));
    mask.forEach((value, index) => {
        if (value) bytes[index >> 3] |= 1 << (index & 7);
    });
    return bytes;
}

function materializedAttributes(attributes, dynamicMask) {
    return attributes.map((attribute, id) => {
        if (dynamicMask[id]) return attribute.slice();
        return [0, 0, 0, 0, 0, 0, 0, STATIC_LOG_DURATION];
    });
}

function modelAttributes(model) {
    return model.dynamicRow.map((row) => {
        if (row === 0) return [0, 0, 0, 0, 0, 0, 0, DYNAMIC_LOG_DURATION];
        const velocity = row === 1 ? DYNAMIC_VELOCITY : -DYNAMIC_VELOCITY;
        return [
            velocity, 0, 0,
            0, 0, 0,
            DYNAMIC_CANONICAL_TIME,
            DYNAMIC_LOG_DURATION,
        ];
    });
}

// Static fixture: a coarse lattice of background Gaussians plus a dense column of
// animated Gaussians.  The column makes the two opposing velocities reorder the
// dynamic stream within a single frame, and both streams stay inside the 65536
// bucket range so the expected order is unambiguous.
function testFixture() {
    const positions = [];
    const dynamicMask = [];
    for (const x of GRID) {
        for (const y of GRID) {
            for (const z of GRID) {
                positions.push([x, y, z]);
                dynamicMask.push(false);
            }
        }
    }
        for (let x = -4; x <= 3.5; x += 0.5) {
            positions.push([x, 0, 0]);
            dynamicMask.push(true);
        }
    return { positions, dynamicMask };
}

// A one dimensional lattice keeps every adjacent depth gap far wider than the
// depth quantisation, which gives the strict (zero budget) scheduler a fixture
// where its sufficient condition can actually hold.
function latticeFixture(count = 160) {
    const positions = [];
    const dynamicMask = [];
    for (let i = 0; i < count; i++) {
        positions.push([i * 0.5 - 40, 0, 0]);
        dynamicMask.push(i % 2 === 0);
    }
    return { positions, dynamicMask };
}

// --------------------------------------------------------------- the worker

function createHarness() {
    const source = fs.readFileSync(MAIN_JS, "utf8");
    const start = source.indexOf("function createWorker(self) {");
    assert.ok(start >= 0, "createWorker must exist in render_shared/main.js");
    const end = source.indexOf("\nconst vertexShaderSource");
    assert.ok(end > start, "createWorker must be followed by the shader source");
    let body = source.slice(start, end).trimEnd();
    // Count every network evaluation so a test can prove that decoding is the
    // only stage that runs one.
    for (const name of ["runTCNN_MLP", "runPyTorch_MLP"]) {
        const marker = `function ${name}(`;
        const at = body.indexOf(marker);
        assert.ok(at >= 0, `${name} must exist in the worker body`);
        const brace = body.indexOf("{", at);
        body = body.slice(0, brace + 1) +
            "\n        globalThis.__mlpCalls = (globalThis.__mlpCalls || 0) + 1;" +
            body.slice(brace + 1);
    }

    const messages = [];
    const warnings = [];
    const sandbox = {
        console: {
            log() {},
            warn(...args) { warnings.push(args.join(" ")); },
            error() {},
            time() {},
            timeEnd() {},
        },
        TextDecoder,
        setTimeout,
        clearTimeout,
    };
    const context = vm.createContext(sandbox);
    vm.runInContext(
        "globalThis.self = globalThis;\n" +
        "globalThis.TMC3_URL = 'about:blank';\n" +
        "globalThis.importScripts = function () {};\n" +
        "globalThis.postMessage = function (message) { globalThis.__post(message); };\n" +
        "globalThis.__mkBytes = function (length) { return new Uint8Array(length); };\n" +
        "globalThis.__mkF32 = function (length) { return new Float32Array(length); };\n" +
        "globalThis.__asF32 = function (view) { return new Float32Array(view.buffer, view.byteOffset, view.byteLength / 4); };\n",
        context,
    );
    sandbox.__post = (message) => messages.push(message);
    vm.runInContext(`(${body})(self);`, context);
    vm.runInContext("self.postMessage = function (message) { globalThis.__post(message); };", context);

    const mkBytes = vm.runInContext("globalThis.__mkBytes", context);
    const mkF32 = vm.runInContext("globalThis.__mkF32", context);
    const asF32 = vm.runInContext("globalThis.__asF32", context);
    const onMessage = vm.runInContext("self.onmessage", context);
    const module = vm.runInContext("self.Module", context);

    let plyBytes = new Uint8Array(0);
    module.FS = {
        writeFile() {},
        readFile() { return vmBytes(plyBytes); },
        unlink() {},
    };
    module.callMain = () => {};

    function vmBytes(hostBytes) {
        const out = mkBytes(hostBytes.length);
        for (let i = 0; i < hostBytes.length; i++) out[i] = hostBytes[i];
        return out;
    }

    function vmFloat32(values) {
        const out = mkF32(values.length);
        for (let i = 0; i < values.length; i++) out[i] = values[i];
        return out;
    }

    const settle = async (rounds = 4) => {
        for (let i = 0; i < rounds; i++) {
            await new Promise((resolve) => setTimeout(resolve, 1));
        }
    };

    return {
        messages,
        warnings,
        asF32,
        mlpCalls() {
            return vm.runInContext("globalThis.__mlpCalls || 0", context);
        },
        async load(model) {
            plyBytes = model.ply;
            messages.length = 0;
            await onMessage({ data: { mobilegs: vmBytes(model.bytes).buffer } });
            await vm.runInContext("self.Module.onRuntimeInitialized", context)();
            await settle();
            const error = messages.find((message) => message.error);
            assert.equal(error, undefined, `worker reported: ${error && error.error}`);
        },
        async sendView(viewProj) {
            await onMessage({ data: { view: vmFloat32(viewProj) } });
            await settle();
        },
        async sendTime(time, requestId) {
            await onMessage({
                data: { time, dynamicRequestId: requestId },
            });
            await settle();
        },
        async sendAdaptiveSort(enabled, budget) {
            await onMessage({ data: { adaptiveSort: { enabled, budget } } });
            await settle();
        },
        reuseFrames() {
            return messages.filter((message) => message.reusedOrder);
        },
        lastFrame() {
            const frames = messages.filter((message) => message.depthIndex);
            assert.ok(frames.length > 0, "a depth-ordered frame must have been posted");
            return frames[frames.length - 1];
        },
        lastOrder() {
            return Array.from(this.lastFrame().depthIndex);
        },
        lastOf(key) {
            const found = messages.filter((message) => message[key]);
            return found.length ? found[found.length - 1] : null;
        },
    };
}

// ------------------------------------------------------------------- specs

test("gated model loads a static and a dynamic stream", async () => {
    const harness = createHarness();
    const { positions, dynamicMask } = testFixture();
    const model = buildModel({ positions, dynamicMask });
    const expectedDynamic = model.dynamicMask.filter(Boolean).length;

    await harness.load(model);
    await harness.sendView(VIEW_PROJECTION);

    const split = harness.lastOf("dynamic");
    assert.ok(split, "the worker must report the split");
    assert.equal(split.dynamic.gated, true);
    assert.equal(split.dynamic.dynamicCount, expectedDynamic);
    assert.equal(split.dynamic.staticCount, model.count - expectedDynamic);
    assert.equal(split.dynamic.staticCount + split.dynamic.dynamicCount, model.count);
});

test("static rows keep a time invariant motion texture", async () => {
    const harness = createHarness();
    const { positions, dynamicMask } = testFixture();
    const model = buildModel({ positions, dynamicMask });

    await harness.load(model);
    await harness.sendView(VIEW_PROJECTION);

    const payload = harness.lastOf("texdata_dynamic");
    assert.ok(payload, "a dynamic motion texture must be posted");
    const texture = payload.texdata_dynamic;
    for (let i = 0; i < model.count; i++) {
        const first = i * 8;
        if (model.dynamicMask[i]) {
            assert.equal(Math.abs(texture[first + 0]), Math.fround(DYNAMIC_VELOCITY));
            assert.equal(texture[first + 3], DYNAMIC_CANONICAL_TIME);
            assert.equal(texture[first + 7], DYNAMIC_LOG_DURATION);
        } else {
            for (let j = 0; j < 8; j++) {
                assert.equal(texture[first + j], j === 7 ? Math.fround(STATIC_LOG_DURATION) : 0,
                    `static Gaussian ${i} must stay time invariant`);
            }
        }
    }
});

test("gated sort matches the exact depth order and caches the static stream", async () => {
    const harness = createHarness();
    const { positions, dynamicMask } = testFixture();
    const model = buildModel({ positions, dynamicMask });
    const attributes = materializedAttributes(modelAttributes(model), model.dynamicMask);
    const staticIds = [];
    const dynamicIds = [];
    model.dynamicMask.forEach((isDynamic, id) => (isDynamic ? dynamicIds : staticIds).push(id));

    await harness.load(model);
    await harness.sendView(VIEW_PROJECTION);

    // The synthetic scene keeps each stream's depth range under the 65536 bucket
    // count, so the stream order is exactly depth-then-row-id and the merged
    // order has to equal the brute force result.
    const ranges = [staticIds, dynamicIds].map((ids) => {
        const keys = ids.map((id) => depthKey(VIEW_PROJECTION, ...model.positions[id]));
        return Math.max(...keys) - Math.min(...keys);
    });
    for (const range of ranges) assert.ok(range < 65535, "test fixture must fit the bucket range");

    const frames = [{ time: 0, viewProj: VIEW_PROJECTION }];
    frames.push({ time: 0, viewProj: ALT_VIEW_PROJECTION });
    frames.push({ time: 0.25, viewProj: ALT_VIEW_PROJECTION });
    frames.push({ time: 0.75, viewProj: ALT_VIEW_PROJECTION });

    const orders = [];
    for (let index = 0; index < frames.length; index++) {
        const frame = frames[index];
        if (index === 1) {
            await harness.sendView(frame.viewProj);
        } else if (index > 0) {
            await harness.sendTime(frame.time, index);
        }
        const order = harness.lastOrder();
        orders.push(order);
        assert.deepEqual(
            [...order].sort((a, b) => a - b),
            Array.from({ length: model.count }, (unused, id) => id),
            "every Gaussian must appear exactly once",
        );
        assert.deepEqual(
            order,
            expectedOrder(model.positions, attributes, frame.time, frame.viewProj),
            `frame ${index} must reproduce the exact depth order`,
        );
    }

    const staticSubsequence = orders.map((order) => order.filter((id) => !model.dynamicMask[id]));
    assert.deepEqual(staticSubsequence[1], staticSubsequence[2],
        "a fixed camera must reuse the static sort across time steps");
    assert.deepEqual(staticSubsequence[2], staticSubsequence[3],
        "a fixed camera must reuse the static sort across time steps");
    assert.notDeepEqual(
        staticSubsequence[0],
        staticSubsequence[1],
        "a camera change must re-project and re-sort the static stream",
    );

    const dynamicSubsequence = orders.map((order) => order.filter((id) => model.dynamicMask[id]));
    assert.notDeepEqual(
        dynamicSubsequence[2],
        dynamicSubsequence[3],
        "advanced time must re-order the moving Gaussians",
    );
});

test("time-only frames reuse the cached static sort", async () => {
    const harness = createHarness();
    const { positions, dynamicMask } = testFixture();
    const model = buildModel({ positions, dynamicMask });

    await harness.load(model);
    await harness.sendView(VIEW_PROJECTION);
    const first = harness.lastFrame().sortStats;
    assert.equal(first.staticResorted, true, "the first frame must project every stream");
    assert.equal(first.dynamicResorted, true);
    assert.equal(first.staticPoints, model.count - model.dynamicMask.filter(Boolean).length);
    assert.equal(first.dynamicPoints, model.dynamicMask.filter(Boolean).length);
    assert.equal(first.merged, true, "two non-empty streams must be merged");

    await harness.sendTime(0.3, 1);
    const animated = harness.lastFrame().sortStats;
    assert.equal(animated.staticResorted, false, "a fixed camera must keep the static projection");
    assert.equal(animated.dynamicResorted, true, "a new time must re-project the animated stream");

    await harness.sendTime(0.6, 2);
    const animatedAgain = harness.lastFrame().sortStats;
    assert.equal(animatedAgain.staticResorted, false);

    await harness.sendView(ALT_VIEW_PROJECTION);
    const moved = harness.lastFrame().sortStats;
    assert.equal(moved.staticResorted, true, "a camera change must re-project the static stream");
    assert.equal(moved.dynamicResorted, true);
});
test("a gate without animation still keeps the exact static order", async () => {
    const harness = createHarness();
    const { positions } = testFixture();
    const model = buildModel({ positions, dynamicMask: new Array(positions.length).fill(false) });
    const attributes = materializedAttributes(modelAttributes(model), model.dynamicMask);
    const ids = positions.map((unused, id) => id);

    await harness.load(model);
    await harness.sendView(VIEW_PROJECTION);
    const first = harness.lastOrder();
    assert.deepEqual(first, expectedOrder(model.positions, attributes, 0, VIEW_PROJECTION));

    await harness.sendTime(0.6, 1);
    const second = harness.lastOrder();
    assert.deepEqual(second, first, "time must not disturb an all-static model");

    const split = harness.lastOf("dynamic");
    assert.equal(split.dynamic.dynamicCount, 0);
    assert.equal(split.dynamic.staticCount, ids.length);
    assert.equal(harness.lastFrame().sortStats.merged, false,
        "an all-static model must not pay for a merge pass");
});

test("a fully dynamic gate matches the monolithic reference sort", async () => {
    const harness = createHarness();
    const { positions } = testFixture();
    const model = buildModel({ positions, dynamicMask: new Array(positions.length).fill(true) });
    const attributes = modelAttributes(model);

    await harness.load(model);
    await harness.sendView(VIEW_PROJECTION);

    const keys = model.positions.map((position, id) => {
        const offset = motionAt(attributes[id], 0);
        return depthKey(VIEW_PROJECTION, position[0] + offset.x, position[1] + offset.y, position[2] + offset.z);
    });
    assert.deepEqual(harness.lastOrder(), referenceMonolithicOrder(keys));
    assert.equal(harness.lastOf("dynamic").dynamic.dynamicCount, model.count);
});

test("a model without gate metadata keeps the pre-gate behaviour", async () => {
    const harness = createHarness();
    const { positions } = testFixture();
    const model = buildModel({
        positions,
        dynamicMask: new Array(positions.length).fill(true),
        gateMetadata: "none",
    });
    const attributes = modelAttributes(model);

    await harness.load(model);
    await harness.sendView(VIEW_PROJECTION);

    const keys = model.positions.map((position, id) => {
        const offset = motionAt(attributes[id], 0);
        return depthKey(VIEW_PROJECTION, position[0] + offset.x, position[1] + offset.y, position[2] + offset.z);
    });
    assert.deepEqual(harness.lastOrder(), referenceMonolithicOrder(keys));
    const split = harness.lastOf("dynamic");
    assert.equal(split.dynamic.dynamicCount, model.count);
    assert.equal(split.dynamic.staticCount, 0);
    assert.equal(harness.lastFrame().sortStats.merged, false,
        "a fully dynamic model must not pay for a merge pass");

    await harness.sendTime(0.4, 1);
    const movedKeys = model.positions.map((position, id) => {
        const offset = motionAt(attributes[id], 0.4);
        return depthKey(VIEW_PROJECTION, position[0] + offset.x, position[1] + offset.y, position[2] + offset.z);
    });
    assert.deepEqual(harness.lastOrder(), referenceMonolithicOrder(movedKeys));
});

test("a malformed gate falls back to an all-dynamic model", async () => {
    const harness = createHarness();
    const { positions, dynamicMask } = testFixture();
    const model = buildModel({
        positions,
        dynamicMask,
        gateMetadata: "mismatch",
    });
    const attributes = modelAttributes(model);

    await harness.load(model);
    await harness.sendView(VIEW_PROJECTION);

    const keys = model.positions.map((position, id) => {
        const offset = motionAt(attributes[id], 0);
        return depthKey(VIEW_PROJECTION, position[0] + offset.x, position[1] + offset.y, position[2] + offset.z);
    });
    assert.deepEqual(harness.lastOrder(), referenceMonolithicOrder(keys));
    assert.equal(harness.lastOf("dynamic").dynamic.dynamicCount, model.count);
    assert.ok(
        harness.warnings.some((warning) => warning.includes("Dynamic gate covers")),
        "the fallback must warn",
    );
});

test("the appearance residual is baked once and never re-evaluated", async () => {
    const harness = createHarness();
    const { positions, dynamicMask } = testFixture();
    // The zero-initialised residual network plus a constant output bias makes
    // every baked coefficient equal to that bias.
    const shBias = 0.25;
    const model = buildModel({ positions, dynamicMask, shBias });

    await harness.load(model);
    await harness.sendView(VIEW_PROJECTION);
    await harness.sendTime(0.4, 1);
    await harness.sendTime(0.9, 2);

    const shPayloads = harness.messages.filter((message) => message.texdata_sh);
    assert.equal(shPayloads.length, 1, "the SH texture is uploaded exactly once");
    const shTexture = shPayloads[0].texdata_sh;
    for (let i = 0; i < model.count * 12; i++) {
        assert.equal(
            shTexture[i],
            shBias,
            `SH coefficient ${i} must carry the residual baked at load time`,
        );
    }
    for (const message of harness.messages) {
        assert.equal(message.dynamicSh, undefined, "no per-frame SH payload may be posted");
    }
});

test("no network is evaluated once a model is decoded", async () => {
    const harness = createHarness();
    const { positions, dynamicMask } = testFixture();
    const model = buildModel({ positions, dynamicMask });

    await harness.load(model);
    const afterLoad = harness.mlpCalls();
    assert.ok(afterLoad > 0, "decoding must evaluate the attribute networks");

    await harness.sendView(VIEW_PROJECTION);
    await harness.sendTime(0.35, 1);
    await harness.sendTime(0.7, 2);
    await harness.sendView(ALT_VIEW_PROJECTION);
    await harness.sendTime(0.9, 3);
    await harness.sendTime(0.1, 4);

    assert.equal(
        harness.mlpCalls(),
        afterLoad,
        "sorting, playback, scrubbing and camera changes must evaluate no network",
    );
});

// ------------------------------------------------- adaptive sort scheduling

// The strict scheduler decides against the tightest adjacent gap of the commit
// order, so the fixture measures that gap itself instead of hard coding it.
function strictScheduleTimes(harness, model, commitTime = 0) {
    const attributes = materializedAttributes(modelAttributes(model), model.dynamicMask);
    const keys = model.positions.map((position, id) => {
        const offset = motionAt(attributes[id], commitTime);
        return depthKey(
            VIEW_PROJECTION,
            position[0] + offset.x,
            position[1] + offset.y,
            position[2] + offset.z,
        );
    });
    const order = Array.from(harness.lastFrame().depthIndex);
    let minGap = Infinity;
    for (let k = 1; k < order.length; k++) {
        minGap = Math.min(minGap, keys[order[k]] - keys[order[k - 1]]);
    }
    // The animated rows only translate, and the depth row of the view is a unit
    // vector, so one unit of time moves a row by this many depth keys.
    const keysPerTime =
        Math.fround(DYNAMIC_VELOCITY) *
        Math.hypot(VIEW_PROJECTION[2], VIEW_PROJECTION[6], VIEW_PROJECTION[10]) *
        4096;
    // The synthetic model stores scale code -2 on every axis, so a Gaussian
    // radius is exp(-2) world units and the budget counts radii.
    const footprint =
        Math.exp(-2) *
        Math.hypot(VIEW_PROJECTION[2], VIEW_PROJECTION[6], VIEW_PROJECTION[10]) *
        4096;
    const survive = (minGap / 2) / keysPerTime;
    return {
        minGap,
        keysPerTime,
        footprint,
        insideTime: survive / 2,
        outsideTime: survive * 3,
    };
}

test("adaptive scheduling reuses an order that the movement bound cannot break", async () => {
    const harness = createHarness();
    const { positions, dynamicMask } = latticeFixture();
    const model = buildModel({ positions, dynamicMask });

    await harness.load(model);
    await harness.sendAdaptiveSort(true, 0);
    await harness.sendView(VIEW_PROJECTION);
    assert.equal(harness.lastFrame().sortStats.adaptive, true);
    const { insideTime, outsideTime } = strictScheduleTimes(harness, model);
    const attributes = materializedAttributes(modelAttributes(model), model.dynamicMask);

    await harness.sendTime(insideTime, 1);
    assert.equal(harness.reuseFrames().length, 1, "a step inside the bound must reuse the order");
    const reuse = harness.reuseFrames()[0];
    assert.equal(reuse.dynamicRequestId, 1, "the reuse must acknowledge the request");
    assert.equal(reuse.depthIndex, undefined, "a reuse must not post a draw list");
    assert.equal(reuse.dynamicTime, insideTime);
    // Strict mode proves that no pair can swap, so the order the main thread
    // keeps drawing must still be the exact reference order at the new time.
    assert.deepEqual(
        harness.lastOrder(),
        expectedOrder(model.positions, attributes, insideTime, VIEW_PROJECTION),
    );

    await harness.sendTime(outsideTime, 2);
    assert.equal(harness.reuseFrames().length, 1, "a step beyond the bound must re-sort");
    const frame = harness.lastFrame();
    assert.equal(frame.dynamicRequestId, 2);
    assert.deepEqual(
        harness.lastOrder(),
        expectedOrder(model.positions, attributes, outsideTime, VIEW_PROJECTION),
    );
});

test("adaptive scheduling re-commits after the policy changes", async () => {
    const harness = createHarness();
    const { positions, dynamicMask } = latticeFixture();
    const model = buildModel({ positions, dynamicMask });

    await harness.load(model);
    await harness.sendView(VIEW_PROJECTION);

    // Adaptive scheduling is off by default: a time frame always posts an order.
    await harness.sendTime(0.05, 1);
    assert.equal(harness.reuseFrames().length, 0);
    assert.equal(harness.lastFrame().sortStats.adaptive, false);

    const { insideTime } = strictScheduleTimes(harness, model, 0.05);
    await harness.sendAdaptiveSort(true, 0);
    await harness.sendTime(0.05 + insideTime, 2);
    assert.equal(harness.lastFrame().dynamicRequestId, 2,
        "the switch must re-commit instead of reusing the pre-policy order");
    await harness.sendTime(0.05 + 2 * insideTime, 3);
    assert.equal(harness.reuseFrames().length, 1);
    assert.equal(harness.reuseFrames()[0].dynamicRequestId, 3);
});

test("the budget admits drift the strict bound refuses", async () => {
    const harness = createHarness();
    const { positions, dynamicMask } = latticeFixture();
    const model = buildModel({ positions, dynamicMask });

    await harness.load(model);
    await harness.sendAdaptiveSort(true, 0);
    await harness.sendView(VIEW_PROJECTION);
    const { keysPerTime, footprint } = strictScheduleTimes(harness, model);
    // A step that moves the animated rows by three quarters of a Gaussian
    // radius: past the strict gap bound, inside a budget of one radius.
    const step = (0.75 * footprint) / keysPerTime;
    const assertStep = async (budget, requestId) => {
        await harness.sendAdaptiveSort(true, budget);
        await harness.sendTime(step, requestId);
        assert.equal(harness.reuseFrames().length, 0, `budget ${budget} must re-commit`);
        await harness.sendTime(2 * step, requestId + 1);
    };

    await assertStep(0, 1);
    assert.equal(harness.reuseFrames().length, 0, "no pair may tie under a zero budget");
    assert.equal(harness.lastFrame().dynamicRequestId, 2);

    await assertStep(0.5, 3);
    assert.equal(harness.reuseFrames().length, 0, "half a radius is not enough");
    assert.equal(harness.lastFrame().dynamicRequestId, 4);

    await assertStep(1, 5);
    assert.equal(harness.reuseFrames().length, 1, "one radius covers the drift");
    assert.equal(harness.reuseFrames()[0].dynamicRequestId, 6);
});
