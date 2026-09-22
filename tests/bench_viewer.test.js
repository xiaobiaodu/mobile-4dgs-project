// Regression tests for the browser bench in tools/bench_viewer/.
//
// The page collects its samples from a real browser, but the aggregation, the
// URL contract and the exports are plain functions, so they run here against
// synthetic samples.  The numbers below are hand computed from the fixtures.

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const METRICS_JS = path.join(__dirname, "..", "tools", "bench_viewer", "metrics.js");

// metrics.js attaches itself to globalThis, which a VM context provides.
function loadMetrics() {
    const context = { URLSearchParams };
    vm.runInNewContext(fs.readFileSync(METRICS_JS, "utf8"), context, {
        filename: METRICS_JS,
    });
    return context.BenchMetrics;
}

const M = loadMetrics();

// metrics.js runs in another realm, so its arrays and objects do not share the
// host prototypes and have to be compared field by field.
function plain(value) {
    return JSON.parse(JSON.stringify(value));
}

function closeTo(actual, expected, epsilon = 1e-9) {
    assert.ok(
        Math.abs(actual - expected) < epsilon,
        "expected " + actual + " to be within " + epsilon + " of " + expected,
    );
}

test("config parses typed query values and clamps what is out of range", () => {
    const parsed = M.parseConfig(
        "?model=garden.json&adaptive=1&budget=-1&sortFps=abc&cam=bogus&sec=0&warm=-4&loop=0",
    );
    assert.equal(parsed.model, "garden.json");
    assert.equal(parsed.adaptive, 1);
    assert.equal(parsed.budget, 0, "a negative drift budget falls back to the exact policy");
    assert.equal(parsed.sortFps, 10, "an unparseable rate keeps the default");
    assert.equal(parsed.cam, "fixed", "an unknown camera mode falls back to the frozen view");
    assert.equal(parsed.sec, M.CONFIG_DEFAULTS.sec);
    assert.equal(parsed.warm, 0);
    assert.equal(parsed.loop, M.CONFIG_DEFAULTS.loop);

    const defaults = M.parseConfig("");
    for (const key of M.CONFIG_ORDER) {
        assert.deepEqual(defaults[key], M.CONFIG_DEFAULTS[key], key);
    }

    assert.equal(M.parseConfig("?adaptive=2").adaptive, 0, "adaptive is a flag");
    assert.equal(M.parseConfig("?si=3").si, 3);
    assert.equal(M.parseConfig("?renderScale=0.5").renderScale, 0.5);
    assert.equal(M.parseConfig("?renderScale=0").renderScale, 1, "zero is not a scale");
    assert.equal(M.parseConfig("?renderScale=5").renderScale, 1, "above 2 would upscale");
    assert.equal(M.parseConfig("?lod=0.5").lod, 0.5);
    assert.equal(M.parseConfig("?lod=0").lod, 1, "zero is not a share");
    assert.equal(M.parseConfig("?lod=2").lod, 1, "a level above one cannot add Gaussians");
    assert.equal(M.parseConfig("?minPixelRadius=2").minPixelRadius, 2);
    assert.equal(M.parseConfig("?minPixelRadius=0").minPixelRadius, 0,
        "zero is the cull being off");
    assert.equal(M.parseConfig("?minPixelRadius=-3").minPixelRadius, 0,
        "a negative radius would cull everything");
});

test("configToSearch omits defaults and round trips", () => {
    const query = "model=garden.json&adaptive=0&sec=5";
    const config = M.parseConfig("?" + query);
    assert.equal(M.configToSearch(config), query);
    assert.deepEqual(M.parseConfig("?" + M.configToSearch(config)), config);

    const extra = M.configToSearch(config, { sweep: 1, si: 2 });
    assert.match(extra, /(^|&)sweep=1(&|$)/);
    assert.match(extra, /(^|&)si=2(&|$)/);
    assert.equal(M.parseConfig("?" + extra).si, 2);
});

test("sweep spec parses entries and ignores empty ones", () => {
    assert.equal(M.parseSweep("").length, M.DEFAULT_SWEEP.length);

    const entries = M.parseSweep("tight:1:0:30|loose:1:2:10|");
    assert.equal(entries.length, 2);
    assert.deepEqual(plain(entries[0]), {
        label: "tight",
        adaptive: 1,
        budget: 0,
        sortFps: 30,
    });
    assert.equal(entries[1].budget, 2);
    assert.equal(entries[1].sortFps, 10);

    const derived = M.parseSweep(":1:0.5:30|:0::60|")[1];
    assert.equal(derived.label, "fixed 60Hz");
    assert.equal(derived.budget, 0, "a missing budget is the exact policy");
    assert.equal(M.parseSweep(":1:0.5:30|")[0].label, "adaptive r=0.5 @30Hz");

    const applied = M.applySweepEntry(M.parseConfig("?model=coffee.json"), {
        label: "adaptive r=1",
        adaptive: 1,
        budget: 1,
        sortFps: 15,
    });
    assert.equal(applied.model, "coffee.json", "a sweep entry only changes the policy");
    assert.equal(applied.budget, 1);
    assert.equal(applied.sortFps, 15);
});

test("percentiles interpolate and summarize known samples", () => {
    const sorted = [1, 2, 3, 4];
    assert.equal(M.percentileSorted(sorted, 0), 1);
    assert.equal(M.percentileSorted(sorted, 0.5), 2.5);
    assert.equal(M.percentileSorted(sorted, 1), 4);
    assert.equal(M.percentileSorted([], 0.5), 0);
    assert.equal(M.percentileSorted([7], 0.5), 7);

    const summary = M.summarize([1, 2, 3, 4]);
    assert.equal(summary.count, 4);
    assert.equal(summary.mean, 2.5);
    assert.equal(summary.p50, 2.5);
    closeTo(summary.p95, 3.85);
    assert.equal(summary.max, 4);
    assert.equal(M.summarize([]).max, 0);
    assert.equal(M.summarize([Number.NaN, 5]).p50, 5, "non finite samples are dropped");
});

test("frame intervals separate suspension stalls from jank", () => {
    const measured = M.frameIntervals([0, 16, 32, 400, 416]);
    assert.deepEqual(plain(measured.intervals), [16, 16, 16]);
    assert.equal(measured.stalls, 1);

    assert.equal(M.jankShare([16, 16, 16]), 0);
    assert.equal(M.jankShare([16, 50, 16, 16]), 0.25);
    assert.equal(M.jankShare([]), 0);
});

// Counts the fields of one CSV line, honouring the quoted form that labels
// containing a separator or a quote are written in.
function countCsvFields(line) {
    let fields = 1;
    let quoted = false;
    for (let i = 0; i < line.length; i++) {
        const character = line[i];
        if (character === "\"") {
            if (quoted && line[i + 1] === "\"") i++;
            else quoted = !quoted;
        } else if (character === "," && !quoted) {
            fields++;
        }
    }
    return fields;
}

// Two seconds of playback on a 60 Hz phone: seven intervals, one of them a
// dropped frame, two index uploads and a 60 ms long task.
function fixture(overrides) {
    const samples = {
        startedAt: 1000,
        stoppedAt: 3000,
        frameTimes: [1000, 1016, 1033, 1050, 1100, 1116, 1133, 1150],
        frameBusy: [1, 2, 3],
        uploads: [{ bytes: 524288, ms: 0.4 }, { bytes: 524288, ms: 0.6 }],
        messages: [
            { t: 1010, kind: "sort", latency: 4 },
            { t: 1200, kind: "reuse", latency: 2 },
            { t: 1400, kind: "reuse", latency: -1 },
        ],
        posts: [{ t: 1005, kind: "time" }],
        longTasks: [{ ms: 60 }],
        heap: { start: 10485760, end: 12582912 },
        battery: { charging: false, levelStart: 0.9, levelEnd: 0.9, series: [] },
        hidden: false,
        canvasResized: false,
    };
    return Object.assign(samples, overrides);
}

const META = {
    label: "fixed 30Hz",
    model: "coffee.json",
    camMode: "fixed",
    sortFps: 30,
    adaptive: 0,
    budget: 0,
    smooth: 1,
    loopSeconds: 10,
    renderSize: "1280x720",
    dpr: 2,
    glRenderer: "Adreno",
    hardwareConcurrency: 8,
    deviceMemory: 4,
    mobile: true,
    userAgent: "ua",
    staticCount: 100,
    dynamicCount: 5,
    vertexCount: 105,
};

test("buildReport turns the samples into the exported columns", () => {
    const report = M.buildReport(fixture(), META);
    assert.equal(report.label, "fixed 30Hz");
    assert.equal(report.measuredSeconds, 2);
    assert.equal(report.mobile, 1);
    assert.equal(report.frames, 8);
    assert.equal(report.framesPerSec, 4);
    assert.equal(report.displayHz, 58.8);
    assert.equal(report.frameP50, 17);
    assert.equal(report.frameP95, 40.1);
    assert.equal(report.frameMax, 50);
    assert.equal(report.jankPct, 14.29);
    assert.equal(report.stalls, 0);
    assert.equal(report.busyP50, 2);
    assert.equal(report.busyMsPerSec, 3);
    assert.equal(report.uploadCount, 2);
    assert.equal(report.uploadPerSec, 1);
    assert.equal(report.uploadMbPerSec, 0.5);
    assert.equal(report.uploadMsPerSec, 0.5);
    assert.equal(report.sortCount, 1);
    assert.equal(report.reuseCount, 2);
    assert.equal(report.reusePct, 66.7);
    assert.equal(report.workerMessages, 3);
    assert.equal(report.latencyP50, 3);
    assert.equal(report.latencyMax, 4);
    assert.equal(report.longTaskPerSec, 0.5);
    assert.equal(report.longTaskMsPerSec, 30);
    assert.equal(report.heapDeltaMB, 2);
    assert.equal(report.batteryLevelStart, 90);
    assert.equal(report.batteryPctPerHour, "");
    assert.equal(report.note, "");
    for (const column of M.REPORT_COLUMNS) {
        assert.ok(column.key in report, "missing column " + column.key);
    }
});

test("buildReport explains suspicious runs in the note column", () => {
    const report = M.buildReport(
        fixture({
            messages: [{ t: 1, kind: "sort", latency: 1 }],
            frameTimes: [1000, 1016, 1600],
            hidden: true,
            canvasResized: true,
            battery: { charging: true, levelStart: 0.5, levelEnd: 0.5, series: [] },
        }),
        Object.assign({}, META, { adaptive: 1, dynamicCount: 0 }),
    );
    assert.match(report.note, /no animated gaussians/);
    assert.match(report.note, /1 stalls/);
    assert.match(report.note, /page hidden/);
    assert.match(report.note, /canvas resized/);
    assert.match(report.note, /adaptive on but no reuse/);
    assert.match(report.note, /charging/);
    assert.equal(report.reusePct, 0);
    assert.equal(report.stalls, 1);
});

test("battery rate is reported only when the level actually moved", () => {
    const series = (start, end, charging = false) => [
        { t: 0, level: start, charging },
        { t: 120000, level: end, charging },
    ];
    const moved = M.buildReport(
        fixture({
            battery: {
                charging: false,
                levelStart: 0.9,
                levelEnd: 0.85,
                series: series(0.9, 0.85),
            },
        }),
        META,
    );
    assert.equal(moved.batteryPctPerHour, 150);

    const flat = M.buildReport(
        fixture({
            battery: {
                charging: false,
                levelStart: 0.9,
                levelEnd: 0.9,
                series: series(0.9, 0.9),
            },
        }),
        META,
    );
    assert.equal(flat.batteryPctPerHour, "");

    const charging = M.buildReport(
        fixture({
            battery: {
                charging: true,
                levelStart: 0.9,
                levelEnd: 0.85,
                series: series(0.9, 0.85, true),
            },
        }),
        META,
    );
    assert.equal(charging.batteryPctPerHour, "", "a charging phone says nothing about energy");
});

test("csv and table exports stay rectangular and quotable", () => {
    const base = M.buildReport(fixture(), META);
    const quoted = Object.assign({}, base, { label: "a,\"b\"" });
    const lines = M.toCsv([base, quoted]).trimEnd().split("\n");
    assert.equal(lines.length, 3);
    const headerCount = countCsvFields(lines[0]);
    assert.equal(headerCount, M.REPORT_COLUMNS.length);
    assert.equal(countCsvFields(lines[1]), headerCount);
    assert.equal(countCsvFields(lines[2]), headerCount);
    assert.match(lines[2], /"a,""b"""/);

    // The label identifies a row, so it has to survive into the text table.
    const rows = M.formatTable([base, quoted]).split("\n");
    assert.equal(rows.length, 3);
    assert.match(rows[0], /^config\s+model\s/);
    assert.match(rows[1], /^fixed 30Hz\s/);
    assert.match(rows[2], /^a,"b"\s/);

    const json = JSON.parse(M.toJson([base]));
    assert.equal(json.reports.length, 1);
    assert.equal(json.reports[0].label, "fixed 30Hz");
});
