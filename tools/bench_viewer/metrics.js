// Measurement plumbing for the browser bench in tools/bench_viewer/.
//
// Nothing here touches the DOM, WebGL or timers.  It owns the URL contract
// that describes one measured configuration, and it turns the raw samples
// collected by the page into the columns of a result table, so the aggregation
// stays testable from node (tests/bench_viewer.test.js).
(function (root) {
    "use strict";

    // Query string keys of the bench page.  The defaults mirror the N3DV scene
    // pages: depth correction at 10 Hz on a ten second clip.  `cam` selects a
    // frozen camera (a fixed view is what lets the scheduler act at all) or the
    // viewer's orbiting demo camera.
    const CONFIG_DEFAULTS = {
        model: "coffee.json",
        camera: "",
        cam: "fixed",
        sortFps: 10,
        adaptive: 1,
        budget: 0.5,
        renderScale: 1,
        smooth: 1,
        loop: 10,
        sec: 10,
        warm: 3,
        auto: 0,
        sweep: 0,
        si: 0,
        panel: 1,
        label: "",
    };

    const STRING_KEYS = ["model", "camera", "cam", "label"];
    const CONFIG_ORDER = Object.keys(CONFIG_DEFAULTS);

    // `?sweep=` is a "|" separated list of label:adaptive:budget:sortFps.  The
    // label may be empty, in which case it is derived from the other fields.
    const DEFAULT_SWEEP = [
        "fixed 30Hz:0:0:30",
        "adaptive r=0.25:1:0.25:30",
        "adaptive r=0.5:1:0.5:30",
        "adaptive r=1:1:1:30",
        "fixed 10Hz:0:0:10",
    ];

    // Result columns, in export order.  The ones marked `table` are also
    // printed by formatTable, which has to stay readable on a phone screen.
    const REPORT_COLUMNS = [
        { key: "label", label: "config", table: true },
        { key: "model", label: "model", table: true },
        { key: "camMode", label: "cam", table: true },
        { key: "sortFps", label: "sortHz", table: true },
        { key: "adaptive", label: "adapt", table: true },
        { key: "budget", label: "budget", table: true },
        { key: "smooth", label: "smooth" },
        { key: "loopSeconds", label: "loopS" },
        { key: "measuredSeconds", label: "sec" },
        { key: "renderSize", label: "render" },
        { key: "renderScale", label: "scale" },
        { key: "dpr", label: "dpr" },
        { key: "glRenderer", label: "gpu" },
        { key: "hardwareConcurrency", label: "cores" },
        { key: "deviceMemory", label: "memGB" },
        { key: "mobile", label: "mobile" },
        { key: "userAgent", label: "ua" },
        { key: "staticCount", label: "static" },
        { key: "dynamicCount", label: "dynamic" },
        { key: "vertexCount", label: "vertices" },
        { key: "frames", label: "frames" },
        { key: "framesPerSec", label: "fps", table: true },
        { key: "displayHz", label: "dispHz" },
        { key: "frameP50", label: "fP50" },
        { key: "frameP95", label: "fP95", table: true },
        { key: "frameP99", label: "fP99" },
        { key: "frameMax", label: "fMax" },
        { key: "jankPct", label: "jank%", table: true },
        { key: "stalls", label: "stalls" },
        { key: "busyP50", label: "busyP50" },
        { key: "busyP95", label: "busyP95" },
        { key: "busyMsPerSec", label: "busyMs/s", table: true },
        { key: "uploadCount", label: "uploads" },
        { key: "uploadPerSec", label: "up/s", table: true },
        { key: "uploadMbPerSec", label: "upMB/s", table: true },
        { key: "uploadMsPerSec", label: "upMs/s", table: true },
        { key: "uploadP95", label: "upP95" },
        { key: "workerMessages", label: "wMsg" },
        { key: "workerPerSec", label: "wMsg/s" },
        { key: "sortCount", label: "sorts", table: true },
        { key: "reuseCount", label: "reuse", table: true },
        { key: "reusePct", label: "reuse%", table: true },
        { key: "latencyP50", label: "latP50" },
        { key: "latencyP95", label: "latP95" },
        { key: "latencyMax", label: "latMax" },
        { key: "longTaskCount", label: "ltCount" },
        { key: "longTaskPerSec", label: "lt/s", table: true },
        { key: "longTaskMsPerSec", label: "ltMs/s", table: true },
        { key: "longTaskMax", label: "ltMax" },
        { key: "heapStartMB", label: "heap0" },
        { key: "heapEndMB", label: "heap1" },
        { key: "heapDeltaMB", label: "heapd" },
        { key: "batteryCharging", label: "chg" },
        { key: "batteryLevelStart", label: "bat0" },
        { key: "batteryLevelEnd", label: "bat1" },
        { key: "batteryPctPerHour", label: "bat%/h", table: true },
        { key: "note", label: "note" },
    ];

    // A frame that takes this much longer than the median frame was not just
    // slow, it dropped at least one refresh interval.
    const STALL_MS = 250;
    const JANK_FACTOR = 1.5;

    function round(value, digits) {
        if (!Number.isFinite(value)) return "";
        const factor = Math.pow(10, digits);
        return Math.round(value * factor) / factor;
    }

    function total(values) {
        let sum = 0;
        for (const value of values) sum += Number(value) || 0;
        return sum;
    }

    // Linear interpolation between the two samples that bracket the quantile.
    function percentileSorted(sorted, p) {
        const count = sorted.length;
        if (count === 0) return 0;
        if (count === 1) return sorted[0];
        const position = (count - 1) * Math.min(1, Math.max(0, p));
        const lower = Math.floor(position);
        const upper = Math.ceil(position);
        if (lower === upper) return sorted[lower];
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    function summarize(values) {
        const numbers = [];
        for (const value of values || []) {
            if (Number.isFinite(value)) numbers.push(value);
        }
        if (numbers.length === 0) {
            return { count: 0, mean: 0, p50: 0, p95: 0, p99: 0, max: 0 };
        }
        numbers.sort((a, b) => a - b);
        return {
            count: numbers.length,
            mean: total(numbers) / numbers.length,
            p50: percentileSorted(numbers, 0.5),
            p95: percentileSorted(numbers, 0.95),
            p99: percentileSorted(numbers, 0.99),
            max: numbers[numbers.length - 1],
        };
    }

    // Intervals between two presented frames, in milliseconds.  Gaps this long
    // mean the page was suspended (background tab, screen off, task switch) and
    // are counted separately instead of being averaged into the frame rate.
    function frameIntervals(frameTimes, stallMs = STALL_MS) {
        const intervals = [];
        let stalls = 0;
        const times = frameTimes || [];
        for (let i = 1; i < times.length; i++) {
            const delta = times[i] - times[i - 1];
            if (!(delta > 0)) continue;
            if (delta >= stallMs) {
                stalls++;
                continue;
            }
            intervals.push(delta);
        }
        return { intervals, stalls };
    }

    // Share of frames that arrived later than one and a half times the median
    // interval, which is the cheapest definition of a dropped frame that does
    // not have to know the display refresh rate.
    function jankShare(intervals, factor = JANK_FACTOR) {
        if (!intervals || intervals.length === 0) return 0;
        const sorted = Array.from(intervals).sort((a, b) => a - b);
        const median = percentileSorted(sorted, 0.5);
        if (!(median > 0)) return 0;
        let janky = 0;
        for (const interval of intervals) {
            if (interval > median * factor) janky++;
        }
        return janky / intervals.length;
    }

    function parseConfig(search) {
        const params = new URLSearchParams(search || "");
        const config = Object.assign({}, CONFIG_DEFAULTS);
        for (const key of CONFIG_ORDER) {
            const raw = params.get(key);
            if (raw === null) continue;
            if (STRING_KEYS.includes(key)) {
                config[key] = raw;
                continue;
            }
            const value = Number(raw);
            if (Number.isFinite(value)) config[key] = value;
        }
        if (!["fixed", "carousel"].includes(config.cam)) config.cam = "fixed";
        if (config.adaptive !== 1) config.adaptive = 0;
        if (!(config.budget >= 0)) config.budget = 0;
        if (!(config.sortFps >= 0)) config.sortFps = 0;
        if (!(config.renderScale > 0) || config.renderScale > 2) config.renderScale = 1;
        if (!(config.sec > 0)) config.sec = CONFIG_DEFAULTS.sec;
        if (!(config.warm >= 0)) config.warm = 0;
        if (!(config.loop > 0)) config.loop = CONFIG_DEFAULTS.loop;
        return config;
    }

    // Only the fields that differ from the defaults are written, so a shared
    // link stays short and a sweep URL shows exactly what it changes.
    function configToSearch(config, extra) {
        const params = new URLSearchParams();
        for (const key of CONFIG_ORDER) {
            const value = config[key];
            if (value === undefined || value === null || value === "") continue;
            if (String(value) === String(CONFIG_DEFAULTS[key])) continue;
            params.set(key, String(value));
        }
        for (const key of Object.keys(extra || {})) {
            params.set(key, String(extra[key]));
        }
        return params.toString();
    }

    function parseSweep(text) {
        const source = typeof text === "string" && text.trim().length > 0
            ? text
            : DEFAULT_SWEEP.join("|");
        return source
            .split("|")
            .filter((item) => item.trim().length > 0)
            .map((item) => {
                const fields = item.split(":");
                const adaptive = Number(fields[1]) === 1 ? 1 : 0;
                let budget = Number(fields[2]);
                let sortFps = Number(fields[3]);
                if (!(budget >= 0)) budget = 0;
                if (!(sortFps >= 0)) sortFps = CONFIG_DEFAULTS.sortFps;
                const derived = adaptive === 1
                    ? "adaptive r=" + budget + " @" + sortFps + "Hz"
                    : "fixed " + sortFps + "Hz";
                return {
                    label: (fields[0] || "").trim() || derived,
                    adaptive,
                    budget,
                    sortFps,
                };
            });
    }

    function applySweepEntry(config, entry) {
        return Object.assign({}, config, {
            label: entry.label,
            adaptive: entry.adaptive,
            budget: entry.budget,
            sortFps: entry.sortFps,
        });
    }

    // Reported only when the battery actually moved during the run.  A ten
    // second window on a phone usually shows no change at the percent
    // resolution the API exposes, and an invented zero would read as a result.
    function batteryRate(series) {
        if (!Array.isArray(series) || series.length < 2) return "";
        const first = series[0];
        const last = series[series.length - 1];
        if (typeof first.level !== "number" || typeof last.level !== "number") return "";
        if (first.charging || last.charging) return "";
        const drop = first.level - last.level;
        const span = last.t - first.t;
        if (!(span > 30000) || !(drop > 0)) return "";
        return round((drop * 100 * 3600000) / span, 2);
    }

    // samples = {
    //   startedAt, stoppedAt,            performance.now() bounds of the window
    //   frameTimes: number[],            rAF timestamps of presented frames
    //   frameBusy: number[],             ms the viewer's frame callback occupied
    //   uploads: [{bytes, ms}],          index buffer uploads on the main thread
    //   messages: [{t, kind, latency}],  worker replies, kind: sort|reuse|...
    //   posts: [{t, kind}],              worker requests, kind: view|time|...
    //   longTasks: [{ms}],               main thread long tasks
    //   heap: {start, end},              usedJSHeapSize in bytes, when available
    //   battery: {charging, series},     level samples during the window
    //   hidden, canvasResized,           counters that explain suspicious runs
    // }
    function buildReport(samples, meta) {
        const startedAt = Number(samples.startedAt) || 0;
        const stoppedAt = Number(samples.stoppedAt) || 0;
        const durationMs = Math.max(1, stoppedAt - startedAt);
        const seconds = durationMs / 1000;

        const frameTimes = samples.frameTimes || [];
        const measured = frameIntervals(frameTimes);
        const intervals = measured.intervals;
        const frameSummary = summarize(intervals);
        const busySummary = summarize(samples.frameBusy || []);
        const busyTotal = total(samples.frameBusy || []);

        const uploads = samples.uploads || [];
        const uploadMsSummary = summarize(uploads.map((entry) => entry.ms));
        const uploadBytes = total(uploads.map((entry) => entry.bytes));
        const uploadMs = total(uploads.map((entry) => entry.ms));

        const messages = samples.messages || [];
        const sorts = messages.filter((entry) => entry.kind === "sort").length;
        const reuses = messages.filter((entry) => entry.kind === "reuse").length;
        const latencySummary = summarize(
            messages.map((entry) => entry.latency).filter((value) => value >= 0),
        );

        const longTasks = (samples.longTasks || []).map((entry) => entry.ms);
        const longTaskSummary = summarize(longTasks);

        const heap = samples.heap || {};
        const battery = samples.battery || { series: [] };
        const reuseShare = sorts + reuses > 0 ? (100 * reuses) / (sorts + reuses) : 0;

        const notes = [];
        if (meta.dynamicCount === 0) notes.push("no animated gaussians");
        if (measured.stalls > 0) notes.push(measured.stalls + " stalls");
        if (samples.hidden) notes.push("page hidden");
        if (samples.canvasResized) notes.push("canvas resized");
        if (meta.adaptive === 1 && sorts > 0 && reuses === 0) {
            notes.push("adaptive on but no reuse");
        }
        if (battery.charging) notes.push("charging");

        const defined = (value) => value !== null && value !== undefined;

        return {
            label: meta.label || "",
            model: meta.model || "",
            camMode: meta.camMode || "",
            sortFps: meta.sortFps,
            adaptive: meta.adaptive,
            budget: meta.budget,
            smooth: meta.smooth,
            loopSeconds: meta.loopSeconds,
            measuredSeconds: round(seconds, 2),
            renderSize: meta.renderSize || "",
            renderScale: round(meta.renderScale, 2),
            dpr: round(meta.dpr, 2),
            glRenderer: meta.glRenderer || "",
            hardwareConcurrency: meta.hardwareConcurrency || "",
            deviceMemory: meta.deviceMemory || "",
            mobile: meta.mobile === true || meta.mobile === false ? (meta.mobile ? 1 : 0) : "",
            userAgent: meta.userAgent || "",
            staticCount: defined(meta.staticCount) ? meta.staticCount : "",
            dynamicCount: defined(meta.dynamicCount) ? meta.dynamicCount : "",
            vertexCount: defined(meta.vertexCount) ? meta.vertexCount : "",
            frames: frameTimes.length,
            framesPerSec: round(frameTimes.length / seconds, 1),
            displayHz: frameSummary.p50 > 0 ? round(1000 / frameSummary.p50, 1) : "",
            frameP50: round(frameSummary.p50, 2),
            frameP95: round(frameSummary.p95, 2),
            frameP99: round(frameSummary.p99, 2),
            frameMax: round(frameSummary.max, 2),
            jankPct: round(jankShare(intervals) * 100, 2),
            stalls: measured.stalls,
            busyP50: round(busySummary.p50, 3),
            busyP95: round(busySummary.p95, 3),
            busyMsPerSec: round(busyTotal / seconds, 1),
            uploadCount: uploads.length,
            uploadPerSec: round(uploads.length / seconds, 1),
            uploadMbPerSec: round(uploadBytes / (1024 * 1024) / seconds, 2),
            uploadMsPerSec: round(uploadMs / seconds, 2),
            uploadP95: round(uploadMsSummary.p95, 3),
            workerMessages: messages.length,
            workerPerSec: round(messages.length / seconds, 1),
            sortCount: sorts,
            reuseCount: reuses,
            reusePct: round(reuseShare, 1),
            latencyP50: round(latencySummary.p50, 2),
            latencyP95: round(latencySummary.p95, 2),
            latencyMax: round(latencySummary.max, 2),
            longTaskCount: longTasks.length,
            longTaskPerSec: round(longTasks.length / seconds, 2),
            longTaskMsPerSec: round(total(longTasks) / seconds, 1),
            longTaskMax: round(longTaskSummary.max, 1),
            heapStartMB: heap.start ? round(heap.start / (1024 * 1024), 1) : "",
            heapEndMB: heap.end ? round(heap.end / (1024 * 1024), 1) : "",
            heapDeltaMB: heap.start && heap.end
                ? round((heap.end - heap.start) / (1024 * 1024), 1)
                : "",
            batteryCharging: battery.charging === true || battery.charging === false
                ? (battery.charging ? 1 : 0)
                : "",
            batteryLevelStart: round(battery.levelStart * 100, 1),
            batteryLevelEnd: round(battery.levelEnd * 100, 1),
            batteryPctPerHour: batteryRate(battery.series),
            note: notes.join("; "),
        };
    }

    function formatCell(value) {
        if (value === "" || value === null || value === undefined) return "-";
        if (typeof value === "number") {
            if (!Number.isFinite(value)) return "-";
            if (Math.abs(value) >= 1000) return String(Math.round(value));
            if (Number.isInteger(value)) return String(value);
            return String(Math.round(value * 1000) / 1000);
        }
        return String(value);
    }

    function tableKeys(columns = REPORT_COLUMNS) {
        return columns.filter((column) => column.table === true).map((column) => column.key);
    }

    function columnLabel(key, columns = REPORT_COLUMNS) {
        const column = columns.find((entry) => entry.key === key);
        return column ? column.label : key;
    }

    // Fixed width text table, the same shape the node bench prints, so a phone
    // session can be read from the console or pasted into a paper draft.
    function formatTable(reports, columns = REPORT_COLUMNS) {
        const keys = tableKeys(columns);
        const headers = keys.map((key) => columnLabel(key, columns));
        const rows = reports.map((report) => keys.map((key) => formatCell(report[key])));
        const widths = headers.map((header, index) => {
            let width = header.length;
            for (const row of rows) width = Math.max(width, row[index].length);
            return width;
        });
        const line = (cells) => cells
            .map((cell, index) => index === 0
                ? cell.padEnd(widths[index])
                : cell.padStart(widths[index]))
            .join("  ")
            .trimEnd();
        return [line(headers)].concat(rows.map(line)).join("\n");
    }

    function csvValue(value) {
        const text = value === null || value === undefined ? "" : String(value);
        return /[",\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
    }

    function toCsv(reports, columns = REPORT_COLUMNS) {
        const keys = columns.map((column) => column.key);
        const lines = [keys.join(",")];
        for (const report of reports) {
            lines.push(keys.map((key) => csvValue(report[key])).join(","));
        }
        return lines.join("\n") + "\n";
    }

    function toJson(reports) {
        return JSON.stringify({ version: 1, reports }, null, 2);
    }

    root.BenchMetrics = {
        CONFIG_DEFAULTS,
        CONFIG_ORDER,
        DEFAULT_SWEEP,
        REPORT_COLUMNS,
        JANK_FACTOR,
        STALL_MS,
        applySweepEntry,
        buildReport,
        columnLabel,
        configToSearch,
        formatTable,
        frameIntervals,
        jankShare,
        parseConfig,
        parseSweep,
        percentileSorted,
        round,
        summarize,
        tableKeys,
        toCsv,
        toJson,
    };
})(typeof globalThis !== "undefined" ? globalThis : this);
