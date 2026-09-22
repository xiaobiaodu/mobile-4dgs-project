// Self test for the bench harness.
//
// It replays synthetic worker traffic through window.BenchView, so the harness
// can be checked on a new browser or device without downloading a model: the
// page either reports SMOKE PASS or names the checks that failed.  Run it with
// `?model=...&adaptive=1&budget=0.5&sortFps=30&warm=0&sec=2&panel=1`; the panel
// markup in smoke.html mirrors index.html.
(function () {
    "use strict";

    const output = document.getElementById("bench-export");
    const checks = [];
    const lines = [];

    function check(name, condition, detail) {
        checks.push((condition ? "PASS " : "FAIL ") + name +
            (detail === undefined ? "" : "  [" + detail + "]"));
    }

    function flush(summary) {
        const failed = checks.filter((line) => line.startsWith("FAIL"));
        if (summary) lines.unshift(summary);
        const text = lines.join("\n") + "\n\n" + checks.join("\n") + "\n";
        if (output) output.value = text;
        // A textarea value is a property, so it does not survive --dump-dom;
        // the hint element carries the same text for headless runs.
        const hint = document.getElementById("bench-hint");
        if (hint) hint.textContent = text;
        document.title = failed.length === 0 ? "SMOKE PASS" : "SMOKE FAIL";
    }

    function delay(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }

    async function waitFor(condition, timeoutMs) {
        const deadline = performance.now() + timeoutMs;
        while (performance.now() < deadline) {
            if (condition()) return true;
            await delay(50);
        }
        return condition();
    }

    // A scratch context so the index upload hook sees a real bufferData call.
    function scratchContext() {
        const canvas = document.createElement("canvas");
        canvas.width = 8;
        canvas.height = 8;
        return canvas.getContext("webgl2") || canvas.getContext("webgl");
    }

    async function run() {
        const bench = window.BenchView;
        if (!bench) {
            check("BenchView exists", false);
            flush("harness did not load");
            return;
        }
        check("BenchView exists", true);

        const query = new URLSearchParams(location.search);
        const config = bench.config;
        const viewer = window.FLUX_GS_CONFIG || {};
        check("url model reaches the viewer", viewer.defaultModel === config.model, String(viewer.defaultModel));
        check("url sort rate reaches the viewer", viewer.dynamicSortFps === config.sortFps, String(viewer.dynamicSortFps));
        check("url policy reaches the viewer",
            viewer.adaptiveSort === (config.adaptive === 1),
            String(viewer.adaptiveSort));
        check("url budget reaches the viewer",
            viewer.adaptiveSortBudget === (config.adaptive === 1 ? config.budget : 0),
            String(viewer.adaptiveSortBudget));
        check("url loop reaches the viewer", viewer.dynamicLoopSeconds === config.loop, String(viewer.dynamicLoopSeconds));
        check("panel built its controls",
            document.querySelectorAll("#bench-controls select").length >= 5,
            document.querySelectorAll("#bench-controls select").length + " selects");

        // The scene message a gated model produces, then one committed order.
        bench.observe({
            dynamic: {
                enabled: true,
                staticCount: 900,
                dynamicCount: 100,
                adaptiveSort: config.adaptive === 1,
                adaptiveSortBudget: config.budget,
            },
        });
        bench.observe({ depthIndex: new Uint32Array([0, 1, 2, 3]), vertexCount: 4 });
        check("harness becomes ready", await waitFor(() => bench.state.ready, 3000));
        check("scene split recorded",
            bench.state.scene.staticCount === 900 && bench.state.scene.dynamicCount === 100,
            bench.state.scene.staticCount + "/" + bench.state.scene.dynamicCount);

        const gl = scratchContext();
        check("webgl context available", Boolean(gl), gl ? String(gl.getParameter(gl.VERSION)) : "none");

        let stop = false;
        let frames = 0;
        const pump = () => {
            frames++;
            if (!stop) requestAnimationFrame(pump);
        };
        requestAnimationFrame(pump);

        // Every third request re-sorts and re-uploads the index buffer, the
        // other two are answered with a reused order.
        async function driveUntilReports(target) {
            const started = bench.start();
            check("run started", started instanceof Promise);
            let request = 0;
            while (!stop && bench.results().length < target) {
                request++;
                bench.post({ time: request / 60, dynamicRequestId: request });
                await delay(20);
                if (request % 3 === 0) {
                    bench.observe({
                        depthIndex: new Uint32Array([1, 2, 3, 0]),
                        vertexCount: 4,
                        dynamicRequestId: request,
                        dynamicTime: request / 60,
                    });
                    if (gl) {
                        gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
                        gl.bufferData(0x8892, new Uint32Array(4096), 0x88e8);
                    }
                } else {
                    bench.observe({
                        reusedOrder: true,
                        vertexCount: 4,
                        dynamicRequestId: request,
                        dynamicTime: request / 60,
                    });
                }
            }
            await started;
            return request;
        }

        const firstRequests = await driveUntilReports(1);
        const reports = bench.results();
        check("one report for one run", reports.length === 1, String(reports.length));
        const report = reports[0] || {};
        check("frames captured", report.frames > 0, String(report.frames) + " raf callbacks");
        check("uploads captured", report.uploadCount > 0, String(report.uploadCount));
        check("sort replies captured", report.sortCount > 0, String(report.sortCount));
        check("reuse replies captured", report.reuseCount > 0, String(report.reuseCount));
        check("reuse share in range", report.reusePct > 0 && report.reusePct < 100, String(report.reusePct));
        check("upload bytes counted", report.uploadMbPerSec > 0, String(report.uploadMbPerSec));
        check("measured window honoured", report.measuredSeconds > 0.5, String(report.measuredSeconds));
        check("csv has one data row", bench.csv().trimEnd().split("\n").length === 2);
        check("table rendered the result",
            document.getElementById("bench-table").textContent.indexOf(report.label) >= 0,
            report.label);

        // A second run must append to the table and to the export.
        await driveUntilReports(2);
        check("second run appended a report", bench.results().length === 2);
        check("csv has two data rows", bench.csv().trimEnd().split("\n").length === 3);
        lines.push(bench.table());
        flush("replayed " + firstRequests + " synthetic depth requests in the first run");
    }

    run().catch((error) => {
        check("harness threw", false, String(error && error.stack ? error.stack : error));
        flush("exception during the smoke run");
    });
})();
