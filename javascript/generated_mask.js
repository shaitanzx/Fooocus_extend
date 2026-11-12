/*
 * Generated mask integration for Gradio Sketch.
 *
 * Gradio Sketch keeps hand-drawn strokes in lines[]. The generated mask is
 * stored only in canvas[key="mask"]. During undo(), Sketch clears that canvas
 * and redraws lines[], so the generated mask disappears together with the
 * first undone stroke.
 *
 * This script:
 *   1. writes the generated PNG into canvas[key="mask"];
 *   2. creates one synthetic anchor stroke so Gradio serializes value.mask;
 *   3. counts real user strokes separately from the synthetic anchor;
 *   4. restores the generated mask during undo of real strokes;
 *   5. lets the final undo remove the synthetic anchor and the generated mask.
 */
(function () {
    "use strict";

    const ROOT_SELECTOR = "#inpaint_mask_canvas";
    const MASK_SELECTOR = 'canvas[key="mask"]';
    const INTERFACE_SELECTOR = 'canvas[key="interface"]';
    const BRUSH_SELECTOR = 'input[aria-label="Brush radius"]';
    const states = new WeakMap();

    function sleep(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }

    function nextFrames(count) {
        return new Promise((resolve) => {
            let left = count;

            function frame() {
                if (left <= 0) {
                    resolve();
                    return;
                }
                left -= 1;
                window.requestAnimationFrame(frame);
            }

            window.requestAnimationFrame(frame);
        });
    }

    function appRoot() {
        return typeof gradioApp === "function" ? gradioApp() : document;
    }

    function getRoot() {
        return appRoot().querySelector(ROOT_SELECTOR);
    }

    async function getCanvases(timeoutMs = 5000) {
        const started = Date.now();

        while (Date.now() - started < timeoutMs) {
            const root = getRoot();
            const maskCanvas = root?.querySelector(MASK_SELECTOR);
            const interfaceCanvas = root?.querySelector(INTERFACE_SELECTOR);

            if (root && maskCanvas && interfaceCanvas) {
                return { root, maskCanvas, interfaceCanvas };
            }

            await sleep(50);
        }

        return null;
    }

    function getState(root) {
        let state = states.get(root);

        if (state) {
            return state;
        }

        state = {
            active: false,
            synthetic: false,
            realStrokeCount: 0,
            restoreOnClear: false,
            generatedImage: null,
            maskCanvas: null,
            interfaceCanvas: null,
            originalClearRect: null,
            listenersInstalled: false
        };

        states.set(root, state);
        return state;
    }

    function getMaskValue(value) {
        if (!value || typeof value !== "object") {
            return null;
        }

        return typeof value.mask === "string" && value.mask.length > 0
            ? value.mask
            : null;
    }

    function loadImage(url) {
        return new Promise((resolve, reject) => {
            const image = new Image();
            image.onload = () => resolve(image);
            image.onerror = () => reject(new Error("Cannot load generated mask"));
            image.src = url;
        });
    }

    function drawImageToMaskCanvas(state) {
        if (!state.generatedImage || !state.maskCanvas) {
            return false;
        }

        const canvas = state.maskCanvas;
        const ctx = canvas.getContext("2d");
        const image = state.generatedImage;

        ctx.save();
        ctx.globalCompositeOperation = "source-over";
        ctx.drawImage(
            image,
            0,
            0,
            image.naturalWidth || image.width,
            image.naturalHeight || image.height,
            0,
            0,
            canvas.width,
            canvas.height
        );
        ctx.restore();
        return true;
    }

    function installMaskClearHook(root, state, maskCanvas) {
        if (state.listenersInstalled && state.maskCanvas === maskCanvas) {
            return;
        }

        state.maskCanvas = maskCanvas;
        state.interfaceCanvas = root.querySelector(INTERFACE_SELECTOR);
        state.originalClearRect = maskCanvas.getContext("2d").clearRect;

        const ctx = maskCanvas.getContext("2d");
        const originalClearRect = ctx.clearRect.bind(ctx);

        ctx.clearRect = function (...args) {
            originalClearRect(...args);

            /*
             * Sketch.undo() calls clear_canvas(), which calls clearRect() on
             * the mask context before drawing the remaining lines. Restore the
             * generated layer at exactly this point, before trigger_on_change.
             */
            if (state.restoreOnClear) {
                state.restoreOnClear = false;
                drawImageToMaskCanvas(state);
            }
        };

        root.addEventListener("mousedown", (event) => {
            if (event.target !== state.interfaceCanvas) {
                return;
            }

            if (event.button === 0 && !state.synthetic && state.active) {
                state.realStrokeCount += 1;
                console.log(
                    "[GeneratedMask] real stroke started:",
                    state.realStrokeCount
                );
            }
        }, true);

        root.addEventListener("click", (event) => {
            const button = event.target.closest?.("button");
            if (!button) {
                return;
            }

            const label = button.getAttribute("aria-label");

            if (label === "Undo" && state.active) {
                if (state.realStrokeCount > 0) {
                    state.realStrokeCount -= 1;
                    state.restoreOnClear = true;
                    console.log(
                        "[GeneratedMask] undo real stroke; remaining:",
                        state.realStrokeCount
                    );
                } else {
                    /*
                     * No real strokes remain. This undo removes the synthetic
                     * anchor from Gradio's lines[] and must clear the mask.
                     */
                    state.restoreOnClear = false;
                    state.active = false;
                    state.generatedImage = null;
                    console.log(
                        "[GeneratedMask] final undo: generated mask removed"
                    );
                }
            }

            if (label === "Clear" && state.active) {
                state.restoreOnClear = false;
                state.active = false;
                state.realStrokeCount = 0;
                state.generatedImage = null;
                console.log("[GeneratedMask] clear: generated mask removed");
            }
        }, true);

        state.listenersInstalled = true;
    }

    function firstForegroundPoint(canvas) {
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

        for (let y = 0; y < canvas.height; y += 1) {
            for (let x = 0; x < canvas.width; x += 1) {
                const i = (y * canvas.width + x) * 4;
                if (data[i + 3] > 0 && (data[i] > 0 || data[i + 1] > 0 || data[i + 2] > 0)) {
                    return { x, y };
                }
            }
        }

        return null;
    }

    function toClientPoint(canvas, point) {
        const rect = canvas.getBoundingClientRect();
        return {
            clientX: rect.left + ((point.x + 0.5) / canvas.width) * rect.width,
            clientY: rect.top + ((point.y + 0.5) / canvas.height) * rect.height
        };
    }

    function sendMouseEvent(canvas, type, point, buttons) {
        canvas.dispatchEvent(new MouseEvent(type, {
            bubbles: true,
            cancelable: true,
            view: window,
            clientX: point.clientX,
            clientY: point.clientY,
            screenX: point.clientX,
            screenY: point.clientY,
            button: 0,
            buttons,
            detail: type === "mousedown" ? 1 : 0
        }));
    }

    async function emitSyntheticAnchor(state) {
        const pointOnMask = firstForegroundPoint(state.maskCanvas);

        if (!pointOnMask) {
            console.warn("[GeneratedMask] generated mask is empty");
            return false;
        }

        const point = toClientPoint(state.interfaceCanvas, pointOnMask);
        const brushInput = state.root?.querySelector(BRUSH_SELECTOR);
        const previousBrushValue = brushInput?.value;

        state.synthetic = true;

        try {
            /* The anchor is only a trigger; keep it one pixel wide. */
            if (brushInput) {
                brushInput.value = "1";
                brushInput.dispatchEvent(new Event("input", { bubbles: true }));
            }

            sendMouseEvent(state.interfaceCanvas, "mousedown", point, 1);
            sendMouseEvent(state.interfaceCanvas, "mousemove", point, 1);
            sendMouseEvent(state.interfaceCanvas, "mouseup", point, 0);
            await nextFrames(3);
        } finally {
            if (brushInput && previousBrushValue !== undefined) {
                brushInput.value = previousBrushValue;
                brushInput.dispatchEvent(new Event("input", { bubbles: true }));
            }
            state.synthetic = false;
        }

        return true;
    }

    async function applyGeneratedMask(value) {
        const canvases = await getCanvases();

        if (!canvases) {
            console.error("[GeneratedMask] mask/interface canvas not found");
            return value;
        }

        const { root, maskCanvas, interfaceCanvas } = canvases;
        const state = getState(root);
        state.maskCanvas = maskCanvas;
        state.interfaceCanvas = interfaceCanvas;
        installMaskClearHook(root, state, maskCanvas);

        const maskUrl = getMaskValue(value);
        if (maskUrl) {
            try {
                state.generatedImage = await loadImage(maskUrl);
                const ctx = maskCanvas.getContext("2d");
                ctx.save();
                ctx.globalCompositeOperation = "source-over";
                ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
                drawImageToMaskCanvas(state);
                ctx.restore();
            } catch (error) {
                console.error("[GeneratedMask] failed to draw mask:", error);
                return value;
            }
        }

        state.active = true;
        state.realStrokeCount = 0;
        state.restoreOnClear = false;

        await emitSyntheticAnchor(state);
        await sleep(50);
        await nextFrames(2);

        console.log("[GeneratedMask] generated mask synchronized");
        return value;
    }

    window.apply_generated_mask = async function (value) {
        return await applyGeneratedMask(value);
    };

    window.sync_generated_mask = async function () {
        return await applyGeneratedMask(null);
    };
})();
