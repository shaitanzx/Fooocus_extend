/*
 * Synchronize a generated mask with Gradio's Sketch component.
 *
 * The generated mask is drawn into canvas[key="mask"]. Then a minimal
 * synthetic mouse stroke is sent to canvas[key="interface"]. Gradio's
 * normal Sketch handlers process that stroke and call trigger_on_change(),
 * so the current mask canvas is serialized into value.mask.
 */
(function () {
    "use strict";

    const ROOT_SELECTOR = "#inpaint_mask_canvas";
    const MASK_SELECTOR = 'canvas[key="mask"]';
    const INTERFACE_SELECTOR = 'canvas[key="interface"]';
    const BRUSH_SELECTOR = 'input[aria-label="Brush radius"]';

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

    function getRoot() {
        const app = typeof gradioApp === "function"
            ? gradioApp()
            : document;

        return app.querySelector(ROOT_SELECTOR);
    }

    async function getCanvases(timeoutMs = 5000) {
        const started = Date.now();

        while (Date.now() - started < timeoutMs) {
            const root = getRoot();
            const maskCanvas = root && root.querySelector(MASK_SELECTOR);
            const interfaceCanvas = root && root.querySelector(INTERFACE_SELECTOR);

            if (maskCanvas && interfaceCanvas) {
                return { root, maskCanvas, interfaceCanvas };
            }

            await sleep(50);
        }

        return null;
    }

    function getMaskValue(value) {
        if (!value || typeof value !== "object") {
            return null;
        }

        if (typeof value.mask === "string" && value.mask.length > 0) {
            return value.mask;
        }

        return null;
    }

    function loadImage(url) {
        return new Promise((resolve, reject) => {
            const image = new Image();

            image.onload = () => resolve(image);
            image.onerror = () => reject(new Error("Cannot load generated mask"));
            image.src = url;
        });
    }

    function getFirstForegroundPoint(canvas) {
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

        for (let y = 0; y < canvas.height; y += 1) {
            for (let x = 0; x < canvas.width; x += 1) {
                const index = (y * canvas.width + x) * 4;
                const alpha = data[index + 3];
                const red = data[index];
                const green = data[index + 1];
                const blue = data[index + 2];

                if (alpha > 0 && (red > 0 || green > 0 || blue > 0)) {
                    return { x, y };
                }
            }
        }

        return null;
    }

    function canvasToClient(canvas, point) {
        const rect = canvas.getBoundingClientRect();

        return {
            clientX: rect.left + ((point.x + 0.5) / canvas.width) * rect.width,
            clientY: rect.top + ((point.y + 0.5) / canvas.height) * rect.height
        };
    }

    function dispatchMouse(target, type, point, buttons, button) {
        const event = new MouseEvent(type, {
            bubbles: true,
            cancelable: true,
            view: window,
            clientX: point.clientX,
            clientY: point.clientY,
            screenX: point.clientX,
            screenY: point.clientY,
            button,
            buttons,
            detail: type === "mousedown" ? 1 : 0
        });

        target.dispatchEvent(event);
    }

    async function emitSyntheticStroke(maskCanvas, interfaceCanvas) {
        const foreground = getFirstForegroundPoint(maskCanvas);

        if (!foreground) {
            console.warn("[GeneratedMask] Mask canvas is empty");
            return false;
        }

        const point = canvasToClient(interfaceCanvas, foreground);
        const brushInput = interfaceCanvas
            .closest(ROOT_SELECTOR)
            ?.querySelector(BRUSH_SELECTOR);

        let previousBrushValue = null;
        let brushChanged = false;

        /*
         * Use a one-pixel synthetic stroke. The real generated mask is already
         * present on maskCanvas, so this stroke exists only to invoke Gradio's
         * normal saveLine()/trigger_on_change() path.
         */
        if (brushInput) {
            previousBrushValue = brushInput.value;
            brushInput.value = "1";
            brushInput.dispatchEvent(new Event("input", { bubbles: true }));
            brushChanged = true;
        }

        dispatchMouse(interfaceCanvas, "mousedown", point, 1, 0);
        dispatchMouse(interfaceCanvas, "mousemove", point, 1, 0);
        dispatchMouse(interfaceCanvas, "mouseup", point, 0, 0);

        await nextFrames(3);

        if (brushChanged) {
            brushInput.value = previousBrushValue;
            brushInput.dispatchEvent(new Event("input", { bubbles: true }));
        }

        return true;
    }

    async function drawMaskToCanvas(maskCanvas, maskUrl) {
        const image = await loadImage(maskUrl);
        const ctx = maskCanvas.getContext("2d");

        ctx.save();
        ctx.globalCompositeOperation = "source-over";
        ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
        ctx.drawImage(
            image,
            0,
            0,
            image.naturalWidth || image.width,
            image.naturalHeight || image.height,
            0,
            0,
            maskCanvas.width,
            maskCanvas.height
        );
        ctx.restore();
    }

    async function applyGeneratedMask(value) {
        const canvases = await getCanvases();

        if (!canvases) {
            console.error(
                "[GeneratedMask] Cannot find #inpaint_mask_canvas " +
                "mask/interface canvases"
            );
            return value;
        }

        const maskUrl = getMaskValue(value);

        /*
         * When called after a Gradio output update, the mask may already be
         * visible on the canvas. If value.mask is present, redraw it so the
         * canvas and value are guaranteed to use the same image.
         */
        if (maskUrl) {
            try {
                await drawMaskToCanvas(canvases.maskCanvas, maskUrl);
                await nextFrames(2);
            } catch (error) {
                console.error("[GeneratedMask] Failed to draw mask", error);
                return value;
            }
        }

        const synced = await emitSyntheticStroke(
            canvases.maskCanvas,
            canvases.interfaceCanvas
        );

        if (synced) {
            await sleep(50);
            await nextFrames(2);
            console.log(
                "[GeneratedMask] Canvas mask synchronized with Gradio"
            );
        }

        return value;
    }

    /*
     * Use this as Gradio's _js callback:
     *
     * .then(
     *     fn=lambda x: x,
     *     inputs=inpaint_mask_image,
     *     outputs=inpaint_mask_image,
     *     _js="apply_generated_mask",
     *     queue=False
     * )
     */
    window.apply_generated_mask = async function (value) {
        return await applyGeneratedMask(value);
    };

    /*
     * This variant is useful when the previous Gradio callback already placed
     * the result on the canvas and no input value is supplied to _js.
     */
    window.sync_generated_mask = async function () {
        return await applyGeneratedMask(null);
    };
})();
