(function () {
    "use strict";

    let virtualDrawing = false;
    let initializedRoot = null;

    function getRoot() {
        return gradioApp().querySelector(
            "#inpaint_mask_canvas"
        );
    }

    function getElements() {
        const root = getRoot();

        if (!root) {
            return null;
        }

        return {
            root,

            maskCanvas: root.querySelector(
                'canvas[key="mask"]'
            ),

            interfaceCanvas: root.querySelector(
                'canvas[key="interface"]'
            ),

            drawingCanvas: root.querySelector(
                'canvas[key="drawing"]'
            ),

            clearButton: root.querySelector(
                'button[aria-label="Clear"], ' +
                '[role="button"][aria-label="Clear"]'
            )
        };
    }

    function getSource(value) {
        if (!value || typeof value !== "object") {
            return null;
        }

        const mask = value.mask;

        if (!mask) {
            return null;
        }

        if (typeof mask === "string") {
            return mask;
        }

        if (typeof mask === "object") {
            return (
                mask.url ||
                mask.path ||
                mask.name ||
                null
            );
        }

        return null;
    }

    function delay(milliseconds) {
        return new Promise((resolve) => {
            setTimeout(resolve, milliseconds);
        });
    }

    function loadImage(source) {
        return new Promise((resolve, reject) => {
            const image = new Image();

            image.onload = function () {
                resolve(image);
            };

            image.onerror = function (error) {
                reject(error);
            };

            image.src = source;
        });
    }

    function getBrushRadius(canvas) {
        /*
         * Пытаемся взять реальный размер кисти из настроек Gradio.
         */
        const root = getRoot();

        if (root) {
            const brushInput = root.querySelector(
                'input[aria-label="Brush radius"]'
            );

            if (brushInput) {
                const value = Number(
                    brushInput.value
                );

                if (
                    Number.isFinite(value) &&
                    value > 0
                ) {
                    return value / 2;
                }
            }
        }

        /*
         * Запасной размер кисти.
         */
        return Math.max(
            8,
            Math.min(
                canvas.width,
                canvas.height
            ) * 0.025
        );
    }

    function normalizeMaskToCanvas(
        image,
        width,
        height
    ) {
        /*
         * Временный canvas. Он НЕ является
         * canvas[key="mask"] компонента.
         */
        const sourceCanvas =
            document.createElement("canvas");

        sourceCanvas.width = width;
        sourceCanvas.height = height;

        const sourceContext =
            sourceCanvas.getContext("2d", {
                willReadFrequently: true
            });

        if (!sourceContext) {
            return null;
        }

        sourceContext.clearRect(
            0,
            0,
            width,
            height
        );

        sourceContext.drawImage(
            image,
            0,
            0,
            width,
            height
        );

        const imageData = sourceContext.getImageData(
            0,
            0,
            width,
            height
        );

        const pixels = imageData.data;

        /*
         * Бинарная маска:
         *
         * true  — здесь нужно рисовать кистью;
         * false — прозрачная область.
         */
        const active = new Uint8Array(
            width * height
        );

        let hasAlpha = false;

        for (let i = 3; i < pixels.length; i += 4) {
            if (pixels[i] > 5) {
                hasAlpha = true;
                break;
            }
        }

        for (
            let y = 0;
            y < height;
            y += 1
        ) {
            for (
                let x = 0;
                x < width;
                x += 1
            ) {
                const pixelIndex =
                    (y * width + x) * 4;

                const red = pixels[pixelIndex];
                const green = pixels[pixelIndex + 1];
                const blue = pixels[pixelIndex + 2];
                const alpha = pixels[pixelIndex + 3];

                /*
                 * Для RGBA-маски используем alpha.
                 * Если alpha полностью отсутствует,
                 * используем яркость RGB.
                 */
                const isMaskPixel = hasAlpha
                    ? alpha > 10
                    : (
                        red > 10 ||
                        green > 10 ||
                        blue > 10
                    );

                if (isMaskPixel) {
                    active[y * width + x] = 1;
                }
            }
        }

        return {
            canvas: sourceCanvas,
            active,
            width,
            height
        };
    }

    function getRunsForRow(
        active,
        width,
        y,
        minimumRunLength
    ) {
        const runs = [];
        let runStart = -1;

        for (
            let x = 0;
            x < width;
            x += 1
        ) {
            const activePixel =
                active[y * width + x] !== 0;

            if (activePixel && runStart < 0) {
                runStart = x;
            }

            const isLastPixel =
                x === width - 1;

            if (
                runStart >= 0 &&
                (!activePixel || isLastPixel)
            ) {
                const runEnd = activePixel && isLastPixel
                    ? x
                    : x - 1;

                if (
                    runEnd - runStart + 1 >=
                    minimumRunLength
                ) {
                    runs.push({
                        start: runStart,
                        end: runEnd
                    });
                }

                runStart = -1;
            }
        }

        return runs;
    }

    function buildStrokePoints(
        normalizedMask,
        brushRadius
    ) {
        const {
            active,
            width,
            height
        } = normalizedMask;

        /*
         * Чем меньше rowStep, тем точнее заполнение,
         * но тем больше будет mousemove-событий.
         */
        const rowStep = Math.max(
            2,
            Math.floor(brushRadius * 0.65)
        );

        const minimumRunLength = Math.max(
            2,
            Math.floor(brushRadius * 0.5)
        );

        const points = [];
        let rowNumber = 0;

        for (
            let y = 0;
            y < height;
            y += rowStep
        ) {
            const runs = getRunsForRow(
                active,
                width,
                y,
                minimumRunLength
            );

            if (runs.length === 0) {
                continue;
            }

            const leftToRight =
                rowNumber % 2 === 0;

            const orderedRuns =
                leftToRight
                    ? runs
                    : runs.slice().reverse();

            for (
                let i = 0;
                i < orderedRuns.length;
                i += 1
            ) {
                const run = orderedRuns[i];

                if (leftToRight) {
                    points.push({
                        x: run.start,
                        y
                    });

                    points.push({
                        x: run.end,
                        y
                    });
                } else {
                    points.push({
                        x: run.end,
                        y
                    });

                    points.push({
                        x: run.start,
                        y
                    });
                }
            }

            rowNumber += 1;
        }

        return points;
    }

    function canvasPointToClientPoint(
        canvas,
        point
    ) {
        const rect =
            canvas.getBoundingClientRect();

        return {
            clientX:
                rect.left +
                point.x *
                rect.width /
                canvas.width,

            clientY:
                rect.top +
                point.y *
                rect.height /
                canvas.height
        };
    }

    function createMouseEvent(
        type,
        canvas,
        point,
        buttons
    ) {
        const clientPoint =
            canvasPointToClientPoint(
                canvas,
                point
            );

        return new MouseEvent(type, {
            bubbles: true,
            cancelable: true,
            view: window,

            clientX: clientPoint.clientX,
            clientY: clientPoint.clientY,

            button: 0,
            buttons,

            /*
             * Для координатных расчётов Gradio
             * также задаём screen/client значения.
             */
            screenX: clientPoint.clientX,
            screenY: clientPoint.clientY
        });
    }

    async function clearPreviousGradioLines(
        elements
    ) {
        /*
         * Кнопка Clear вызывает штатный
         * sketch.clear_mask(), который очищает
         * внутренний массив lines[].
         */
        if (elements.clearButton) {
            elements.clearButton.click();

            /*
             * Даём Svelte/Gradio завершить redraw.
             */
            await delay(100);
        }
    }

    async function drawMaskAsOneGradioLine(
        maskSource
    ) {
        if (virtualDrawing) {
            console.warn(
                "[VirtualMask] Drawing is already running."
            );

            return false;
        }

        const elements = getElements();

        if (
            !elements ||
            !elements.maskCanvas ||
            !elements.interfaceCanvas
        ) {
            console.error(
                "[VirtualMask] Required canvases not found."
            );

            return false;
        }

        virtualDrawing = true;

        try {
            const maskImage =
                await loadImage(maskSource);

            const maskCanvas =
                elements.maskCanvas;

            const interfaceCanvas =
                elements.interfaceCanvas;

            const brushRadius =
                getBrushRadius(maskCanvas);

            const normalizedMask =
                normalizeMaskToCanvas(
                    maskImage,
                    maskCanvas.width,
                    maskCanvas.height
                );

            if (!normalizedMask) {
                return false;
            }

            const points = buildStrokePoints(
                normalizedMask,
                brushRadius
            );

            if (points.length < 2) {
                console.warn(
                    "[VirtualMask] Mask contains no drawable area."
                );

                return false;
            }

            console.log(
                "[VirtualMask] Starting one synthetic stroke.",
                {
                    points: points.length,
                    brushRadius,
                    width: maskCanvas.width,
                    height: maskCanvas.height
                }
            );

            /*
             * Один mousedown.
             *
             * С этого момента Gradio создаёт один
             * внутренний points[] для одной линии.
             */
            interfaceCanvas.dispatchEvent(
                createMouseEvent(
                    "mousedown",
                    interfaceCanvas,
                    points[0],
                    1
                )
            );

            /*
             * Много mousemove, но mouseup будет только один —
             * после окончания всей маски.
             */
            for (
                let i = 1;
                i < points.length;
                i += 1
            ) {
                const currentPoint = points[i];

                interfaceCanvas.dispatchEvent(
                    createMouseEvent(
                        "mousemove",
                        interfaceCanvas,
                        currentPoint,
                        1
                    )
                );

                /*
                 * Небольшая пауза нужна не для логики Gradio,
                 * а чтобы пользователь видел виртуальное рисование.
                 *
                 * Поставьте 0 для максимально быстрой генерации.
                 */
                if (i % 4 === 0) {
                    await delay(2);
                }
            }

            const lastPoint =
                points[points.length - 1];

            /*
             * Один mouseup.
             *
             * Здесь Gradio сохранит всю виртуальную
             * маску как один элемент lines[].
             */
            interfaceCanvas.dispatchEvent(
                createMouseEvent(
                    "mouseup",
                    interfaceCanvas,
                    lastPoint,
                    0
                )
            );

            /*
             * Даём Gradio выполнить saveLine()
             * и dispatch("change").
             */
            await delay(100);

            console.log(
                "[VirtualMask] One synthetic stroke completed."
            );

            return true;
        } finally {
            virtualDrawing = false;
        }
    }

    async function applyGeneratedMask(value) {
        const maskSource = getSource(value);

        if (!maskSource) {
            console.warn(
                "[VirtualMask] value.mask is empty.",
                value
            );

            return value;
        }

        const elements = getElements();

        if (!elements) {
            console.error(
                "[VirtualMask] Inpaint mask component not found."
            );

            return value;
        }

        initializedRoot = elements.root;

        /*
         * Важно:
         *
         * 1. не рисуем mask напрямую через drawImage()
         *    в canvas[key="mask"];
         *
         * 2. сначала очищаем старые линии через
         *    штатную кнопку Clear;
         *
         * 3. затем создаём одну synthetic line.
         */
        await clearPreviousGradioLines(elements);

        await drawMaskAsOneGradioLine(
            maskSource
        );

        return value;
    }

    /*
     * Функция вызывается из webui.py:
     *
     * .then(
     *     fn=lambda: None,
     *     inputs=inpaint_mask_image,
     *     outputs=None,
     *     _js="apply_generated_mask"
     * )
     */
    window.apply_generated_mask = function (value) {
        /*
         * Не возвращаем Promise наружу как значение компонента.
         * Gradio получает исходный value, а сама отрисовка
         * выполняется асинхронно в браузере.
         */
        applyGeneratedMask(value).catch((error) => {
            console.error(
                "[VirtualMask] Failed to draw generated mask:",
                error
            );
        });

        return value;
    };

    window.getVirtualMaskDebugState = function () {
        const elements = getElements();

        return {
            initialized: Boolean(initializedRoot),
            virtualDrawing,
            hasMaskCanvas: Boolean(
                elements && elements.maskCanvas
            ),
            hasInterfaceCanvas: Boolean(
                elements && elements.interfaceCanvas
            ),
            hasClearButton: Boolean(
                elements && elements.clearButton
            )
        };
    };

    if (typeof onUiLoaded === "function") {
        onUiLoaded(function () {
            const elements = getElements();

            if (elements) {
                initializedRoot = elements.root;
            }
        });
    }
})();
