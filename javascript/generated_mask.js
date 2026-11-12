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

    function getMaskSource(value) {
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
         * Запасной радиус кисти.
         */
        return Math.max(
            8,
            Math.min(
                canvas.width,
                canvas.height
            ) * 0.025
        );
    }

    /*
     * Загружает изображение маски во временный canvas
     * и преобразует его в бинарный массив active.
     *
     * Основной canvas[key="mask"] здесь не используется.
     */
    function normalizeMaskToCanvas(
        image,
        width,
        height
    ) {
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

        const imageData =
            sourceContext.getImageData(
                0,
                0,
                width,
                height
            );

        const pixels = imageData.data;

        /*
         * active[index] = 1 означает,
         * что соответствующий пиксель принадлежит маске.
         */
        const active = new Uint8Array(
            width * height
        );

        /*
         * Если у изображения есть ненулевой alpha,
         * используем alpha-канал.
         * Иначе используем RGB.
         */
        let hasAlpha = false;

        for (
            let i = 3;
            i < pixels.length;
            i += 4
        ) {
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

                const red =
                    pixels[pixelIndex];

                const green =
                    pixels[pixelIndex + 1];

                const blue =
                    pixels[pixelIndex + 2];

                const alpha =
                    pixels[pixelIndex + 3];

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

    /*
     * Разбиение бинарной маски на связные компоненты.
     *
     * Используется 8-связность:
     * пиксели, соприкасающиеся по диагонали,
     * считаются частью одной области.
     */
    function splitMaskIntoComponents(
        active,
        width,
        height,
        minimumArea
    ) {
        const visited = new Uint8Array(
            width * height
        );

        const components = [];

        const directions = [
            [-1, -1],
            [0, -1],
            [1, -1],

            [-1, 0],
            [1, 0],

            [-1, 1],
            [0, 1],
            [1, 1]
        ];

        function indexOf(x, y) {
            return y * width + x;
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
                const startIndex =
                    indexOf(x, y);

                if (
                    active[startIndex] === 0 ||
                    visited[startIndex] !== 0
                ) {
                    continue;
                }

                const queue = [
                    { x, y }
                ];

                const pixels = [];

                visited[startIndex] = 1;

                let queueIndex = 0;

                while (
                    queueIndex < queue.length
                ) {
                    const current =
                        queue[queueIndex];

                    queueIndex += 1;
                    pixels.push(current);

                    for (
                        const direction
                        of directions
                    ) {
                        const dx = direction[0];
                        const dy = direction[1];

                        const nx =
                            current.x + dx;

                        const ny =
                            current.y + dy;

                        if (
                            nx < 0 ||
                            nx >= width ||
                            ny < 0 ||
                            ny >= height
                        ) {
                            continue;
                        }

                        const neighborIndex =
                            indexOf(nx, ny);

                        if (
                            active[neighborIndex] === 0 ||
                            visited[neighborIndex] !== 0
                        ) {
                            continue;
                        }

                        visited[neighborIndex] = 1;

                        queue.push({
                            x: nx,
                            y: ny
                        });
                    }
                }

                if (
                    pixels.length >= minimumArea
                ) {
                    components.push(pixels);
                }
            }
        }

        return components;
    }

    function componentToActiveMask(
        component,
        width,
        height
    ) {
        const active = new Uint8Array(
            width * height
        );

        for (
            const pixel of component
        ) {
            active[
                pixel.y * width + pixel.x
            ] = 1;
        }

        return active;
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
            const isActive =
                active[y * width + x] !== 0;

            if (
                isActive &&
                runStart < 0
            ) {
                runStart = x;
            }

            const isLastPixel =
                x === width - 1;

            if (
                runStart >= 0 &&
                (!isActive || isLastPixel)
            ) {
                const runEnd =
                    isActive && isLastPixel
                        ? x
                        : x - 1;

                const runLength =
                    runEnd - runStart + 1;

                if (
                    runLength >=
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

    /*
     * Строит траекторию одной связной области.
     *
     * Все точки этой функции будут отправлены
     * между одной парой mousedown/mouseup.
     */
    function buildComponentStrokePoints(
        component,
        width,
        height,
        brushRadius
    ) {
        const componentMask =
            componentToActiveMask(
                component,
                width,
                height
            );

        /*
         * Расстояние между горизонтальными
         * проходами виртуальной кисти.
         */
        const rowStep = Math.max(
            2,
            Math.floor(
                brushRadius * 0.65
            )
        );

        const minimumRunLength = Math.max(
            2,
            Math.floor(
                brushRadius * 0.5
            )
        );

        const points = [];

        let rowNumber = 0;

        for (
            let y = 0;
            y < height;
            y += rowStep
        ) {
            const runs =
                getRunsForRow(
                    componentMask,
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
                const run of orderedRuns
            ) {
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

            clientX:
                clientPoint.clientX,

            clientY:
                clientPoint.clientY,

            screenX:
                clientPoint.clientX,

            screenY:
                clientPoint.clientY,

            button: 0,
            buttons
        });
    }

    async function clearPreviousGradioLines(
        elements
    ) {
        /*
         * Используем штатную кнопку Clear.
         * Она очищает lines[] и mask canvas.
         */
        if (elements.clearButton) {
            elements.clearButton.click();

            await delay(120);
        }
    }

    /*
     * Рисует одну связную область как одну
     * запись Gradio lines[].
     */
    async function drawSyntheticStroke(
        interfaceCanvas,
        points
    ) {
        if (
            !points ||
            points.length < 2
        ) {
            return false;
        }

        const firstPoint = points[0];

        /*
         * ВАЖНО:
         * ровно один mousedown на одну область.
         */
        interfaceCanvas.dispatchEvent(
            createMouseEvent(
                "mousedown",
                interfaceCanvas,
                firstPoint,
                1
            )
        );

        for (
            let i = 1;
            i < points.length;
            i += 1
        ) {
            const point = points[i];

            interfaceCanvas.dispatchEvent(
                createMouseEvent(
                    "mousemove",
                    interfaceCanvas,
                    point,
                    1
                )
            );

            /*
             * Небольшая задержка нужна,
             * чтобы маска визуально рисовалась.
             * Для максимальной скорости можно поставить 0.
             */
            if (i % 4 === 0) {
                await delay(2);
            }
        }

        const lastPoint =
            points[points.length - 1];

        /*
         * ВАЖНО:
         * ровно один mouseup на одну область.
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
        await delay(60);

        return true;
    }

    /*
     * Рисует каждую связную область отдельным
     * synthetic stroke.
     */
    async function drawMaskAsSeparateGradioLines(
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

            const width =
                maskCanvas.width;

            const height =
                maskCanvas.height;

            const normalizedMask =
                normalizeMaskToCanvas(
                    maskImage,
                    width,
                    height
                );

            if (!normalizedMask) {
                return false;
            }

            const brushRadius =
                getBrushRadius(maskCanvas);

            /*
             * Минимальный размер области.
             * Увеличьте значение, если нужно
             * отбрасывать мелкий шум.
             */
            const minimumArea = Math.max(
                20,
                Math.floor(
                    brushRadius * brushRadius * 0.25
                )
            );

            const components =
                splitMaskIntoComponents(
                    normalizedMask.active,
                    width,
                    height,
                    minimumArea
                );

            if (
                components.length === 0
            ) {
                console.warn(
                    "[VirtualMask] No mask components found."
                );

                return false;
            }

            /*
             * Крупные области рисуются первыми.
             * Последняя нарисованная область будет
             * первой удаляться через Undo.
             */
            components.sort(
                function (a, b) {
                    return b.length - a.length;
                }
            );

            console.log(
                "[VirtualMask] Connected components:",
                components.length
            );

            for (
                let i = 0;
                i < components.length;
                i += 1
            ) {
                const component =
                    components[i];

                const points =
                    buildComponentStrokePoints(
                        component,
                        width,
                        height,
                        brushRadius
                    );

                if (
                    points.length < 2
                ) {
                    continue;
                }

                console.log(
                    "[VirtualMask] Drawing component",
                    i + 1,
                    "of",
                    components.length,
                    "points:",
                    points.length
                );

                /*
                 * Каждая область получает
                 * отдельную пару mouse events.
                 *
                 * Между областями есть mouseup,
                 * поэтому они не соединяются линией.
                 */
                await drawSyntheticStroke(
                    interfaceCanvas,
                    points
                );

                await delay(80);
            }

            await delay(120);

            console.log(
                "[VirtualMask] All mask components drawn."
            );

            return true;
        } finally {
            virtualDrawing = false;
        }
    }

    async function applyGeneratedMask(value) {
        const maskSource =
            getMaskSource(value);

        if (!maskSource) {
            console.warn(
                "[VirtualMask] value.mask is empty.",
                value
            );

            return value;
        }

        const elements =
            getElements();

        if (!elements) {
            console.error(
                "[VirtualMask] Inpaint component not found."
            );

            return value;
        }

        initializedRoot =
            elements.root;

        /*
         * Удаляем старые ручные линии
         * и старую маску штатным механизмом.
         */
        await clearPreviousGradioLines(
            elements
        );

        /*
         * Маска теперь попадает в lines[]
         * через synthetic mouse events.
         */
        await drawMaskAsSeparateGradioLines(
            maskSource
        );

        return value;
    }

    /*
     * Вызывается из webui.py через _js.
     */
    window.apply_generated_mask = function (value) {
        applyGeneratedMask(value).catch(
            function (error) {
                console.error(
                    "[VirtualMask] Failed to draw mask:",
                    error
                );
            }
        );

        return value;
    };

    /*
     * Диагностика в DevTools.
     */
    window.getVirtualMaskDebugState =
        function () {
            const elements =
                getElements();

            return {
                initialized:
                    Boolean(initializedRoot),

                virtualDrawing,

                hasMaskCanvas:
                    Boolean(
                        elements &&
                        elements.maskCanvas
                    ),

                hasInterfaceCanvas:
                    Boolean(
                        elements &&
                        elements.interfaceCanvas
                    ),

                hasClearButton:
                    Boolean(
                        elements &&
                        elements.clearButton
                    )
            };
        };

    if (
        typeof onUiLoaded ===
        "function"
    ) {
        onUiLoaded(function () {
            const elements =
                getElements();

            if (elements) {
                initializedRoot =
                    elements.root;
            }
        });
    }
})();
