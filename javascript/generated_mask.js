(function () {
    "use strict";

    /*
     * Состояние текущей сгенерированной маски.
     *
     * history[0] — только сгенерированная маска;
     * history[1] — маска + первая ручная линия;
     * history[2] — маска + первая и вторая ручные линии.
     */
    let generatedMaskSource = null;
    let maskHistory = [];

    let initializedRoot = null;
    let initializedInterfaceCanvas = null;
    let initializedUndoButton = null;
    let initializedClearButton = null;

    let strokeInProgress = false;
    let lastCapturedState = null;

    function getRoot() {
        return gradioApp().querySelector("#inpaint_mask_canvas");
    }

    function getComponentElements() {
        const root = getRoot();

        if (!root) {
            return null;
        }

        return {
            root: root,

            maskCanvas: root.querySelector(
                'canvas[key="mask"]'
            ),

            drawingCanvas: root.querySelector(
                'canvas[key="drawing"]'
            ),

            interfaceCanvas: root.querySelector(
                'canvas[key="interface"]'
            ),

            undoButton: root.querySelector(
                'button[aria-label="Undo"], ' +
                '[role="button"][aria-label="Undo"]'
            ),

            clearButton: root.querySelector(
                'button[aria-label="Clear"], ' +
                '[role="button"][aria-label="Clear"]'
            ),

            removeImageButton: root.querySelector(
                'button[aria-label="Remove Image"], ' +
                '[role="button"][aria-label="Remove Image"]'
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

    function clearMaskCanvas() {
        const elements = getComponentElements();

        if (!elements || !elements.maskCanvas) {
            return;
        }

        const context = elements.maskCanvas.getContext("2d");

        if (!context) {
            return;
        }

        context.clearRect(
            0,
            0,
            elements.maskCanvas.width,
            elements.maskCanvas.height
        );
    }

    function getMaskCanvasData() {
        const elements = getComponentElements();

        if (!elements || !elements.maskCanvas) {
            return null;
        }

        return elements.maskCanvas.toDataURL("image/png");
    }

    async function drawSourceToMaskCanvas(source) {
        const elements = getComponentElements();

        if (!elements || !elements.maskCanvas) {
            return false;
        }

        const image = await loadImage(source);
        const canvas = elements.maskCanvas;
        const context = canvas.getContext("2d");

        if (!context) {
            return false;
        }

        context.save();

        context.globalCompositeOperation = "source-over";

        context.clearRect(
            0,
            0,
            canvas.width,
            canvas.height
        );

        context.drawImage(
            image,
            0,
            0,
            canvas.width,
            canvas.height
        );

        context.restore();

        return true;
    }

    async function restoreHistoryState(dataUrl) {
        if (!dataUrl) {
            clearMaskCanvas();
            return;
        }

        try {
            await drawSourceToMaskCanvas(dataUrl);

            console.log(
                "[GeneratedMask] Restored history state."
            );
        } catch (error) {
            console.error(
                "[GeneratedMask] Failed to restore history state:",
                error
            );
        }
    }

    async function captureCurrentMaskState() {
        if (!generatedMaskSource) {
            return;
        }

        const currentState = getMaskCanvasData();

        if (!currentState) {
            return;
        }

        /*
         * Не добавляем одно и то же состояние несколько раз.
         */
        if (currentState === lastCapturedState) {
            return;
        }

        lastCapturedState = currentState;
        maskHistory.push(currentState);

        console.log(
            "[GeneratedMask] Saved state. History length:",
            maskHistory.length
        );
    }

    function scheduleCaptureAfterDrawing() {
        /*
         * Gradio рисует линию после pointer/mouse event.
         * Поэтому считываем canvas с небольшой задержкой.
         */
        setTimeout(() => {
            requestAnimationFrame(() => {
                captureCurrentMaskState();
            });
        }, 80);
    }

    function installDrawingTracking(elements) {
        if (
            !elements ||
            !elements.interfaceCanvas ||
            elements.interfaceCanvas === initializedInterfaceCanvas
        ) {
            return;
        }

        initializedInterfaceCanvas = elements.interfaceCanvas;

        const interfaceCanvas = elements.interfaceCanvas;

        /*
         * Основной вариант для Gradio.
         */
        interfaceCanvas.addEventListener(
            "pointerdown",
            function (event) {
                if (event.button !== undefined && event.button !== 0) {
                    return;
                }

                strokeInProgress = true;
            },
            true
        );

        interfaceCanvas.addEventListener(
            "pointerup",
            function () {
                if (!strokeInProgress) {
                    return;
                }

                strokeInProgress = false;
                scheduleCaptureAfterDrawing();
            },
            true
        );

        interfaceCanvas.addEventListener(
            "pointercancel",
            function () {
                strokeInProgress = false;
            },
            true
        );

        /*
         * Запасной вариант для браузеров/сборок,
         * в которых canvas использует mouse events.
         */
        interfaceCanvas.addEventListener(
            "mouseup",
            function () {
                if (!strokeInProgress) {
                    return;
                }

                strokeInProgress = false;
                scheduleCaptureAfterDrawing();
            },
            true
        );

        interfaceCanvas.addEventListener(
            "touchend",
            function () {
                if (!strokeInProgress) {
                    return;
                }

                strokeInProgress = false;
                scheduleCaptureAfterDrawing();
            },
            true
        );

        console.log(
            "[GeneratedMask] Drawing tracking installed."
        );
    }

    function installUndoHandler(elements) {
        if (
            !elements ||
            !elements.undoButton ||
            elements.undoButton === initializedUndoButton
        ) {
            return;
        }

        initializedUndoButton = elements.undoButton;

        elements.undoButton.addEventListener(
            "click",
            function () {
                if (!generatedMaskSource) {
                    return;
                }

                /*
                 * Сначала штатный Gradio undo удаляет последнюю
                 * линию и очищает mask canvas.
                 *
                 * Затем мы восстанавливаем предыдущий полный
                 * снимок canvas.
                 */
                setTimeout(() => {
                    requestAnimationFrame(() => {
                        requestAnimationFrame(async () => {
                            if (maskHistory.length > 1) {
                                /*
                                 * Удаляем последнее состояние:
                                 *
                                 * [mask, mask+line1, mask+line2]
                                 * → [mask, mask+line1]
                                 */
                                maskHistory.pop();

                                const previousState =
                                    maskHistory[maskHistory.length - 1];

                                lastCapturedState = previousState;

                                await restoreHistoryState(
                                    previousState
                                );

                                console.log(
                                    "[GeneratedMask] Undo manual line. " +
                                    "Remaining states:",
                                    maskHistory.length
                                );
                            } else if (maskHistory.length === 1) {
                                /*
                                 * После отмены всех ручных линий
                                 * следующее Undo удаляет базовую
                                 * сгенерированную маску.
                                 */
                                maskHistory = [];
                                generatedMaskSource = null;
                                lastCapturedState = null;

                                clearMaskCanvas();

                                console.log(
                                    "[GeneratedMask] " +
                                    "Generated mask removed."
                                );
                            }
                        });
                    });
                }, 180);
            },
            true
        );

        console.log(
            "[GeneratedMask] Undo handler installed."
        );
    }

    function installClearHandler(elements) {
        if (
            !elements ||
            !elements.clearButton ||
            elements.clearButton === initializedClearButton
        ) {
            return;
        }

        initializedClearButton = elements.clearButton;

        elements.clearButton.addEventListener(
            "click",
            function () {
                /*
                 * Кнопка Clear должна удалить маску полностью,
                 * включая сохранённую историю.
                 */
                setTimeout(() => {
                    generatedMaskSource = null;
                    maskHistory = [];
                    lastCapturedState = null;

                    console.log(
                        "[GeneratedMask] Clear pressed. " +
                        "History reset."
                    );
                }, 100);
            },
            true
        );

        console.log(
            "[GeneratedMask] Clear handler installed."
        );
    }

    function installRemoveImageHandler(elements) {
        if (
            !elements ||
            !elements.removeImageButton
        ) {
            return;
        }

        /*
         * Этот обработчик ставится только один раз на конкретную
         * кнопку. Для надёжности используем свойство элемента.
         */
        if (elements.removeImageButton.__generatedMaskHandler) {
            return;
        }

        elements.removeImageButton.__generatedMaskHandler = true;

        elements.removeImageButton.addEventListener(
            "click",
            function () {
                setTimeout(() => {
                    generatedMaskSource = null;
                    maskHistory = [];
                    lastCapturedState = null;

                    console.log(
                        "[GeneratedMask] Image removed. " +
                        "History reset."
                    );
                }, 100);
            },
            true
        );
    }

    function initializeComponent() {
        const elements = getComponentElements();

        if (!elements) {
            return;
        }

        installDrawingTracking(elements);
        installUndoHandler(elements);
        installClearHandler(elements);
        installRemoveImageHandler(elements);

        initializedRoot = elements.root;
    }

    function initializeRepeatedly(attempt = 0) {
        initializeComponent();

        if (attempt >= 50) {
            return;
        }

        setTimeout(() => {
            initializeRepeatedly(attempt + 1);
        }, 100);
    }

    async function applyGeneratedMask(value) {
        const maskSource = getMaskSource(value);

        if (!maskSource) {
            console.warn(
                "[GeneratedMask] value.mask is empty.",
                value
            );

            return value;
        }

        generatedMaskSource = maskSource;
        maskHistory = [];
        lastCapturedState = null;
        strokeInProgress = false;

        initializeRepeatedly();

        /*
         * Ждём, пока Gradio завершит обновление компонента.
         */
        setTimeout(async () => {
            try {
                const loaded =
                    await drawSourceToMaskCanvas(
                        generatedMaskSource
                    );

                if (!loaded) {
                    console.warn(
                        "[GeneratedMask] Mask canvas is not ready."
                    );

                    return;
                }

                /*
                 * Сохраняем первый снимок:
                 * только сгенерированная маска.
                 */
                const baseState = getMaskCanvasData();

                if (baseState) {
                    maskHistory = [baseState];
                    lastCapturedState = baseState;
                }

                console.log(
                    "[GeneratedMask] Base mask registered. " +
                    "History length:",
                    maskHistory.length
                );
            } catch (error) {
                console.error(
                    "[GeneratedMask] Cannot apply generated mask:",
                    error
                );
            }
        }, 250);

        return value;
    }

    /*
     * Вызывается из webui.py через _js:
     *
     * _js="apply_generated_mask"
     */
    window.apply_generated_mask = function (value) {
        applyGeneratedMask(value);
        return value;
    };

    /*
     * Диагностика из DevTools.
     */
    window.getGeneratedMaskDebugState = function () {
        return {
            hasGeneratedMask: Boolean(generatedMaskSource),
            historyLength: maskHistory.length,
            initialized: Boolean(initializedRoot),
            drawingTracking:
                Boolean(initializedInterfaceCanvas),
            undoTracking:
                Boolean(initializedUndoButton),
            clearTracking:
                Boolean(initializedClearButton)
        };
    };

    /*
     * Полный ручной сброс состояния.
     */
    window.resetGeneratedMaskState = function () {
        generatedMaskSource = null;
        maskHistory = [];
        lastCapturedState = null;
        strokeInProgress = false;

        clearMaskCanvas();

        console.log(
            "[GeneratedMask] State reset."
        );
    };

    if (typeof onUiLoaded === "function") {
        onUiLoaded(function () {
            initializeRepeatedly();
        });
    }

    if (typeof onAfterUiUpdate === "function") {
        onAfterUiUpdate(function () {
            initializeComponent();
        });
    }
})();
