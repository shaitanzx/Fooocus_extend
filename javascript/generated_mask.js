(function () {
    "use strict";

    let generatedMaskSource = null;
    let manualLineCount = 0;
    let undoHandlerInstalled = false;
    let drawingHandlersInstalled = false;
    let restoringMask = false;

    function getRoot() {
        return gradioApp().querySelector("#inpaint_mask_canvas");
    }

    function getCanvases() {
        const root = getRoot();

        if (!root) {
            return null;
        }

        return {
            root,
            mask: root.querySelector('canvas[key="mask"]'),
            interface: root.querySelector('canvas[key="interface"]'),
            drawing: root.querySelector('canvas[key="drawing"]'),
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
            return mask.url || mask.path || mask.name || null;
        }

        return null;
    }

    function clearCanvas(canvas) {
        if (!canvas) {
            return;
        }

        const context = canvas.getContext("2d");

        if (!context) {
            return;
        }

        context.clearRect(
            0,
            0,
            canvas.width,
            canvas.height
        );
    }

    function loadImage(source) {
        return new Promise((resolve, reject) => {
            const image = new Image();

            image.onload = () => resolve(image);
            image.onerror = reject;
            image.src = source;
        });
    }

    async function restoreGeneratedMask() {
        if (!generatedMaskSource || restoringMask) {
            return;
        }

        const canvases = getCanvases();

        if (!canvases || !canvases.mask) {
            return;
        }

        restoringMask = true;

        try {
            const image = await loadImage(generatedMaskSource);
            const canvas = canvases.mask;
            const context = canvas.getContext("2d");

            if (!context) {
                return;
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

            console.log(
                "[GeneratedMask] Base mask restored after undo."
            );
        } catch (error) {
            console.error(
                "[GeneratedMask] Cannot restore base mask:",
                error
            );
        } finally {
            restoringMask = false;
        }
    }

    function registerManualDrawingTracking() {
        if (drawingHandlersInstalled) {
            return;
        }

        const canvases = getCanvases();

        if (!canvases || !canvases.interface) {
            return;
        }

        const interfaceCanvas = canvases.interface;

        /*
         * В Gradio верхним canvas, который принимает мышь,
         * является canvas[key="interface"], а не mask.
         */
        interfaceCanvas.addEventListener(
            "mousedown",
            function (event) {
                if (event.button !== 0) {
                    return;
                }

                if (restoringMask) {
                    return;
                }

                manualLineCount += 1;

                console.log(
                    "[GeneratedMask] Manual lines:",
                    manualLineCount
                );
            },
            true
        );

        /*
         * Если пользователь полностью удалил изображение
         * или компонент был очищен, счётчик нужно сбросить.
         */
        interfaceCanvas.addEventListener(
            "mouseleave",
            function () {
                // Только визуальное событие. Счётчик не меняем.
            },
            true
        );

        drawingHandlersInstalled = true;
    }

    function findUndoButton() {
        const root = getRoot();

        if (!root) {
            return null;
        }

        /*
         * IconButton Gradio обычно содержит aria-label="Undo".
         */
        return root.querySelector(
            'button[aria-label="Undo"], [role="button"][aria-label="Undo"]'
        );
    }

    function installUndoHandler() {
        if (undoHandlerInstalled) {
            return;
        }

        const undoButton = findUndoButton();

        if (!undoButton) {
            return;
        }

        undoButton.addEventListener(
            "click",
            function () {
                if (!generatedMaskSource) {
                    return;
                }

                /*
                 * Сначала Gradio штатно удалит последнюю линию
                 * и очистит mask canvas.
                 *
                 * Поэтому восстанавливаем базовую маску
                 * после завершения обработчика Gradio.
                 */
                if (manualLineCount > 0) {
                    manualLineCount -= 1;

                    setTimeout(() => {
                        restoreGeneratedMask();
                    }, 0);

                    console.log(
                        "[GeneratedMask] Undo manual line. Remaining:",
                        manualLineCount
                    );
                } else {
                    /*
                     * Все ручные линии уже отменены.
                     * Следующий Undo должен удалить и сгенерированную маску.
                     */
                    generatedMaskSource = null;

                    const canvases = getCanvases();

                    if (canvases) {
                        clearCanvas(canvases.mask);
                    }

                    console.log(
                        "[GeneratedMask] Base generated mask removed."
                    );
                }
            },
            true
        );

        undoHandlerInstalled = true;
    }

    function initializeComponent(attempt = 0) {
        registerManualDrawingTracking();
        installUndoHandler();

        if (
            (!drawingHandlersInstalled || !undoHandlerInstalled) &&
            attempt < 50
        ) {
            setTimeout(() => {
                initializeComponent(attempt + 1);
            }, 100);
        }
    }

    window.apply_generated_mask = function (value) {
        const maskSource = getMaskSource(value);

        if (!maskSource) {
            return value;
        }

        /*
         * Новая сгенерированная маска начинает новый цикл undo.
         */
        generatedMaskSource = maskSource;
        manualLineCount = 0;

        initializeComponent();

        setTimeout(async () => {
            await restoreGeneratedMask();
            initializeComponent();
        }, 100);

        console.log(
            "[GeneratedMask] New base mask registered."
        );

        return value;
    };

    /*
     * Позволяет вручную сбросить состояние из консоли:
     *
     * window.reset_generated_mask_state()
     */
    window.reset_generated_mask_state = function () {
        generatedMaskSource = null;
        manualLineCount = 0;

        const canvases = getCanvases();

        if (canvases) {
            clearCanvas(canvases.mask);
        }

        console.log(
            "[GeneratedMask] State reset."
        );
    };

    onUiLoaded(function () {
        initializeComponent();
    });

    onAfterUiUpdate(function () {
        initializeComponent();
    });
})();
