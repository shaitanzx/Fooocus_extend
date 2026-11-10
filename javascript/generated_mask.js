(function () {
    "use strict";

    function getImageSource(value) {
        if (!value) {
            return null;
        }

        if (typeof value === "string") {
            return value;
        }

        if (typeof value === "object") {
            return value.url || value.path || value.name || null;
        }

        return null;
    }

    function findMaskComponent() {
        const root = gradioApp().querySelector("#inpaint_mask_canvas");

        if (!root) {
            return null;
        }

        const maskCanvas = root.querySelector('canvas[key="mask"]');
        const interfaceCanvas = root.querySelector('canvas[key="interface"]');

        if (!maskCanvas) {
            return null;
        }

        return {
            root,
            maskCanvas,
            interfaceCanvas,
        };
    }

    function synchronizeGradioImage(component) {
        const interfaceCanvas = component.interfaceCanvas;

        if (!interfaceCanvas) {
            return;
        }

        /*
         * Image.svelte реагирует на событие canvas interface.
         * В результате он заново считывает canvas[key="mask"]
         * и обновляет внутреннее значение:
         *
         * {
         *     image: ...,
         *     mask: ...
         * }
         */
        const rect = interfaceCanvas.getBoundingClientRect();

        const eventData = {
            bubbles: true,
            cancelable: true,
            clientX: rect.left - 10000,
            clientY: rect.top - 10000,
            button: 0,
            buttons: 0,
            view: window,
        };

        interfaceCanvas.dispatchEvent(
            new MouseEvent("mousedown", eventData)
        );

        interfaceCanvas.dispatchEvent(
            new MouseEvent("mousemove", eventData)
        );

        interfaceCanvas.dispatchEvent(
            new MouseEvent("mouseup", eventData)
        );
    }

    function drawMaskToCanvas(maskSource, attempt = 0) {
        const component = findMaskComponent();

        if (!component) {
            if (attempt < 30) {
                setTimeout(() => {
                    drawMaskToCanvas(maskSource, attempt + 1);
                }, 100);
            } else {
                console.warn(
                    "[GeneratedMask] Mask canvas was not found."
                );
            }

            return;
        }

        const maskImage = new Image();

        maskImage.onload = function () {
            const canvas = component.maskCanvas;
            const context = canvas.getContext("2d");

            if (!context) {
                console.warn(
                    "[GeneratedMask] Cannot get mask canvas context."
                );
                return;
            }

            context.save();

            /*
             * Это именно отдельный mask-layer.
             * Подложка canvas[key="drawing"] не изменяется.
             */
            context.globalCompositeOperation = "source-over";

            context.clearRect(
                0,
                0,
                canvas.width,
                canvas.height
            );

            context.drawImage(
                maskImage,
                0,
                0,
                canvas.width,
                canvas.height
            );

            context.restore();

            /*
             * Передаём изменённый canvas обратно во внутреннее
             * состояние Gradio, чтобы последующее нажатие Generate
             * получило эту маску как обычную нарисованную маску.
             */
            synchronizeGradioImage(component);

            console.log(
                "[GeneratedMask] Generated mask loaded into canvas[key=\"mask\"]."
            );
        };

        maskImage.onerror = function () {
            console.error(
                "[GeneratedMask] Failed to load generated mask."
            );
        };

        maskImage.src = maskSource;
    }

    /*
     * Эта функция вызывается из webui.py через _js.
     *
     * value ожидается в формате:
     *
     * {
     *     image: "...base64 image...",
     *     mask: "...base64 mask..."
     * }
     */
    window.apply_generated_mask = function (value) {
        if (!value || typeof value !== "object") {
            console.warn(
                "[GeneratedMask] Invalid component value:",
                value
            );
            return value;
        }

        const maskSource = getImageSource(value.mask);

        if (!maskSource) {
            console.warn(
                "[GeneratedMask] value.mask is empty:",
                value
            );
            return value;
        }

        drawMaskToCanvas(maskSource);

        /*
         * Не меняем value.image и value.mask.
         * Возвращаем исходное значение компонента.
         */
        return value;
    };
})();
