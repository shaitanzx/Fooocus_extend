onUiLoaded(async() => {

    const defaultHotkeysConfig = {
        canvas_hotkey_zoom: "Shift",
        canvas_hotkey_adjust: "Ctrl",
        canvas_zoom_undo_extra_key: "Ctrl",
        canvas_zoom_hotkey_undo: "KeyZ",
        canvas_hotkey_reset: "KeyR",
        canvas_hotkey_fullscreen: "KeyS",
        canvas_hotkey_move: "KeyF",
        canvas_hotkey_eraser: "KeyE",
        canvas_show_tooltip: true,
        canvas_auto_expand: true,
        canvas_blur_prompt: true,
    };

    const hotkeysConfig = { ...defaultHotkeysConfig };
    const elemData = {};
    let activeElement = null;
    let isMoving = false;

    function isModifierKey(event, key) {
        if (key === "Ctrl") return event.ctrlKey;
        if (key === "Shift") return event.shiftKey;
        if (key === "Alt") return event.altKey;
        return false;
    }

    function getCanvas(targetElement, key) {
        return targetElement.querySelector(`canvas[key="${key}"]`);
    }

    function hasHorizontalScrollbar(element) {
        return element.scrollWidth > element.clientWidth;
    }

    function getCanvasPoint(canvas, event) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        return {
            x: Math.max(0, Math.min(canvas.width, (event.clientX - rect.left) * scaleX)),
            y: Math.max(0, Math.min(canvas.height, (event.clientY - rect.top) * scaleY)),
        };
    }

    function applyZoomAndPan(elemId) {
        const targetElement = gradioApp().querySelector(elemId);
        if (!targetElement) return;

        // Важно: далее используется именно zoomLevel, а не zoom.
        elemData[elemId] = {
            zoomLevel: 1,
            panX: 0,
            panY: 0,
        };

        let fullScreenMode = false;
        let isKeyDownHandlerAttached = false;
        let isEraserMode = false;
        let isDrawingEraser = false;
        let lastEraserPoint = null;
        let brushDiameter = 40;
        let overlay = null;

        targetElement.style.transformOrigin = "0 0";

        function getBrushRadius() {
            return Math.max(1, brushDiameter / 2);
        }

        function updateBrushDiameter() {
            const input = targetElement.querySelector("input[aria-label='Brush radius']");
            if (!input) return;

            const value = Number.parseFloat(input.value);
            if (Number.isFinite(value) && value > 0) {
                brushDiameter = value;
            }
        }

        updateBrushDiameter();

        const brushInput = targetElement.querySelector("input[aria-label='Brush radius']");
        if (brushInput) {
            brushInput.addEventListener("input", updateBrushDiameter);
            brushInput.addEventListener("change", updateBrushDiameter);
        }

        function getInterfaceCanvas() {
            return getCanvas(targetElement, "interface");
        }

        function createOverlay() {
            if (overlay && overlay.isConnected) return overlay;

            const interfaceCanvas = getInterfaceCanvas();
            if (!interfaceCanvas || !interfaceCanvas.parentElement) return null;

            overlay = document.createElement("canvas");
            overlay.className = "fooocus-eraser-overlay";
            overlay.style.position = "absolute";
            overlay.style.pointerEvents = "none";
            overlay.style.zIndex = "1000";

            interfaceCanvas.parentElement.style.position = "relative";
            interfaceCanvas.parentElement.appendChild(overlay);
            updateOverlayGeometry();
            return overlay;
        }

        function updateOverlayGeometry() {
            const interfaceCanvas = getInterfaceCanvas();
            if (!interfaceCanvas || !overlay) return;

            const parentRect = interfaceCanvas.parentElement.getBoundingClientRect();
            const canvasRect = interfaceCanvas.getBoundingClientRect();

            overlay.width = interfaceCanvas.width;
            overlay.height = interfaceCanvas.height;
            overlay.style.left = `${canvasRect.left - parentRect.left}px`;
            overlay.style.top = `${canvasRect.top - parentRect.top}px`;
            overlay.style.width = `${canvasRect.width}px`;
            overlay.style.height = `${canvasRect.height}px`;
        }

        function clearOverlay() {
            if (!overlay) return;
            overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
        }

        function drawEraserCursor(point) {
            const currentOverlay = createOverlay();
            if (!currentOverlay) return;

            const interfaceCanvas = getInterfaceCanvas();
            if (!interfaceCanvas) return;

            updateOverlayGeometry();
            const ctx = currentOverlay.getContext("2d");
            ctx.clearRect(0, 0, currentOverlay.width, currentOverlay.height);

            ctx.save();
            ctx.fillStyle = "rgba(255, 255, 255, 0.25)";
            ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(point.x, point.y, getBrushRadius(), 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            ctx.restore();
        }

        function eraseCircle(maskCanvas, point, radius) {
            const ctx = maskCanvas.getContext("2d");
            ctx.save();
            ctx.globalCompositeOperation = "destination-out";
            ctx.beginPath();
            ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        }

        function eraseLine(maskCanvas, from, to, radius) {
            const ctx = maskCanvas.getContext("2d");
            ctx.save();
            ctx.globalCompositeOperation = "destination-out";
            ctx.lineWidth = radius * 2;
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            ctx.beginPath();
            ctx.moveTo(from.x, from.y);
            ctx.lineTo(to.x, to.y);
            ctx.stroke();
            ctx.restore();
        }

        function notifyGradioSketchChanged() {
            const interfaceCanvas = getInterfaceCanvas();
            const maskCanvas = getCanvas(targetElement, "mask");
            if (!interfaceCanvas || !maskCanvas) return;

            // Gradio собирает значение sketch-компонента в pointerup.
            // Событие отправляется после завершения последнего RAF-рисования.
            const rect = interfaceCanvas.getBoundingClientRect();
            const eventInit = {
                bubbles: true,
                cancelable: true,
                composed: true,
                pointerId: 1,
                isPrimary: true,
                button: 0,
                clientX: rect.left + rect.width / 2,
                clientY: rect.top + rect.height / 2,
            };

            interfaceCanvas.dispatchEvent(new PointerEvent("pointerup", eventInit));
            maskCanvas.dispatchEvent(new Event("input", { bubbles: true, composed: true }));
            maskCanvas.dispatchEvent(new Event("change", { bubbles: true, composed: true }));
        }

        /*
         * Эти обработчики намеренно НЕ вызывают preventDefault() и
         * stopImmediatePropagation(). Сначала штатный Gradio handler получает
         * pointer-событие, затем наш код стирает маску.
         */
        function handleEraserDown(event) {
            if (!isEraserMode || event.button !== 0) return;

            const maskCanvas = getCanvas(targetElement, "mask");
            if (!maskCanvas) return;

            isDrawingEraser = true;
            lastEraserPoint = getCanvasPoint(maskCanvas, event);

            requestAnimationFrame(() => {
                if (lastEraserPoint) {
                    eraseCircle(maskCanvas, lastEraserPoint, getBrushRadius());
                }
            });
        }

        function handleEraserMove(event) {
            if (!isEraserMode) return;

            const maskCanvas = getCanvas(targetElement, "mask");
            if (!maskCanvas) return;

            const point = getCanvasPoint(maskCanvas, event);
            drawEraserCursor(point);

            if (!isDrawingEraser || !lastEraserPoint) return;

            const from = lastEraserPoint;
            const to = point;
            lastEraserPoint = point;

            requestAnimationFrame(() => {
                eraseLine(maskCanvas, from, to, getBrushRadius());
            });
        }

        function handleEraserUp() {
            isDrawingEraser = false;
            lastEraserPoint = null;

            // Дожидаемся последнего eraseLine(), затем просим Gradio заново
            // сериализовать пару image/mask.
            requestAnimationFrame(() => {
                requestAnimationFrame(notifyGradioSketchChanged);
            });
        }

        function setEraserMode(enabled) {
            isEraserMode = enabled;
            targetElement.style.outline = enabled ? "3px solid #ff4444" : "none";
            targetElement.style.outlineOffset = "-3px";

            const interfaceCanvas = getInterfaceCanvas();
            if (interfaceCanvas) {
                interfaceCanvas.style.cursor = enabled ? "none" : "";
            }
            targetElement.style.cursor = enabled ? "none" : "";

            if (!enabled) {
                clearOverlay();
                if (overlay) overlay.remove();
                overlay = null;
            } else {
                createOverlay();
            }
        }

        function resetZoom() {
            elemData[elemId] = {
                zoomLevel: 1,
                panX: 0,
                panY: 0,
            };

            targetElement.style.transform = "translate(0px, 0px) scale(1)";
            targetElement.style.overflow = "hidden";
            targetElement.style.width = "";
            targetElement.isZoomed = false;
            fullScreenMode = false;
        }

        function toggleOverlap(forced = "") {
            if (forced === "off") {
                targetElement.style.zIndex = "0";
            } else if (forced === "on") {
                targetElement.style.zIndex = "998";
            } else {
                targetElement.style.zIndex = targetElement.style.zIndex === "998" ? "0" : "998";
            }
        }

        function updateZoom(newZoomLevel, mouseX, mouseY) {
            const data = elemData[elemId];
            newZoomLevel = Math.max(0.1, Math.min(newZoomLevel, 15));

            data.panX += mouseX - (mouseX * newZoomLevel) / data.zoomLevel;
            data.panY += mouseY - (mouseY * newZoomLevel) / data.zoomLevel;
            data.zoomLevel = newZoomLevel;

            targetElement.style.transform =
                `translate(${data.panX}px, ${data.panY}px) scale(${data.zoomLevel})`;
            targetElement.style.overflow = "visible";
            targetElement.isZoomed = true;
            toggleOverlap("on");
            updateOverlayGeometry();
        }

        function changeZoomLevel(operation, event) {
            if (!isModifierKey(event, hotkeysConfig.canvas_hotkey_zoom)) return;

            event.preventDefault();
            const data = elemData[elemId];
            let delta = 0.2;
            if (data.zoomLevel > 7) delta = 0.9;
            else if (data.zoomLevel > 2) delta = 0.6;

            const rect = targetElement.getBoundingClientRect();
            updateZoom(
                data.zoomLevel + (operation === "+" ? delta : -delta),
                event.clientX - rect.left,
                event.clientY - rect.top
            );
        }

        function adjustBrushSize(event) {
            const input = targetElement.querySelector("input[aria-label='Brush radius']");
            if (!input) return;

            const maxValue = Number.parseFloat(input.max) || 100;
            const currentValue = Number.parseFloat(input.value) || brushDiameter;
            const step = maxValue * 0.05;
            const nextValue = Math.min(
                maxValue,
                Math.max(1, currentValue + (event.deltaY > 0 ? -step : step))
            );

            input.value = nextValue;
            input.dispatchEvent(new Event("input", { bubbles: true }));
            input.dispatchEvent(new Event("change", { bubbles: true }));
            brushDiameter = nextValue;
        }

        function undoLastAction(event) {
            const hasExtraKey = isModifierKey(
                event,
                hotkeysConfig.canvas_zoom_undo_extra_key
            );
            if (event.button < 3 && !hasExtraKey) return;

            const undoButton = targetElement.querySelector("button[aria-label='Undo']");
            if (undoButton) {
                event.preventDefault();
                undoButton.click();
            }
        }

        function fitToScreen() {
            const canvas = getInterfaceCanvas();
            if (!canvas) return;

            if (fullScreenMode) {
                resetZoom();
                return;
            }

            targetElement.style.width = `${canvas.offsetWidth + 2}px`;
            targetElement.style.overflow = "visible";
            targetElement.style.transform = "translate(0px, 0px) scale(1)";

            const rect = targetElement.getBoundingClientRect();
            const scale = Math.min(
                window.innerWidth / targetElement.offsetWidth,
                window.innerHeight / targetElement.offsetHeight
            );

            const offsetX =
                (window.innerWidth - targetElement.offsetWidth * scale) / 2 - rect.left;
            const offsetY =
                (window.innerHeight - targetElement.offsetHeight * scale) / 2 - rect.top;

            targetElement.style.transform =
                `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;
            elemData[elemId].zoomLevel = scale;
            elemData[elemId].panX = offsetX;
            elemData[elemId].panY = offsetY;
            fullScreenMode = true;
            toggleOverlap("on");
            updateOverlayGeometry();
        }

        function handleKeyDown(event) {
            if (
                (event.ctrlKey && ["KeyC", "KeyV"].includes(event.code)) ||
                event.code === "F5"
            ) return;

            if (
                !hotkeysConfig.canvas_blur_prompt &&
                ["TEXTAREA", "INPUT"].includes(event.target.nodeName)
            ) return;

            if (event.code === hotkeysConfig.canvas_hotkey_eraser && activeElement === elemId) {
                event.preventDefault();
                setEraserMode(!isEraserMode);
                return;
            }

            if (event.code === hotkeysConfig.canvas_hotkey_reset) {
                event.preventDefault();
                resetZoom();
            } else if (event.code === hotkeysConfig.canvas_hotkey_fullscreen) {
                event.preventDefault();
                fitToScreen();
            } else if (event.code === hotkeysConfig.canvas_zoom_hotkey_undo) {
                undoLastAction(event);
            }
        }

        function handleMoveKeyDown(event) {
            if (
                event.code === hotkeysConfig.canvas_hotkey_move &&
                !event.ctrlKey &&
                !event.metaKey &&
                isKeyDownHandlerAttached
            ) {
                event.preventDefault();
                isMoving = true;
            }
        }

        function handleMoveKeyUp(event) {
            if (event.code === hotkeysConfig.canvas_hotkey_move) {
                isMoving = false;
            }
        }

        function handleMove(event) {
            if (!isMoving || activeElement !== elemId) return;

            const data = elemData[elemId];
            data.panX += event.movementX * 2;
            data.panY += event.movementY * 2;
            targetElement.style.transform =
                `translate(${data.panX}px, ${data.panY}px) scale(${data.zoomLevel})`;
            targetElement.style.pointerEvents = "none";
            targetElement.style.overflow = "visible";
            updateOverlayGeometry();
        }

        function handleMouseEnter() {
            if (!isKeyDownHandlerAttached) {
                document.addEventListener("keydown", handleKeyDown);
                isKeyDownHandlerAttached = true;
                activeElement = elemId;
            }
        }

        function handleMouseLeave() {
            if (isKeyDownHandlerAttached) {
                document.removeEventListener("keydown", handleKeyDown);
                isKeyDownHandlerAttached = false;
                activeElement = null;
            }
            if (isEraserMode) clearOverlay();
            targetElement.style.pointerEvents = "auto";
        }

        function handleWheel(event) {
            if (isModifierKey(event, hotkeysConfig.canvas_hotkey_adjust)) {
                event.preventDefault();
                adjustBrushSize(event);
                return;
            }
            changeZoomLevel(event.deltaY > 0 ? "-" : "+", event);
        }

        // Важно: без capture-фазы и без stopImmediatePropagation().
        targetElement.addEventListener("pointerdown", handleEraserDown);
        targetElement.addEventListener("pointermove", handleEraserMove);
        window.addEventListener("pointerup", handleEraserUp);
        window.addEventListener("pointercancel", handleEraserUp);

        targetElement.addEventListener("mouseenter", handleMouseEnter);
        targetElement.addEventListener("mouseleave", handleMouseLeave);
        targetElement.addEventListener("mousemove", handleMove);
        targetElement.addEventListener("wheel", handleWheel, { passive: false });
        targetElement.addEventListener("auxclick", undoLastAction);

        document.addEventListener("keydown", handleMoveKeyDown);
        document.addEventListener("keyup", handleMoveKeyUp);

        window.addEventListener("resize", () => {
            resetZoom();
            updateOverlayGeometry();
        });

        const fileInput = targetElement.querySelector("input[type='file']");
        if (fileInput) fileInput.addEventListener("click", resetZoom);

        const observer = new MutationObserver(() => {
            updateOverlayGeometry();
        });
        observer.observe(targetElement, {
            attributes: true,
            childList: true,
            subtree: true,
        });

        // Автоматически подгоняем большие изображения после их появления.
        if (hotkeysConfig.canvas_auto_expand) {
            targetElement.addEventListener("mousemove", () => {
                if (hasHorizontalScrollbar(targetElement) && !targetElement.isExpanded) {
                    targetElement.isExpanded = true;
                    setTimeout(() => {
                        fitToScreen();
                        resetZoom();
                    }, 10);
                }
            });
        }
    }

    applyZoomAndPan("#inpaint_canvas");
    applyZoomAndPan("#inpaint_mask_canvas");
    applyZoomAndPan("#cleaner_canvas");
    applyZoomAndPan("#cleaner_video_canvas");
});
