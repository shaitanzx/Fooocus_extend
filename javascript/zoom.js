onUiLoaded(async() => {
    function hasHorizontalScrollbar(element) {
        return element.scrollWidth > element.clientWidth;
    }

    function isModifierKey(event, key) {
        switch (key) {
        case "Ctrl": return event.ctrlKey;
        case "Shift": return event.shiftKey;
        case "Alt": return event.altKey;
        default: return false;
        }
    }

    function createHotkeyConfig(defaultHotkeysConfig) {
        const result = {};
        for (const key in defaultHotkeysConfig) {
            result[key] = defaultHotkeysConfig[key];
        }
        return result;
    }

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

    const hotkeysConfig = createHotkeyConfig(defaultHotkeysConfig);

    let isMoving = false;
    let activeElement;
    const elemData = {};

    function applyZoomAndPan(elemId) {
        const targetElement = gradioApp().querySelector(elemId);
        if (!targetElement) return;

        targetElement.style.transformOrigin = "0 0";
        elemData[elemId] = { zoom: 1, panX: 0, panY: 0 };
        let fullScreenMode = false;

        let isEraserMode = false;
        let isDrawingEraser = false;
        let lastEraserX = 0;
        let lastEraserY = 0;

        function getBrushRadius(root) {
            const input = root.querySelector("input[aria-label='Brush radius']");
            const value = input ? parseFloat(input.value) : NaN;
            const diameter = Number.isFinite(value) && value > 0 ? value : 20;
            return diameter / 2;
        }

        function getCanvasPoint(canvas, event) {
            const rect = canvas.getBoundingClientRect();
            return {
                x: (event.clientX - rect.left) * (canvas.width / rect.width),
                y: (event.clientY - rect.top) * (canvas.height / rect.height)
            };
        }

        function eraseAt(canvas, x, y, radius) {
            const ctx = canvas.getContext('2d');
            ctx.save();
            ctx.globalCompositeOperation = 'destination-out';
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        }

        function eraseLine(canvas, x1, y1, x2, y2, radius) {
            const ctx = canvas.getContext('2d');
            ctx.save();
            ctx.globalCompositeOperation = 'destination-out';
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.lineWidth = radius * 2;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.stroke();
            ctx.restore();
        }

        function handleEraserDown(e) {
            if (!isEraserMode || e.button !== 0) return;
            
            const maskCanvas = targetElement.querySelector('canvas[key="mask"]');
            const drawingCanvas = targetElement.querySelector('canvas[key="drawing"]') || targetElement.querySelector('canvas[key="interface"]');
            if (!maskCanvas || !drawingCanvas) return;

            e.preventDefault();
            e.stopImmediatePropagation();
            isDrawingEraser = true;

            const pos = getCanvasPoint(drawingCanvas, e);
            lastEraserX = pos.x;
            lastEraserY = pos.y;
            const radius = getBrushRadius(targetElement);

            eraseAt(maskCanvas, lastEraserX, lastEraserY, radius);
            eraseAt(drawingCanvas, lastEraserX, lastEraserY, radius);
        }

        function handleEraserMove(e) {
            if (isEraserMode && e.target.tagName === 'CANVAS') {
                e.preventDefault();
                e.stopImmediatePropagation();
            }
            
            if (!isEraserMode || !isDrawingEraser) return;

            const maskCanvas = targetElement.querySelector('canvas[key="mask"]');
            const drawingCanvas = targetElement.querySelector('canvas[key="drawing"]') || targetElement.querySelector('canvas[key="interface"]');
            if (!maskCanvas || !drawingCanvas) return;

            const pos = getCanvasPoint(drawingCanvas, e);
            const radius = getBrushRadius(targetElement);

            eraseLine(maskCanvas, lastEraserX, lastEraserY, pos.x, pos.y, radius);
            eraseLine(drawingCanvas, lastEraserX, lastEraserY, pos.x, pos.y, radius);

            lastEraserX = pos.x;
            lastEraserY = pos.y;
        }

        function handleEraserUp(e) {
            if (!isDrawingEraser) return;
            isDrawingEraser = false;
            
            const drawingCanvas = targetElement.querySelector('canvas[key="drawing"]') || targetElement.querySelector('canvas[key="interface"]');
            const maskCanvas = targetElement.querySelector('canvas[key="mask"]');
            
            if (drawingCanvas) {
                drawingCanvas.dispatchEvent(new Event('input', { bubbles: true }));
                drawingCanvas.dispatchEvent(new Event('change', { bubbles: true }));
            }
            if (maskCanvas) {
                maskCanvas.dispatchEvent(new Event('input', { bubbles: true }));
                maskCanvas.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }

        function createTooltip() {
            const toolTipElemnt = targetElement.querySelector(".image-container");
            if (!toolTipElemnt) return;
            
            const tooltip = document.createElement("div");
            tooltip.className = "canvas-tooltip";
            const info = document.createElement("i");
            info.className = "canvas-tooltip-info";
            info.textContent = "";
            const tooltipContent = document.createElement("div");
            tooltipContent.className = "canvas-tooltip-content";

            const hotkeysInfo = [
                { configKey: "canvas_hotkey_zoom", action: "Zoom canvas", keySuffix: " + wheel" },
                { configKey: "canvas_hotkey_adjust", action: "Adjust brush size", keySuffix: " + wheel" },
                { configKey: "canvas_zoom_hotkey_undo", action: "Undo last action", keyPrefix: `${hotkeysConfig.canvas_zoom_undo_extra_key} + ` },
                { configKey: "canvas_hotkey_reset", action: "Reset zoom" },
                { configKey: "canvas_hotkey_fullscreen", action: "Fullscreen mode" },
                { configKey: "canvas_hotkey_move", action: "Move canvas" },
                { configKey: "canvas_hotkey_eraser", action: "Toggle Eraser Mode" }
            ];

            const hotkeys = hotkeysInfo.map((info) => {
                const configValue = hotkeysConfig[info.configKey];
                let key = configValue.slice(-1);
                if (info.keySuffix) key = `${configValue}${info.keySuffix}`;
                if (info.keyPrefix && info.keyPrefix !== "None + ") key = `${info.keyPrefix}${configValue.slice(3)}`;
                return { key, action: info.action };
            });

            hotkeys.forEach(hotkey => {
                const p = document.createElement("p");
                p.innerHTML = `<b>${hotkey.key}</b> - ${hotkey.action}`;
                tooltipContent.appendChild(p);
            });

            tooltip.append(info, tooltipContent);
            toolTipElemnt.appendChild(tooltip);
        }

        if (hotkeysConfig.canvas_show_tooltip) createTooltip();

        function resetZoom() {
            elemData[elemId] = { zoomLevel: 1, panX: 0, panY: 0 };
            targetElement.style.overflow = "hidden";
            targetElement.isZoomed = false;
            targetElement.style.transform = `scale(${elemData[elemId].zoomLevel}) translate(${elemData[elemId].panX}px, ${elemData[elemId].panY}px)`;
            toggleOverlap("off");
            fullScreenMode = false;
            const closeBtn = targetElement.querySelector("button[aria-label='Remove Image']");
            if (closeBtn) closeBtn.addEventListener("click", resetZoom);
            targetElement.style.width = "";
        }

        function toggleOverlap(forced = "") {
            const zIndex1 = "0";
            const zIndex2 = "998";
            targetElement.style.zIndex = targetElement.style.zIndex !== zIndex2 ? zIndex2 : zIndex1;
            if (forced === "off") targetElement.style.zIndex = zIndex1;
            else if (forced === "on") targetElement.style.zIndex = zIndex2;
        }

        function adjustBrushSize(elemId, deltaY, withoutValue = false, percentage = 5) {
            const input = gradioApp().querySelector(`${elemId} input[aria-label='Brush radius']`) ||
                          gradioApp().querySelector(`${elemId} button[aria-label="Use brush"]`);
            if (input) {
                input.click();
                if (!withoutValue) {
                    const maxValue = parseFloat(input.getAttribute("max")) || 100;
                    const changeAmount = maxValue * (percentage / 100);
                    const newValue = parseFloat(input.value) + (deltaY > 0 ? -changeAmount : changeAmount);
                    input.value = Math.min(Math.max(newValue, 0), maxValue);
                    input.dispatchEvent(new Event("change"));
                }
            }
        }

        const fileInput = gradioApp().querySelector(`${elemId} input[type="file"][accept="image/*"].svelte-116rqfv`);
        if (fileInput) fileInput.addEventListener("click", resetZoom);

        function updateZoom(newZoomLevel, mouseX, mouseY) {
            newZoomLevel = Math.max(0.1, Math.min(newZoomLevel, 15));
            elemData[elemId].panX += mouseX - (mouseX * newZoomLevel) / elemData[elemId].zoomLevel;
            elemData[elemId].panY += mouseY - (mouseY * newZoomLevel) / elemData[elemId].zoomLevel;
            targetElement.style.transformOrigin = "0 0";
            targetElement.style.transform = `translate(${elemData[elemId].panX}px, ${elemData[elemId].panY}px) scale(${newZoomLevel})`;
            targetElement.style.overflow = "visible";
            toggleOverlap("on");
            return newZoomLevel;
        }

        function changeZoomLevel(operation, e) {
            if (isModifierKey(e, hotkeysConfig.canvas_hotkey_zoom)) {
                e.preventDefault();
                let delta = 0.2;
                if (elemData[elemId].zoomLevel > 7) delta = 0.9;
                else if (elemData[elemId].zoomLevel > 2) delta = 0.6;
                fullScreenMode = false;
                elemData[elemId].zoomLevel = updateZoom(
                    elemData[elemId].zoomLevel + (operation === "+" ? delta : -delta),
                    e.clientX - targetElement.getBoundingClientRect().left,
                    e.clientY - targetElement.getBoundingClientRect().top
                );
                targetElement.isZoomed = true;
            }
        }

        function fitToElement() {
            targetElement.style.transform = `translate(${0}px, ${0}px) scale(${1})`;
            let parentElement = targetElement.closest('[id^="component-"]');
            const scale = Math.min((parentElement.clientWidth - 24) / targetElement.offsetWidth, parentElement.clientHeight / targetElement.offsetHeight);
            targetElement.style.transform = `translate(${0}px, ${0}px) scale(${scale})`;
            elemData[elemId].zoomLevel = scale;
            elemData[elemId].panX = 0;
            elemData[elemId].panY = 0;
            fullScreenMode = false;
            toggleOverlap("off");
        }

        function undoLastAction(e) {
            let isCtrlPressed = isModifierKey(e, hotkeysConfig.canvas_zoom_undo_extra_key);
            if (e.button >= 3) isCtrlPressed = true;
            else if (!isModifierKey(e, hotkeysConfig.canvas_zoom_undo_extra_key)) return;
            const undoBtn = document.querySelector(`${activeElement} button[aria-label="Undo"]`);
            if (isCtrlPressed && undoBtn) {
                e.preventDefault();
                undoBtn.click();
            }
        }

        function fitToScreen() {
            const canvas = gradioApp().querySelector(`${elemId} canvas[key="interface"]`);
            if (!canvas) return;
            targetElement.style.width = (canvas.offsetWidth + 2) + "px";
            targetElement.style.overflow = "visible";
            if (fullScreenMode) {
                resetZoom();
                fullScreenMode = false;
                return;
            }
            targetElement.style.transform = `translate(${0}px, ${0}px) scale(${1})`;
            const scrollbarWidth = window.innerWidth - document.documentElement.clientWidth;
            const elementRect = targetElement.getBoundingClientRect();
            const scale = Math.min((window.innerWidth - scrollbarWidth) / targetElement.offsetWidth, window.innerHeight / targetElement.offsetHeight);
            const offsetX = (window.innerWidth - scrollbarWidth - targetElement.offsetWidth * scale) / 2 - elementRect.x;
            const offsetY = (window.innerHeight - targetElement.offsetHeight * scale) / 2 - elementRect.y;
            targetElement.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;
            elemData[elemId].zoomLevel = scale;
            elemData[elemId].panX = offsetX;
            elemData[elemId].panY = offsetY;
            fullScreenMode = true;
            toggleOverlap("on");
        }

        function handleKeyDown(event) {
            if ((event.ctrlKey && event.code === 'KeyV') || (event.ctrlKey && event.code === 'KeyC') || event.code === "F5") return;
            if (!hotkeysConfig.canvas_blur_prompt && (event.target.nodeName === 'TEXTAREA' || event.target.nodeName === 'INPUT')) return;

            if (event.code === hotkeysConfig.canvas_hotkey_eraser && activeElement === elemId) {
                event.preventDefault();
                isEraserMode = !isEraserMode;
                targetElement.style.outline = isEraserMode ? "3px solid #ff4444" : "none";
                targetElement.style.outlineOffset = "-3px";
                return;
            }

            const hotkeyActions = {
                [hotkeysConfig.canvas_hotkey_reset]: resetZoom,
                [hotkeysConfig.canvas_hotkey_overlap]: toggleOverlap,
                [hotkeysConfig.canvas_hotkey_fullscreen]: fitToScreen,
                [hotkeysConfig.canvas_zoom_hotkey_undo]: undoLastAction,
            };
            const action = hotkeyActions[event.code];
            if (action) {
                event.preventDefault();
                action(event);
            }
            if (isModifierKey(event, hotkeysConfig.canvas_hotkey_zoom) || isModifierKey(event, hotkeysConfig.canvas_hotkey_adjust)) {
                event.preventDefault();
            }
        }

        targetElement.isExpanded = false;
        function autoExpand() {
            const canvas = document.querySelector(`${elemId} canvas[key="interface"]`);
            if (canvas && hasHorizontalScrollbar(targetElement) && !targetElement.isExpanded) {
                targetElement.style.visibility = "hidden";
                setTimeout(() => {
                    fitToScreen();
                    resetZoom();
                    targetElement.style.visibility = "visible";
                    targetElement.isExpanded = true;
                }, 10);
            }
        }

        targetElement.addEventListener("mousemove", (e) => { mouseX = e.offsetX; mouseY = e.offsetY; });
        targetElement.addEventListener("auxclick", undoLastAction);

        const observer = new MutationObserver((mutationsList) => {
            for (let mutation of mutationsList) {
              if (mutation.type === 'attributes' && mutation.attributeName === 'style' && mutation.target.tagName.toLowerCase() === 'canvas') {
                targetElement.isExpanded = false;
                setTimeout(resetZoom, 10);
              }
            }
        });
      
        if (hotkeysConfig.canvas_auto_expand) {
            targetElement.addEventListener("mousemove", autoExpand);
            observer.observe(targetElement, { attributes: true, childList: true, subtree: true });
        }

        let isKeyDownHandlerAttached = false;
        function handleMouseMove() {
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
        }

        targetElement.addEventListener('pointerdown', handleEraserDown, true);
        targetElement.addEventListener('pointermove', handleEraserMove, true);
        window.addEventListener('pointerup', handleEraserUp, true);

        targetElement.addEventListener("mousemove", handleMouseMove);
        targetElement.addEventListener("mouseleave", handleMouseLeave);

        targetElement.addEventListener("wheel", e => {
            const operation = e.deltaY > 0 ? "-" : "+";
            changeZoomLevel(operation, e);
            if (isModifierKey(e, hotkeysConfig.canvas_hotkey_adjust)) {
                e.preventDefault();
                adjustBrushSize(elemId, e.deltaY);
            }
        });

        function handleMoveKeyDown(e) {
            if ((e.ctrlKey && e.code === 'KeyV') || (e.ctrlKey && e.code === 'KeyC') || e.code === "F5") return;
            if (!hotkeysConfig.canvas_blur_prompt && (e.target.nodeName === 'TEXTAREA' || e.target.nodeName === 'INPUT')) return;
            if (e.code === hotkeysConfig.canvas_hotkey_move) {
                if (!e.ctrlKey && !e.metaKey && isKeyDownHandlerAttached) {
                    e.preventDefault();
                    document.activeElement.blur();
                    isMoving = true;
                }
            }
        }

        function handleMoveKeyUp(e) {
            if (e.code === hotkeysConfig.canvas_hotkey_move) isMoving = false;
        }

        document.addEventListener("keydown", handleMoveKeyDown);
        document.addEventListener("keyup", handleMoveKeyUp);

        function updatePanPosition(movementX, movementY) {
            let panSpeed = 2;
            if (elemData[elemId].zoomLevel > 8) panSpeed = 3.5;
            elemData[elemId].panX += movementX * panSpeed;
            elemData[elemId].panY += movementY * panSpeed;
            requestAnimationFrame(() => {
                targetElement.style.transform = `translate(${elemData[elemId].panX}px, ${elemData[elemId].panY}px) scale(${elemData[elemId].zoomLevel})`;
                toggleOverlap("on");
            });
        }

        function handleMoveByKey(e) {
            if (isMoving && elemId === activeElement) {
                updatePanPosition(e.movementX, e.movementY);
                targetElement.style.pointerEvents = "none";
                targetElement.style.overflow = "visible";
            } else {
                targetElement.style.pointerEvents = "auto";
            }
        }

        window.onblur = function() {
            isMoving = false;
            isDrawingEraser = false;
        };

        function checkForOutBox() {
            const parentElement = targetElement.closest('[id^="component-"]');
            if (parentElement.offsetWidth < targetElement.offsetWidth && !targetElement.isExpanded) {
                resetZoom();
                targetElement.isExpanded = true;
            }
            if (parentElement.offsetWidth < targetElement.offsetWidth && elemData[elemId].zoomLevel == 1) {
                resetZoom();
            }
            if (parentElement.offsetWidth < targetElement.offsetWidth && targetElement.offsetWidth * elemData[elemId].zoomLevel > parentElement.offsetWidth && elemData[elemId].zoomLevel < 1 && !targetElement.isZoomed) {
                resetZoom();
            }
        }

        targetElement.addEventListener("mousemove", checkForOutBox);

        window.addEventListener('resize', (e) => {
            resetZoom();
            targetElement.isExpanded = false;
            targetElement.isZoomed = false;
        });

        gradioApp().addEventListener("mousemove", handleMoveByKey);
    }

    applyZoomAndPan("#inpaint_canvas");
    applyZoomAndPan("#inpaint_mask_canvas");
    applyZoomAndPan("#cleaner_canvas");
    applyZoomAndPan("#cleaner_video_canvas");
});