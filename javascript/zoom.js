onUiLoaded(async() => {
    function hasHorizontalScrollbar(element) {
        return element.scrollWidth > element.clientWidth;
    }

    function isModifierKey(event, key) {
        switch (key) {
        case "Ctrl":
            return event.ctrlKey;
        case "Shift":
            return event.shiftKey;
        case "Alt":
            return event.altKey;
        default:
            return false;
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

        if (!targetElement) {
            console.log("Element not found");
            return;
        }

        targetElement.style.transformOrigin = "0 0";

        elemData[elemId] = {
            zoom: 1,
            panX: 0,
            panY: 0
        };

        let fullScreenMode = false;

        // === НОВОЕ: Кастомный курсор для ластика ===
        let eraserCursor = null;
        let eraserCursorSize = 40;
        let eraserCursorColor = '#ffffff';
        let isShiftPressed = false;
        
        // Получаем текущий размер кисти
        function getCurrentBrushSize() {
            const input = gradioApp().querySelector(
                `${elemId} input[aria-label='Brush radius']`
            );
            if (input) {
                const radius = parseFloat(input.value) || 20;
                return radius * 2;
            }
            return 40;
        }
        
        // Получаем цвет кисти
        function getBrushColor() {
            // Пробуем найти input типа color
            const colorInput = targetElement.querySelector('input[type="color"]');
            if (colorInput && colorInput.value) {
                return colorInput.value;
            }
            
            // Пробуем найти кнопку с цветом кисти
            const colorBtn = targetElement.querySelector('[class*="color"]');
            if (colorBtn) {
                const style = window.getComputedStyle(colorBtn);
                if (style.backgroundColor && style.backgroundColor !== 'rgba(0, 0, 0, 0)') {
                    return style.backgroundColor;
                }
            }
            
            // По умолчанию белый
            return '#ffffff';
        }
        
        function createEraserCursor() {
            if (eraserCursor) return;
            
            eraserCursor = document.createElement('div');
            eraserCursor.className = 'eraser-cursor';
            eraserCursor.style.cssText = `
                position: fixed;
                pointer-events: none;
                border: 2px solid ${eraserCursorColor};
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.05);
                transform: translate(-50%, -50%);
                z-index: 99999;
                display: none;
                box-shadow: 0 0 4px rgba(0, 0, 0, 0.5);
            `;
            document.body.appendChild(eraserCursor);
        }
        
        function showEraserCursor(x, y, size, color) {
            if (!eraserCursor) createEraserCursor();
            
            eraserCursorSize = size;
            eraserCursorColor = color;
            eraserCursor.style.width = size + 'px';
            eraserCursor.style.height = size + 'px';
            eraserCursor.style.left = x + 'px';
            eraserCursor.style.top = y + 'px';
            eraserCursor.style.borderColor = color;
            eraserCursor.style.display = 'block';
            
            targetElement.style.cursor = 'none';
        }
        
        function hideEraserCursor() {
            if (eraserCursor) {
                eraserCursor.style.display = 'none';
            }
            targetElement.style.cursor = 'crosshair';
        }
        
        function updateEraserCursor(x, y) {
            if (eraserCursor && eraserCursor.style.display !== 'none') {
                eraserCursor.style.left = x + 'px';
                eraserCursor.style.top = y + 'px';
                
                // Обновляем размер если изменился
                const newBrushSize = getCurrentBrushSize();
                if (newBrushSize !== eraserCursorSize) {
                    eraserCursor.style.width = newBrushSize + 'px';
                    eraserCursor.style.height = newBrushSize + 'px';
                    eraserCursorSize = newBrushSize;
                }
                
                // Обновляем цвет если изменился
                const newBrushColor = getBrushColor();
                if (newBrushColor !== eraserCursorColor) {
                    eraserCursor.style.borderColor = newBrushColor;
                    eraserCursorColor = newBrushColor;
                }
            }
        }
        
        // === НОВОЕ: Обработчики клавиши Shift ===
        const shiftKeyDownHandler = (e) => {
            if (e.key === 'Shift' && !isShiftPressed && activeElement === elemId) {
                isShiftPressed = true;
                
                const brushSize = getCurrentBrushSize();
                const brushColor = getBrushColor();
                
                showEraserCursor(e.clientX, e.clientY, brushSize, brushColor);
            }
        };
        
        const shiftKeyUpHandler = (e) => {
            if (e.key === 'Shift') {
                isShiftPressed = false;
                hideEraserCursor();
            }
        };
        
        const mouseMoveHandler = (e) => {
            if (isShiftPressed && activeElement === elemId) {
                updateEraserCursor(e.clientX, e.clientY);
            }
        };
        
        document.addEventListener('keydown', shiftKeyDownHandler);
        document.addEventListener('keyup', shiftKeyUpHandler);
        document.addEventListener('mousemove', mouseMoveHandler);

        function createTooltip() {
            const toolTipElemnt = targetElement.querySelector(".image-container");
            const tooltip = document.createElement("div");
            tooltip.className = "canvas-tooltip";

            const info = document.createElement("i");
            info.className = "canvas-tooltip-info";
            info.textContent = "";

            const tooltipContent = document.createElement("div");
            tooltipContent.className = "canvas-tooltip-content";

            const hotkeysInfo = [
                {
                    configKey: "canvas_hotkey_zoom",
                    action: "Zoom canvas",
                    keySuffix: " + wheel"
                },
                {
                    configKey: "canvas_hotkey_adjust",
                    action: "Adjust brush size",
                    keySuffix: " + wheel"
                },
                {configKey: "canvas_zoom_hotkey_undo", action: "Undo last action", keyPrefix: `${hotkeysConfig.canvas_zoom_undo_extra_key} + ` },
                {configKey: "canvas_hotkey_reset", action: "Reset zoom"},
                {
                    configKey: "canvas_hotkey_fullscreen",
                    action: "Fullscreen mode"
                },
                {configKey: "canvas_hotkey_move", action: "Move canvas"},
                {action: "Eraser (hold Shift)", key: "Shift"}
            ];

            const hotkeys = hotkeysInfo.map((info) => {
                const configValue = hotkeysConfig[info.configKey];
        
                let key = info.key || configValue.slice(-1);
        
                if (info.keySuffix) {
                  key = `${configValue}${info.keySuffix}`;
                }
        
                if (info.keyPrefix && info.keyPrefix !== "None + ") {
                  key = `${info.keyPrefix}${configValue[3]}`;
                }
        
                return {
                  key,
                  action: info.action,
                };
              });
        
              hotkeys.forEach(hotkey => {
                  const p = document.createElement("p");
                  p.innerHTML = `<b>${hotkey.key}</b> - ${hotkey.action}`;
                  tooltipContent.appendChild(p);
              });
        
              tooltip.append(info, tooltipContent);
              toolTipElemnt.appendChild(tooltip);
        }

        if (hotkeysConfig.canvas_show_tooltip) {
            createTooltip();
        }

        function resetZoom() {
            elemData[elemId] = {
                zoomLevel: 1,
                panX: 0,
                panY: 0
            };

            targetElement.style.overflow = "hidden";
            targetElement.isZoomed = false;
            targetElement.style.transform = `scale(${elemData[elemId].zoomLevel}) translate(${elemData[elemId].panX}px, ${elemData[elemId].panY}px)`;

            const canvas = gradioApp().querySelector(
                `${elemId} canvas[key="interface"]`
            );

            toggleOverlap("off");
            fullScreenMode = false;

            const closeBtn = targetElement.querySelector("button[aria-label='Remove Image']");
            if (closeBtn) {
                closeBtn.addEventListener("click", resetZoom);
            }

            if (canvas) {
                const parentElement = targetElement.closest('[id^="component-"]');
                if (
                    canvas &&
                    parseFloat(canvas.style.width) > parentElement.offsetWidth &&
                    parseFloat(targetElement.style.width) > parentElement.offsetWidth
                ) {
                    fitToElement();
                    return;
                }
            }

            targetElement.style.width = "";
        }

        function toggleOverlap(forced = "") {
            const zIndex1 = "0";
            const zIndex2 = "998";

            targetElement.style.zIndex =
                targetElement.style.zIndex !== zIndex2 ? zIndex2 : zIndex1;

            if (forced === "off") {
                targetElement.style.zIndex = zIndex1;
            } else if (forced === "on") {
                targetElement.style.zIndex = zIndex2;
            }
        }

        function adjustBrushSize(elemId, deltaY, withoutValue = false, percentage = 5) {
            const input =
                gradioApp().querySelector(
                    `${elemId} input[aria-label='Brush radius']`
                ) ||
                gradioApp().querySelector(
                    `${elemId} button[aria-label="Use brush"]`
                );

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

        const fileInput = gradioApp().querySelector(
            `${elemId} input[type="file"][accept="image/*"].svelte-116rqfv`
        );
        if (fileInput) {
            fileInput.addEventListener("click", resetZoom);
        }

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
            // Не зумим если зажат Shift (режим ластика)
            if (isShiftPressed) {
                return;
            }
            
            if (isModifierKey(e, hotkeysConfig.canvas_hotkey_zoom)) {
                e.preventDefault();

                let zoomPosX, zoomPosY;
                let delta = 0.2;

                if (elemData[elemId].zoomLevel > 7) {
                    delta = 0.9;
                } else if (elemData[elemId].zoomLevel > 2) {
                    delta = 0.6;
                }

                zoomPosX = e.clientX;
                zoomPosY = e.clientY;

                fullScreenMode = false;
                elemData[elemId].zoomLevel = updateZoom(
                    elemData[elemId].zoomLevel + (operation === "+" ? delta : -delta),
                    zoomPosX - targetElement.getBoundingClientRect().left,
                    zoomPosY - targetElement.getBoundingClientRect().top
                );

                targetElement.isZoomed = true;
            }
        }

        function fitToElement() {
            targetElement.style.transform = `translate(${0}px, ${0}px) scale(${1})`;

            let parentElement = targetElement.closest('[id^="component-"]');

            const elementWidth = targetElement.offsetWidth;
            const elementHeight = targetElement.offsetHeight;

            const screenWidth = parentElement.clientWidth - 24;
            const screenHeight = parentElement.clientHeight;

            const scaleX = screenWidth / elementWidth;
            const scaleY = screenHeight / elementHeight;
            const scale = Math.min(scaleX, scaleY);

            const offsetX = 0;
            const offsetY = 0;

            targetElement.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;

            elemData[elemId].zoomLevel = scale;
            elemData[elemId].panX = offsetX;
            elemData[elemId].panY = offsetY;

            fullScreenMode = false;
            toggleOverlap("off");
        }

        function undoLastAction(e) {
            let isCtrlPressed = isModifierKey(e, hotkeysConfig.canvas_zoom_undo_extra_key)
            const isAuxButton = e.button >= 3;
            
            if (isAuxButton) {
              isCtrlPressed = true
            } else {
              if (!isModifierKey(e, hotkeysConfig.canvas_zoom_undo_extra_key)) return;
            }

            const undoBtn = document.querySelector(`${activeElement} button[aria-label="Undo"]`);
        
            if ((isCtrlPressed) && undoBtn ) {
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

            const elementWidth = targetElement.offsetWidth;
            const elementHeight = targetElement.offsetHeight;
            const screenWidth = window.innerWidth - scrollbarWidth;
            const screenHeight = window.innerHeight;

            const elementRect = targetElement.getBoundingClientRect();
            const elementY = elementRect.y;
            const elementX = elementRect.x;

            const scaleX = screenWidth / elementWidth;
            const scaleY = screenHeight / elementHeight;
            const scale = Math.min(scaleX, scaleY);

            const computedStyle = window.getComputedStyle(targetElement);
            const transformOrigin = computedStyle.transformOrigin;
            const [originX, originY] = transformOrigin.split(" ");
            const originXValue = parseFloat(originX);
            const originYValue = parseFloat(originY);

            const offsetX = (screenWidth - elementWidth * scale) / 2 - elementX - originXValue * (1 - scale);
            const offsetY = (screenHeight - elementHeight * scale) / 2 - elementY - originYValue * (1 - scale);

            targetElement.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;

            elemData[elemId].zoomLevel = scale;
            elemData[elemId].panX = offsetX;
            elemData[elemId].panY = offsetY;

            fullScreenMode = true;
            toggleOverlap("on");
        }

        function handleKeyDown(event) {
            if ((event.ctrlKey && event.code === 'KeyV') || (event.ctrlKey && event.code === 'KeyC') || event.code === "F5") {
                return;
            }

            if (!hotkeysConfig.canvas_blur_prompt) {
                if (event.target.nodeName === 'TEXTAREA' || event.target.nodeName === 'INPUT') {
                    return;
                }
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

        function getMousePosition(e) {
            mouseX = e.offsetX;
            mouseY = e.offsetY;
        }

        targetElement.isExpanded = false;
        function autoExpand() {
            const canvas = document.querySelector(`${elemId} canvas[key="interface"]`);
            if (canvas) {
                if (hasHorizontalScrollbar(targetElement) && targetElement.isExpanded === false) {
                    targetElement.style.visibility = "hidden";
                    setTimeout(() => {
                        fitToScreen();
                        resetZoom();
                        targetElement.style.visibility = "visible";
                        targetElement.isExpanded = true;
                    }, 10);
                }
            }
        }

        targetElement.addEventListener("mousemove", getMousePosition);
        targetElement.addEventListener("auxclick", undoLastAction);

        const observer = new MutationObserver((mutationsList, observer) => {
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

        // === ИЗМЕНЕНО: Ластик на Shift + Click ===
        let eraserLastX = 0;
        let eraserLastY = 0;
        let isEraserDrawing = false;
        
        function getEraserCoordinates(e, canvas) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            return {
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            };
        }
        
        function startErasing(e) {
            // Проверяем, что нажат Shift и левая кнопка мыши
            if (!e.shiftKey || e.button !== 0) return;
            
            e.preventDefault();
            e.stopPropagation();
            
            isEraserDrawing = true;
            
            const maskCanvas = targetElement.querySelector('canvas[key="mask"]');
            const drawingCanvas = targetElement.querySelector('canvas[key="drawing"]');
            
            if (!maskCanvas || !drawingCanvas) {
                console.warn('[Eraser] Canvas не найдены');
                return;
            }
            
            const maskCtx = maskCanvas.getContext('2d');
            const drawingCtx = drawingCanvas.getContext('2d');
            
            const pos = getEraserCoordinates(e, drawingCanvas);
            eraserLastX = pos.x;
            eraserLastY = pos.y;
            
            const brushSize = getCurrentBrushSize();
            
            maskCtx.save();
            maskCtx.globalCompositeOperation = 'destination-out';
            maskCtx.beginPath();
            maskCtx.arc(eraserLastX, eraserLastY, brushSize / 2, 0, Math.PI * 2);
            maskCtx.fill();
            maskCtx.restore();
            
            drawingCtx.save();
            drawingCtx.globalCompositeOperation = 'destination-out';
            drawingCtx.beginPath();
            drawingCtx.arc(eraserLastX, eraserLastY, brushSize / 2, 0, Math.PI * 2);
            drawingCtx.fill();
            drawingCtx.restore();
        }
        
        function continueErasing(e) {
            if (!isEraserDrawing || !e.shiftKey) {
                if (isEraserDrawing) {
                    stopErasing(e);
                }
                return;
            }
            
            e.preventDefault();
            e.stopPropagation();
            
            const maskCanvas = targetElement.querySelector('canvas[key="mask"]');
            const drawingCanvas = targetElement.querySelector('canvas[key="drawing"]');
            
            if (!maskCanvas || !drawingCanvas) return;
            
            const maskCtx = maskCanvas.getContext('2d');
            const drawingCtx = drawingCanvas.getContext('2d');
            
            const pos = getEraserCoordinates(e, drawingCanvas);
            const brushSize = getCurrentBrushSize();
            
            maskCtx.save();
            maskCtx.globalCompositeOperation = 'destination-out';
            maskCtx.beginPath();
            maskCtx.moveTo(eraserLastX, eraserLastY);
            maskCtx.lineTo(pos.x, pos.y);
            maskCtx.lineWidth = brushSize;
            maskCtx.lineCap = 'round';
            maskCtx.lineJoin = 'round';
            maskCtx.stroke();
            maskCtx.restore();
            
            drawingCtx.save();
            drawingCtx.globalCompositeOperation = 'destination-out';
            drawingCtx.beginPath();
            drawingCtx.moveTo(eraserLastX, eraserLastY);
            drawingCtx.lineTo(pos.x, pos.y);
            drawingCtx.lineWidth = brushSize;
            drawingCtx.lineCap = 'round';
            drawingCtx.lineJoin = 'round';
            drawingCtx.stroke();
            drawingCtx.restore();
            
            eraserLastX = pos.x;
            eraserLastY = pos.y;
        }
        
        function stopErasing(e) {
            if (!isEraserDrawing) return;
            
            isEraserDrawing = false;
            
            const maskCanvas = targetElement.querySelector('canvas[key="mask"]');
            const drawingCanvas = targetElement.querySelector('canvas[key="drawing"]');
            
            if (drawingCanvas) {
                drawingCanvas.dispatchEvent(new Event('input', { bubbles: true }));
                drawingCanvas.dispatchEvent(new Event('change', { bubbles: true }));
            }
            if (maskCanvas) {
                maskCanvas.dispatchEvent(new Event('input', { bubbles: true }));
                maskCanvas.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
        
        targetElement.addEventListener('pointerdown', startErasing, true);
        targetElement.addEventListener('pointermove', continueErasing, true);
        targetElement.addEventListener('pointerup', stopErasing, true);
        targetElement.addEventListener('pointerleave', stopErasing, true);

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
            if ((e.ctrlKey && e.code === 'KeyV') || (e.ctrlKey && e.code === 'KeyC') || e.code === "F5") {
                return;
            }

            if (!hotkeysConfig.canvas_blur_prompt) {
                if (e.target.nodeName === 'TEXTAREA' || e.target.nodeName === 'INPUT') {
                    return;
                }
            }

            if (e.code === hotkeysConfig.canvas_hotkey_move) {
                if (!e.ctrlKey && !e.metaKey && isKeyDownHandlerAttached) {
                    e.preventDefault();
                    document.activeElement.blur();
                    isMoving = true;
                }
            }
        }

        function handleMoveKeyUp(e) {
            if (e.code === hotkeysConfig.canvas_hotkey_move) {
                isMoving = false;
            }
        }

        document.addEventListener("keydown", handleMoveKeyDown);
        document.addEventListener("keyup", handleMoveKeyUp);

        function updatePanPosition(movementX, movementY) {
            let panSpeed = 2;

            if (elemData[elemId].zoomLevel > 8) {
                panSpeed = 3.5;
            }

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
            isShiftPressed = false;
            hideEraserCursor();
            if (isEraserDrawing) {
                stopErasing(new Event('pointerup'));
            }
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