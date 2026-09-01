onUiLoaded(function() {
    const ERASER_ICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M16.24 3.56l4.95 4.94c.78.79.78 2.05 0 2.83L12 20.5c-.79.79-2.05.79-2.83 0l-4.95-4.94c-.78-.79-.78-2.05 0-2.83L13.41 3.56c.79-.78 2.05-.78 2.83 0zM4.22 18.79l1.42 1.42L4.22 21.63 2.81 20.22l1.41-1.43z"/></svg>`;

    function injectEraserIntoSketch(container) {
        // Avoid double injection
        if (container.querySelector('.custom-eraser-btn')) return;

        const interfaceCanvas = container.querySelector("canvas[key='interface']");
        const maskCanvas = container.querySelector("canvas[key='mask']");
        if (!interfaceCanvas || !maskCanvas) return;

        // 1. Find or create toolbar area
        let toolbar = container.querySelector('.controls, .tool-buttons');
        if (!toolbar) {
            // Create a floating toolbar if not found
            toolbar = document.createElement('div');
            toolbar.className = 'custom-eraser-toolbar';
            toolbar.style.cssText = 'position: absolute; top: 10px; right: 10px; z-index: 10; display: flex; gap: 5px;';
            container.style.position = 'relative';
            container.appendChild(toolbar);
        }

        // 2. Create Eraser Button
        const eraserBtn = document.createElement('button');
        eraserBtn.className = 'custom-eraser-btn';
        eraserBtn.innerHTML = ERASER_ICON;
        eraserBtn.title = 'Toggle Eraser';
        eraserBtn.style.cssText = 'background: var(--button-secondary-background-fill); border: 1px solid var(--border-color-primary); border-radius: 4px; padding: 4px; cursor: pointer; display: flex; align-items: center;';
        toolbar.prepend(eraserBtn);

        let isErasing = false;
        let isDrawing = false;
        let lastX = 0, lastY = 0;

        eraserBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            isErasing = !isErasing;
            eraserBtn.style.background = isErasing ? 'var(--button-primary-background-fill)' : 'var(--button-secondary-background-fill)';
            interfaceCanvas.style.cursor = isErasing ? 'cell' : 'crosshair';
            
            // Optional: Disable Gradio's default drawing when eraser is active
            if (isErasing) {
                interfaceCanvas.style.pointerEvents = 'none';
                // Re-enable for our custom handlers
                interfaceCanvas.style.pointerEvents = 'auto';
            }
        });

        // 3. Intercept Pointer Events for Erasing
        function getPos(e) {
            const rect = interfaceCanvas.getBoundingClientRect();
            return {
                x: (e.clientX - rect.left) * (interfaceCanvas.width / rect.width),
                y: (e.clientY - rect.top) * (interfaceCanvas.height / rect.height)
            };
        }

        interfaceCanvas.addEventListener('pointerdown', (e) => {
            if (!isErasing) return;
            
            e.preventDefault();
            e.stopPropagation();
            isDrawing = true;
            
            const pos = getPos(e);
            lastX = pos.x;
            lastY = pos.y;
            
            const ctx = maskCanvas.getContext('2d');
            ctx.globalCompositeOperation = 'destination-out';
            ctx.beginPath();
            ctx.arc(lastX, lastY, getBrushRadius(), 0, Math.PI * 2);
            ctx.fill();
        }, true);

        interfaceCanvas.addEventListener('pointermove', (e) => {
            if (!isErasing || !isDrawing) return;
            
            e.preventDefault();
            e.stopPropagation();
            
            const pos = getPos(e);
            const ctx = maskCanvas.getContext('2d');
            ctx.globalCompositeOperation = 'destination-out';
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(pos.x, pos.y);
            ctx.lineWidth = getBrushRadius() * 2;
            ctx.lineCap = 'round';
            ctx.stroke();
            
            lastX = pos.x;
            lastY = pos.y;
        }, true);

        const stopDrawing = (e) => {
            if (!isDrawing) return;
            isDrawing = false;
            // Trigger Gradio update so Python receives the new mask
            interfaceCanvas.dispatchEvent(new Event('change', { bubbles: true }));
        };

        interfaceCanvas.addEventListener('pointerup', stopDrawing, true);
        interfaceCanvas.addEventListener('pointerleave', stopDrawing, true);
    }

    function getBrushRadius() {
        // Try to find the brush radius slider in the component
        const slider = gradioApp().querySelector("input[aria-label='Brush radius']");
        return slider ? parseFloat(slider.value) : 20;
    }

    // Observe DOM to apply to dynamically created sketch components
    const observer = new MutationObserver((mutations) => {
        mutations.forEach(m => {
            m.addedNodes.forEach(node => {
                if (node.nodeType === 1) {
                    if (node.classList && node.classList.contains('gradio-image')) {
                        injectEraserIntoSketch(node);
                    } else {
                        const sketches = node.querySelectorAll ? node.querySelectorAll('.gradio-image') : [];
                        sketches.forEach(injectEraserIntoSketch);
                    }
                }
            });
        });
    });

    observer.observe(gradioApp(), { childList: true, subtree: true });

    // Initial check
    gradioApp().querySelectorAll('.gradio-image').forEach(injectEraserIntoSketch);
});