console.log("🔧 [Eraser Script] Загрузка...");

function initEraser() {
    // Ищем все canvas на странице
    const canvases = document.querySelectorAll('canvas');
    console.log(`[Eraser] Найдено canvas элементов: ${canvases.length}`);

    canvases.forEach(canvas => {
        const parent = canvas.parentElement;
        if (!parent || parent.classList.contains('eraser-attached')) return;

        // Проверяем, похож ли этот canvas на Gradio Sketch (ищем ключи или соседние элементы)
        const isSketch = canvas.getAttribute('key') === 'interface' || 
                         canvas.getAttribute('key') === 'mask' ||
                         parent.querySelector('canvas[key="mask"]') ||
                         parent.querySelector('canvas[key="interface"]');

        if (!isSketch) return;

        console.log("✅ [Eraser] Целевой sketch canvas найден:", canvas);
        parent.classList.add('eraser-attached');

        // 1. Создаем кнопку ластика (яркую, чтобы ее точно было видно)
        const btn = document.createElement('button');
        btn.innerHTML = '🧹 ЛАСТИК';
        btn.className = 'eraser-toggle-btn';
        btn.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 99999;
            padding: 8px 16px;
            background: #ffffff;
            color: #000000;
            border: 2px solid #ff4444;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            font-family: sans-serif;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        `;
        
        // Убедимся, что у родителя есть position: relative для корректного позиционирования кнопки
        const computedStyle = window.getComputedStyle(parent);
        if (computedStyle.position === 'static') {
            parent.style.position = 'relative';
        }
        parent.appendChild(btn);

        let isErasing = false;
        let isDrawing = false;
        let lastX = 0, lastY = 0;

        // Находим именно mask canvas для рисования прозрачности
        const maskCanvas = parent.querySelector('canvas[key="mask"]') || canvas;
        const ctx = maskCanvas.getContext('2d', { willReadFrequently: true });

        // 2. Логика кнопки
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            e.preventDefault();
            isErasing = !isErasing;
            
            if (isErasing) {
                btn.style.background = '#ff4444';
                btn.style.color = '#ffffff';
                canvas.style.cursor = 'cell'; // Курсор-прицел для ластика
                console.log("[Eraser] Режим СТИРАНИЯ включен");
            } else {
                btn.style.background = '#ffffff';
                btn.style.color = '#000000';
                canvas.style.cursor = 'crosshair';
                console.log("[Eraser] Режим КИСТИ включен");
            }
        });

        // 3. Расчет координат с учетом возможного Zoom/Pan (из твоего первого скрипта!)
        function getCoordinates(e) {
            const rect = maskCanvas.getBoundingClientRect();
            // Отношение реального разрешения canvas к его отображаемому CSS-размеру
            const scaleX = maskCanvas.width / rect.width;
            const scaleY = maskCanvas.height / rect.height;
            return {
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            };
        }

        // 4. Перехват событий в фазе CAPTURE (true), чтобы сработать раньше Gradio
        canvas.addEventListener('pointerdown', (e) => {
            if (!isErasing) return;
            console.log("[Eraser] pointerdown сработал");
            e.preventDefault();
            e.stopPropagation(); // Блокируем рисование черной кистью от Gradio
            
            isDrawing = true;
            const pos = getCoordinates(e);
            lastX = pos.x;
            lastY = pos.y;

            ctx.globalCompositeOperation = 'destination-out';
            ctx.beginPath();
            ctx.arc(lastX, lastY, 20, 0, Math.PI * 2); // Радиус 20 по умолчанию
            ctx.fill();
        }, true);

        canvas.addEventListener('pointermove', (e) => {
            if (!isErasing || !isDrawing) return;
            e.preventDefault();
            e.stopPropagation();
            
            const pos = getCoordinates(e);
            ctx.globalCompositeOperation = 'destination-out';
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(pos.x, pos.y);
            ctx.lineWidth = 40; // Диаметр = радиус * 2
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.stroke();
            
            lastX = pos.x;
            lastY = pos.y;
        }, true);

        const stopErasing = (e) => {
            if (!isDrawing) return;
            console.log("[Eraser] pointerup, применяем изменения");
            isDrawing = false;
            ctx.globalCompositeOperation = 'source-over'; // Сброс режима
            
            // Принудительно сообщаем Gradio об изменении
            canvas.dispatchEvent(new Event('input', { bubbles: true }));
            canvas.dispatchEvent(new Event('change', { bubbles: true }));
            
            // Специфичный триггер для кастомных компонентов Gradio
            const customEvent = new CustomEvent('gradio:change', { bubbles: true, detail: { value: null } });
            parent.dispatchEvent(customEvent);
        };

        canvas.addEventListener('pointerup', stopErasing, true);
        canvas.addEventListener('pointerleave', stopErasing, true);
    });
}

// Запускаем сразу и следим за изменениями DOM (на случай динамической загрузки)
initEraser();
const observer = new MutationObserver(() => {
    setTimeout(initEraser, 500); // Небольшая задержка, чтобы DOM успел отрисоваться
});
observer.observe(document.body, { childList: true, subtree: true });