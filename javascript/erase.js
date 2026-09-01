console.log("🔧 [Eraser Script v2] Загрузка...");

function findCanvasDeep(root = document) {
    const canvases = [];
    
    // 1. Ищем в обычном DOM
    root.querySelectorAll('canvas').forEach(c => canvases.push(c));
    
    // 2. Ищем в Shadow DOM (рекурсивно)
    root.querySelectorAll('*').forEach(el => {
        if (el.shadowRoot) {
            canvases.push(...findCanvasDeep(el.shadowRoot));
        }
    });
    
    return canvases;
}

function analyzeComponent() {
    console.log("\n=== 🔍 ДИАГНОСТИКА КОМПОНЕНТА ===");
    
    // Ищем все возможные контейнеры
    const possibleContainers = [
        ...document.querySelectorAll('.gradio-image'),
        ...document.querySelectorAll('[class*="sketch"]'),
        ...document.querySelectorAll('[class*="canvas"]'),
        ...document.querySelectorAll('[class*="image"]')
    ];
    
    console.log(`Найдено возможных контейнеров: ${possibleContainers.length}`);
    
    possibleContainers.forEach((container, idx) => {
        console.log(`\n--- Контейнер #${idx} ---`);
        console.log('Классы:', container.className);
        console.log('ID:', container.id);
        console.log('Тег:', container.tagName);
        
        // Проверяем Shadow DOM
        if (container.shadowRoot) {
            console.log('✅ Есть Shadow DOM!');
            const shadowCanvases = findCanvasDeep(container.shadowRoot);
            console.log(`Canvas в Shadow DOM: ${shadowCanvases.length}`);
            shadowCanvases.forEach((c, i) => {
                console.log(`  Canvas #${i}:`, c);
                console.log(`    Размер: ${c.width}x${c.height}`);
                console.log(`    Атрибуты:`, Array.from(c.attributes).map(a => `${a.name}="${a.value}"`).join(', '));
            });
        }
        
        // Ищем canvas в обычном DOM
        const normalCanvases = container.querySelectorAll('canvas');
        console.log(`Canvas в обычном DOM: ${normalCanvases.length}`);
        normalCanvases.forEach((c, i) => {
            console.log(`  Canvas #${i}:`, c);
            console.log(`    Размер: ${c.width}x${c.height}`);
            console.log(`    Атрибуты:`, Array.from(c.attributes).map(a => `${a.name}="${a.value}"`).join(', '));
        });
        
        // Проверяем другие элементы рисования
        const svgs = container.querySelectorAll('svg');
        const imgs = container.querySelectorAll('img');
        const divsWithBg = container.querySelectorAll('div[style*="background"]');
        
        console.log(`SVG элементов: ${svgs.length}`);
        console.log(`IMG элементов: ${imgs.length}`);
        console.log(`DIV с background: ${divsWithBg.length}`);
        
        // Выводим структуру (первые 3 уровня)
        console.log('Структура (первые 3 уровня):');
        function printTree(el, depth = 0) {
            if (depth > 3) return;
            const indent = '  '.repeat(depth);
            const attrs = Array.from(el.attributes).map(a => `${a.name}="${a.value}"`).join(' ');
            console.log(`${indent}<${el.tagName.toLowerCase()} ${attrs}>`);
            Array.from(el.children).forEach(child => printTree(child, depth + 1));
        }
        printTree(container);
    });
    
    console.log("\n=== КОНЕЦ ДИАГНОСТИКИ ===\n");
}

// Запускаем диагностику сразу и через промежутки времени
analyzeComponent();
setTimeout(analyzeComponent, 1000);
setTimeout(analyzeComponent, 3000);

// Наблюдаем за изменениями DOM
const observer = new MutationObserver(() => {
    console.log("[Observer] DOM изменился, запускаем диагностику...");
    analyzeComponent();
});

observer.observe(document.body, { 
    childList: true, 
    subtree: true,
    attributes: true
});