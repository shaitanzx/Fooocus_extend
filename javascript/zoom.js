colorPickerInput.addEventListener('input', () => {
    if (isEraserMode && lastCursorPos) {
        clearEraserCursor(interfaceCanvas, lastCursorPos);
        drawEraserCursor(interfaceCanvas, lastCursorPos, lastCursorPos.radius, null);
    }
});