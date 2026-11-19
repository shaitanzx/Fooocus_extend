(() => {
    if (window.fooocusImageCompareListenerInstalled) {
        return;
    }

    window.fooocusImageCompareListenerInstalled = true;

    document.addEventListener("input", (event) => {
        const input = event.target;

        if (
            !input ||
            !input.classList ||
            !input.classList.contains("fooocus-image-compare__range")
        ) {
            return;
        }

        const slider = input.closest(".fooocus-image-compare");

        if (!slider) {
            return;
        }

        slider.style.setProperty(
            "--fooocus-compare-position",
            `${input.value}%`
        );
    });
})();