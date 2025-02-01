// Function to load and apply the theme
function loadAndApplyTheme() {
    // Fetch the theme data from the server
    fetch('/get_theme_data')
        .then(response => response.json())
        .then(themeData => {
            // Apply the theme to the page
            const root = document.documentElement;
            root.style.setProperty("--theme-color", themeData.color);
            root.style.setProperty("--theme-bg-color", themeData.background_color);
            root.style.setProperty("--theme-font", themeData.font);
            root.style.setProperty("--theme-text-color", themeData.text_color);
            root.style.setProperty("--button-color", themeData.button_color);
            root.style.setProperty("--button-hover-color", themeData.button_hover_color);
            root.style.setProperty("--button-text-color", themeData.button_text_color);
            document.body.style.fontSize = `${themeData.size}px`;
        })
        .catch(error => {
            console.error("Error loading theme data:", error);
        });
}

// Load and apply the theme when the page loads
window.addEventListener("load", loadAndApplyTheme);