// Apply saved theme on page load
window.onload = () => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        document.documentElement.classList.add(savedTheme);
    }
};

// Toggle between light and dark theme
function toggleTheme() {
    if (document.documentElement.classList.contains('dark-theme')) {
        document.documentElement.classList.remove('dark-theme');
        localStorage.setItem('theme', ''); // Reset to light theme
    } else {
        document.documentElement.classList.add('dark-theme');
        localStorage.setItem('theme', 'dark-theme');
    }
}
