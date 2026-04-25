// ================== AJAX FORM SUBMIT ==================
const form = document.getElementById('registerForm');
if (form) {
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(form);

        try {
            const response = await fetch('/register', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            showToast(result.message, result.status === "danger" ? "error" : result.status);

            if (result.status === "success") {
                loginForm.reset();
                // Можно поставить редирект чуть позже
                setTimeout(() => {
                    window.location.href = "/login";
                }, 1000);
            }

        } catch (err) {
            showToast("Ошибка отправки формы.", "error");
            console.error(err);
        }
    });
}

// ================== TOAST FUNCTIONS ==================
function showToast(text, type = "info") {
    let toastRoot = document.querySelector('.toast-container');
    if (!toastRoot) {
        toastRoot = document.createElement('div');
        toastRoot.className = 'toast-container';
        document.body.appendChild(toastRoot);
    }

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<div class="toast-body">${text}</div>`;

    const closeBtn = document.createElement('button');
    closeBtn.className = 'close-btn';
    closeBtn.innerHTML = '×';
    closeBtn.addEventListener('click', () => hideToast(toast));
    toast.appendChild(closeBtn);

    toastRoot.appendChild(toast);

    setTimeout(() => toast.classList.add('show'), 20);
    setTimeout(() => hideToast(toast), 3500);
}

function hideToast(toast) {
    toast.classList.add('fade-out');
    setTimeout(() => toast.remove(), 400);
}

// ================== FORM INPUT ANIMATION ==================
const inputs = document.querySelectorAll('.register-form input');
inputs.forEach(input => {
    input.addEventListener('focus', () => {
        input.style.borderColor = '#1ca7ec';
        input.style.boxShadow = '0 4px 12px rgba(28,167,236,0.3)';
    });
    input.addEventListener('blur', () => {
        input.style.borderColor = '#ccc';
        input.style.boxShadow = 'none';
    });
});
