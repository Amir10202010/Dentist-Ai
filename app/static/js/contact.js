// ================== SCROLL ANIMATIONS ==================
const observerOptions = {
  threshold: 0.15
};

const observer = new IntersectionObserver((entries, observer) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      observer.unobserve(entry.target);
    }
  });
}, observerOptions);

document.querySelectorAll('.scroll-animate').forEach(elem => {
  observer.observe(elem);
});

// ================== HERO PARALLAX ==================
const heroText = document.querySelector('.contact-hero .hero-content');

function heroParallax() {
  const scrollPos = window.scrollY;
  if (heroText) {
    heroText.style.transform = `translateY(${scrollPos * 0.2}px)`;
  }
}

window.addEventListener('scroll', heroParallax);

// ================== FORM INPUT ANIMATION ==================
const inputs = document.querySelectorAll('.contact-form input, .contact-form textarea');

inputs.forEach(input => {
  input.addEventListener('focus', () => {
    input.style.borderColor = 'var(--primary)';
    input.style.boxShadow = '0 4px 12px rgba(28,167,236,0.3)';
  });
  input.addEventListener('blur', () => {
    input.style.borderColor = '#ccc';
    input.style.boxShadow = 'none';
  });
});

// ================== MAP FADE-IN ==================
const map = document.querySelector('.contact-map');
function mapFadeIn() {
  if (!map) return;
  const mapTop = map.getBoundingClientRect().top;
  const windowHeight = window.innerHeight;
  if (mapTop < windowHeight - 50) {
    map.classList.add('visible');
  }
}

window.addEventListener('scroll', mapFadeIn);
window.addEventListener('load', mapFadeIn);

// ================== PAGE LOAD ANIMATION ==================
window.addEventListener('load', () => {
  if (heroText) heroText.classList.add('visible');

  document.querySelectorAll('.contact-info, .contact-form').forEach(elem => {
    setTimeout(() => {
      elem.classList.add('visible');
    }, 200);
  });

  mapFadeIn();
});


// ================== FLASH TOASTS ==================
(function() {
  const flashContainer = document.querySelector('.flash-container');
  if (!flashContainer) return;

  let toastRoot = document.querySelector('.toast-container');
  if (!toastRoot) {
    toastRoot = document.createElement('div');
    toastRoot.className = 'toast-container';
    document.body.appendChild(toastRoot);
  }

  const flashes = Array.from(flashContainer.querySelectorAll('.flash'));
  flashes.forEach((f, i) => {
    const toast = document.createElement('div');
    const category = Array.from(f.classList).find(c => c !== 'flash') || 'info';
    toast.className = `toast ${category}`;
    toast.innerHTML = `<div class="toast-body">${f.innerHTML}</div>`;

    const closeBtn = document.createElement('button');
    closeBtn.className = 'close-btn';
    closeBtn.innerHTML = '×';
    closeBtn.addEventListener('click', () => {
      hideToast(toast);
    });
    toast.appendChild(closeBtn);

    toastRoot.appendChild(toast);

    setTimeout(() => toast.classList.add('show'), 20);

    const displayTime = 3500 + i * 350;
    setTimeout(() => hideToast(toast), displayTime);
  });

  flashContainer.remove();

  function hideToast(toastEl) {
    if (!toastEl) return;
    toastEl.classList.add('fade-out');
    setTimeout(() => {
      try { toastEl.remove(); } catch(e) {}
    }, 420);
  }
})();

// ================== AJAX FORM SUBMIT ==================
const form = document.getElementById('contactForm');
if (form) {
    form.addEventListener('submit', async (e) => {
        e.preventDefault(); // предотвратить перезагрузку

        const formData = new FormData(form);

        try {
            const response = await fetch('/send_message', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            showToast(result.message, result.status); // показываем тост

            if (result.status === "success") {
                form.reset(); // очистить форму
            }

        } catch (err) {
            showToast("Ошибка отправки сообщения.", "error");
        }
    });
}

// ================== TOAST FUNCTION ==================
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
