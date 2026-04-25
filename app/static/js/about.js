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

// Все элементы с классом scroll-animate
document.querySelectorAll('.scroll-animate').forEach(elem => {
  observer.observe(elem);
});

// ================== SUCCESS CARDS ==================
const successCards = document.querySelectorAll('.success-card');

function animateCards() {
  successCards.forEach((card, index) => {
    const cardPosition = card.getBoundingClientRect().top;
    const windowHeight = window.innerHeight;

    if (cardPosition < windowHeight - 50) {
      setTimeout(() => {
        card.classList.add('visible');
      }, index * 200);
    }
  });
}

window.addEventListener('scroll', animateCards);
window.addEventListener('load', animateCards);

// ================== HERO PARALLAX ==================
const heroImage = document.querySelector('.hero-image img');
const heroText = document.querySelector('.hero-text');

function heroParallax() {
  const scrollPos = window.scrollY;
  if (heroImage) {
    heroImage.style.transform = `translateY(${scrollPos * 0.2}px) scale(1.02)`;
  }
  if (heroText) {
    heroText.style.transform = `translateY(${scrollPos * 0.1}px)`;
  }
}

window.addEventListener('scroll', heroParallax);

// ================== LOAD ANIMATION ==================
window.addEventListener('load', () => {
  if (heroImage) heroImage.classList.add('visible');
  if (heroText) heroText.classList.add('visible');

  animateCards();
});
