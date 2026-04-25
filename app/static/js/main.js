// static/js/scroll-animate.js

// ======== ANIMATION ========
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
