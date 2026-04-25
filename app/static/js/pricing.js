document.querySelectorAll(".faq-item").forEach(item => {
  const btn = item.querySelector("button");
  const answer = item.querySelector(".faq-answer");

  answer.style.maxHeight = "0px";
  answer.style.transition = "max-height 0.35s ease"; 

  btn.addEventListener("click", () => {
    const open = item.classList.contains("active");

    document.querySelectorAll(".faq-item").forEach(i => {
      i.classList.remove("active");
      i.querySelector(".faq-answer").style.maxHeight = "0px";
    });

    if (!open) {
      item.classList.add("active");
      answer.style.maxHeight = answer.scrollHeight + "px";
    }
  });
});


// Анимация появления FAQ при скролле
const faqItems = document.querySelectorAll(".faq-item");

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add("visible");
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.2 });

faqItems.forEach(item => observer.observe(item));
