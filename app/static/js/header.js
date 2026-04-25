// static/js/header.js

// ======== BURGER MENU ========
const headerBurger = document.getElementById("header-burger");
const headerMobileMenu = document.getElementById("header-mobileMenu");

const burgerSvg = '/static/images/burger.svg';
const closeSvg = '/static/images/close.svg';

headerBurger.innerHTML = `<img src="${burgerSvg}" alt="menu" style="width:28px; height:28px;">`;

headerBurger.addEventListener("click", () => {
  headerMobileMenu.classList.toggle("active");

  if (headerMobileMenu.classList.contains("active")) {
    headerBurger.innerHTML = `<img src="${closeSvg}" alt="close" style="width:28px; height:28px;">`;
  } else {
    headerBurger.innerHTML = `<img src="${burgerSvg}" alt="menu" style="width:28px; height:28px;">`;
  }
});

document.querySelectorAll(".header-mobile-menu a, .header-mobile-menu .header-btn").forEach(item => {
  item.addEventListener("click", () => {
    headerMobileMenu.classList.remove("active");
    headerBurger.innerHTML = `<img src="${burgerSvg}" alt="menu" style="width:28px; height:28px;">`;
  });
});

// ======== DESKTOP LANGUAGE DROPDOWN ========
const headerLangBtn = document.getElementById("header-langBtn");
const headerLangDropdown = document.getElementById("header-langDropdown");

headerLangBtn.addEventListener("click", () => {
  headerLangDropdown.classList.toggle("active");
});

document.addEventListener("click", (e) => {
  if (!headerLangBtn.contains(e.target) && !headerLangDropdown.contains(e.target)) {
    headerLangDropdown.classList.remove("active");
  }
});

headerLangDropdown.querySelectorAll("div").forEach(option => {
  option.addEventListener("click", () => {
    headerLangDropdown.classList.remove("active");
  });
});

// ======== MOBILE LANGUAGE DROPDOWN ========
const headerMobileLangBtn = document.getElementById("header-mobileLangBtn");
const headerMobileLangDropdown = document.getElementById("header-mobileLangDropdown");

headerMobileLangBtn.addEventListener("click", () => {
  headerMobileLangDropdown.classList.toggle("active");
});

headerMobileLangDropdown.querySelectorAll("div").forEach(option => {
  option.addEventListener("click", () => {
    headerMobileLangDropdown.classList.remove("active");
  });
});

// ======== NAVIGATION INDICATOR ========
const headerNavLinks = document.querySelectorAll(".header-nav-links a");
const headerIndicator = document.querySelector(".header-nav-indicator");

function moveIndicator(link) {
  headerIndicator.style.width = link.offsetWidth + "px";
  headerIndicator.style.left = link.offsetLeft + "px";
}

const headerActiveLink = document.querySelector(".header-nav-links a.active");
if (headerActiveLink) moveIndicator(headerActiveLink);

headerNavLinks.forEach(link => {
  link.addEventListener("mouseenter", () => moveIndicator(link));
  link.addEventListener("mouseleave", () => {
    if (headerActiveLink) moveIndicator(headerActiveLink);
    else headerIndicator.style.width = "0";
  });
});
