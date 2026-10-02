// scroll reveal
const io = new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting) { e.target.classList.add("show");
        io.unobserve(e.target); }
}), { threshold: 0.15 });
document.querySelectorAll(".reveal").forEach(el => io.observe(el));

// number counter
document.querySelectorAll("[data-count]").forEach(el => {
    const end = parseFloat(el.dataset.count),
        suffix = el.dataset.suffix || "";
    let n = 0;
    const step = Math.max(0.1, end / 60);
    const t = setInterval(() => {
        n += step;
        if (n >= end) { n = end;
            clearInterval(t); }
        el.textContent = (Number.isInteger(end) ? Math.round(n) : n.toFixed(1)) + suffix;
    }, 25);
});

// loading spinner on predict
const f = document.querySelector("form.predict-form");
if (f) f.addEventListener("submit", () => {
    f.querySelector("button").innerHTML = '<span class="spinner"></span> Analysing...';
});