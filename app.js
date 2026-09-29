"use strict";
const byId = id => document.getElementById(id);

function pauseAll(except) {
  document.querySelectorAll("video").forEach(video => {
    if (video !== except) video.pause();
  });
}

function mediaCard(sample, method) {
  const input = !method;
  const card = document.createElement("article");
  card.className = `media-card${method?.ours ? " ours" : ""}`;
  const label = document.createElement("div");
  label.className = "media-label";
  const text = document.createElement("div");
  const heading = document.createElement("h4");
  heading.textContent = input ? "Input video" : method.name;
  const subtitle = document.createElement("p");
  subtitle.textContent = input ? "Silent visual reference" : method.backbone;
  text.append(heading, subtitle);
  label.append(text);
  const tag = input ? "Silent" : method.ours ? "Ours" : method.tag;
  if (tag) {
    const badge = document.createElement("span");
    badge.className = `badge${input ? " input" : method.ours ? "" : ` tag-${tag.toLowerCase()}`}`;
    badge.textContent = tag;
    label.append(badge);
  }
  const video = document.createElement("video");
  video.controls = true;
  video.playsInline = true;
  video.preload = "none";
  video.muted = input;
  video.loop = byId("loop").checked;
  video.poster = sample.poster;
  video.src = input ? sample.video : sample.outputs[method.id];
  video.setAttribute("aria-label", `${input ? "Silent input" : `${method.name}, ${method.backbone}${method.ours ? ", Ours" : ", baseline"}`}: ${sample.source} to ${sample.target}`);
  video.addEventListener("play", () => {
    pauseAll(video);
    card.classList.add("playing");
  });
  ["pause", "ended"].forEach(event => video.addEventListener(event, () => card.classList.remove("playing")));
  video.addEventListener("error", () => {
    if (!video.getAttribute("src") || !card.isConnected || card.querySelector(".media-error")) return;
    const error = document.createElement("p");
    error.className = "media-error";
    error.textContent = "This video could not load. ";
    const link = document.createElement("a");
    link.href = video.src;
    link.textContent = "Open video directly ↗";
    error.append(link);
    card.append(error);
  });
  card.append(label, video);
  return card;
}

function promptLine(kind, text) {
  const line = document.createElement("p");
  line.className = `${kind.toLowerCase()}-prompt`;
  const label = document.createElement("span");
  label.textContent = kind;
  const value = document.createElement("strong");
  value.textContent = text;
  line.append(label, value);
  return line;
}

function promptCell(sample, number) {
  const cell = document.createElement("div");
  cell.className = "prompt-cell";
  const index = document.createElement("span");
  index.className = "sample-number";
  index.textContent = String(number).padStart(2, "0");
  const arrow = document.createElement("span");
  arrow.className = "prompt-arrow";
  arrow.setAttribute("aria-hidden", "true");
  arrow.textContent = "→";
  cell.append(index, promptLine("Source", sample.source), arrow, promptLine("Target", sample.target));
  return cell;
}

function renderComparison(data) {
  const dataset = data.datasets[0];
  byId("dataset-name").textContent = dataset.name;
  byId("sample-count").textContent = `${dataset.samples.length} examples`;
  byId("comparison").style.setProperty("--columns", data.methods.length + 1);
  byId("comparison").replaceChildren(...dataset.samples.map((sample, i) => {
    const row = document.createElement("div");
    row.className = "sample-row";
    row.setAttribute("role", "group");
    row.setAttribute("aria-label", `Example ${i + 1}: ${sample.source} to ${sample.target}`);
    row.append(promptCell(sample, i + 1), mediaCard(sample), ...data.methods.map(m => mediaCard(sample, m)));
    return row;
  }));
}

function setupColumnScroll() {
  const frame = byId("comparison-frame");
  const scroller = byId("comparison-scroll");
  const update = () => {
    const max = scroller.scrollWidth - scroller.clientWidth;
    frame.classList.toggle("at-start", scroller.scrollLeft <= 1);
    frame.classList.toggle("at-end", scroller.scrollLeft >= max - 1);
    byId("scroll-prev").disabled = scroller.scrollLeft <= 1;
    byId("scroll-next").disabled = scroller.scrollLeft >= max - 1;
  };
  const step = direction => {
    const card = scroller.querySelector(".media-card");
    const column = card.offsetWidth + parseFloat(getComputedStyle(card.parentElement).columnGap);
    const columns = Math.max(1, Math.floor(scroller.clientWidth / column));
    scroller.scrollBy({ left: direction * columns * column, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  };
  byId("scroll-prev").addEventListener("click", () => step(-1));
  byId("scroll-next").addEventListener("click", () => step(1));
  scroller.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update);
  update();
}

async function init() {
  try {
    const response = await fetch("samples.json");
    if (!response.ok) throw new Error(`Manifest: ${response.status}`);
    renderComparison(await response.json());
    byId("loop").addEventListener("change", event => document.querySelectorAll("video").forEach(v => { v.loop = event.target.checked; }));
    document.addEventListener("visibilitychange", () => { if (document.hidden) pauseAll(); });
    byId("demo-app").hidden = false;
    byId("load-status").hidden = true;
    setupColumnScroll();
  } catch (error) {
    byId("load-status").textContent = "The examples could not be loaded. Please refresh the page to try again.";
    console.error(error);
  }
}
init();
