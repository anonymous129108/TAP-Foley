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

function scrollButton(label, text) {
  const button = document.createElement("button");
  button.type = "button";
  button.setAttribute("aria-label", label);
  button.textContent = text;
  return button;
}

function sampleRow(sample, number, methods) {
  const row = document.createElement("div");
  row.className = "sample-row";
  row.setAttribute("role", "listitem");
  row.setAttribute("aria-label", `Example ${number}: ${sample.source} to ${sample.target}`);
  const head = document.createElement("div");
  head.className = "sample-head";
  const buttons = document.createElement("div");
  buttons.className = "scroll-buttons";
  const prev = scrollButton(`Show earlier methods for example ${number}`, "←");
  const next = scrollButton(`Show more methods for example ${number}`, "→");
  buttons.append(prev, next);
  head.append(promptCell(sample, number), buttons);
  const frame = document.createElement("div");
  frame.className = "row-frame at-start";
  const scroller = document.createElement("div");
  scroller.className = "row-scroll";
  scroller.tabIndex = 0;
  scroller.setAttribute("aria-label", `Method outputs for example ${number}; scroll sideways for more`);
  const track = document.createElement("div");
  track.className = "row-track";
  track.append(mediaCard(sample), ...methods.map(m => mediaCard(sample, m)));
  scroller.append(track);
  frame.append(scroller);
  row.append(head, frame);

  const update = () => {
    const max = scroller.scrollWidth - scroller.clientWidth;
    frame.classList.toggle("at-start", scroller.scrollLeft <= 1);
    frame.classList.toggle("at-end", scroller.scrollLeft >= max - 1);
    prev.disabled = scroller.scrollLeft <= 1;
    next.disabled = scroller.scrollLeft >= max - 1;
  };
  const step = direction => {
    const card = track.firstElementChild;
    const column = card.offsetWidth + parseFloat(getComputedStyle(track).columnGap);
    const columns = Math.max(1, Math.floor(scroller.clientWidth / column));
    scroller.scrollBy({ left: direction * columns * column, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  };
  prev.addEventListener("click", () => step(-1));
  next.addEventListener("click", () => step(1));
  scroller.addEventListener("scroll", update, { passive: true });
  row.update = update;
  return row;
}

function renderComparison(data) {
  const dataset = data.datasets[0];
  byId("dataset-name").textContent = dataset.name;
  byId("sample-count").textContent = `${dataset.samples.length} examples`;
  byId("comparison").replaceChildren(...dataset.samples.map((sample, i) => sampleRow(sample, i + 1, data.methods)));
}

function updateRows() {
  document.querySelectorAll(".sample-row").forEach(row => row.update());
}

async function init() {
  try {
    const response = await fetch("samples.json?v=20260930b");
    if (!response.ok) throw new Error(`Manifest: ${response.status}`);
    renderComparison(await response.json());
    byId("loop").addEventListener("change", event => document.querySelectorAll("video").forEach(v => { v.loop = event.target.checked; }));
    document.addEventListener("visibilitychange", () => { if (document.hidden) pauseAll(); });
    byId("demo-app").hidden = false;
    byId("load-status").hidden = true;
    updateRows();
    window.addEventListener("resize", updateRows);
  } catch (error) {
    byId("load-status").textContent = "The examples could not be loaded. Please refresh the page to try again.";
    console.error(error);
  }
}
init();
