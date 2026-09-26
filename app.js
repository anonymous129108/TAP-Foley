"use strict";
const byId = id => document.getElementById(id);
let data;
let datasetIndex = 0;
let sampleIndex = 0;
const descriptions = {
  "vggsound": "20 curated examples used in the final user study, shown here with method identities revealed.",
  "greatest-hits": "16 additional examples originally prepared for the user study, but not included in the final survey. Ours uses the current constant-α TAP-Foley outputs on the same selected video–prompt pairs."
};

function pauseAll(except) {
  document.querySelectorAll("video").forEach(video => {
    if (video !== except) video.pause();
  });
}

function disposePlayers() {
  document.querySelectorAll("video").forEach(video => {
    video.pause();
    video.removeAttribute("src");
    video.load();
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
  if (input || method.ours) {
    const badge = document.createElement("span");
    badge.className = `badge${input ? " input" : ""}`;
    badge.textContent = input ? "Silent" : "Ours";
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

function renderSample() {
  disposePlayers();
  const dataset = data.datasets[datasetIndex];
  const sample = dataset.samples[sampleIndex];
  byId("sample-select").value = String(sampleIndex);
  byId("sample-count").textContent = `${String(sampleIndex + 1).padStart(2, "0")} / ${dataset.samples.length}`;
  byId("source-prompt").textContent = sample.source;
  byId("target-prompt").textContent = sample.target;
  byId("previous").disabled = sampleIndex === 0;
  byId("next").disabled = sampleIndex === dataset.samples.length - 1;
  byId("ours-grid").replaceChildren(mediaCard(sample), ...data.methods.filter(m => m.ours).map(m => mediaCard(sample, m)));
  byId("baseline-grid").replaceChildren(...data.methods.filter(m => !m.ours).map(m => mediaCard(sample, m)));
}

function chooseDataset(index) {
  datasetIndex = index;
  sampleIndex = 0;
  const dataset = data.datasets[index];
  document.querySelectorAll("[data-dataset]").forEach(tab => {
    const selected = tab.dataset.dataset === dataset.id;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  byId("demo-panel").setAttribute("aria-labelledby", `tab-${dataset.id}`);
  byId("dataset-description").textContent = descriptions[dataset.id];
  byId("sample-select").replaceChildren(...dataset.samples.map((sample, i) => {
    const option = document.createElement("option");
    option.value = i;
    option.textContent = `${String(i + 1).padStart(2, "0")} · ${sample.source} → ${sample.target}`;
    return option;
  }));
  renderSample();
}

async function init() {
  try {
    const response = await fetch("samples.json");
    if (!response.ok) throw new Error(`Manifest: ${response.status}`);
    data = await response.json();
    document.querySelectorAll("[data-dataset]").forEach((tab, index) => {
      tab.addEventListener("click", () => chooseDataset(index));
      tab.addEventListener("keydown", event => {
        if (!["ArrowRight", "ArrowLeft", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        const next = event.key === "Home" ? 0 : event.key === "End" ? data.datasets.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + data.datasets.length) % data.datasets.length;
        chooseDataset(next);
        document.querySelectorAll("[data-dataset]")[next].focus();
      });
    });
    byId("sample-select").addEventListener("change", event => { sampleIndex = Number(event.target.value); renderSample(); });
    byId("previous").addEventListener("click", () => { if (sampleIndex > 0) { sampleIndex--; renderSample(); } });
    byId("next").addEventListener("click", () => { if (sampleIndex < data.datasets[datasetIndex].samples.length - 1) { sampleIndex++; renderSample(); } });
    byId("loop").addEventListener("change", event => document.querySelectorAll("video").forEach(v => { v.loop = event.target.checked; }));
    document.addEventListener("visibilitychange", () => { if (document.hidden) pauseAll(); });
    chooseDataset(0);
    byId("demo-app").hidden = false;
    byId("load-status").hidden = true;
  } catch (error) {
    byId("load-status").textContent = "The examples could not be loaded. Please refresh the page to try again.";
    console.error(error);
  }
}
init();
