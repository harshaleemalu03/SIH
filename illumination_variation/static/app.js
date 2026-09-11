const $ = (s) => document.querySelector(s);

const sourceInput = $("#source-input");
const referenceInput = $("#reference-input");
const sourceDrop = $("#source-drop");
const referenceDrop = $("#reference-drop");
const sourcePreview = $("#source-preview");
const referencePreview = $("#reference-preview");
const sourceName = $("#source-name");
const referenceName = $("#reference-name");
const btn = $("#register-btn");
const hint = $("#hint");
const processing = $("#processing");
const output = $("#output");
const outMatch = $("#out-match");
const outRegistered = $("#out-registered");
const mMatches = $("#m-matches");
const mInliers = $("#m-inliers");
const mRatio = $("#m-ratio");
const mMse = $("#m-mse");
const errorBox = $("#error");

let sourceFile = null;
let referenceFile = null;

function bindUpload(input, drop, preview, nameEl, setFile) {
  drop.addEventListener("click", () => input.click());
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("dragover"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("dragover"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("dragover");
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f, input, drop, preview, nameEl, setFile);
  });
  input.addEventListener("change", () => {
    const f = input.files[0];
    if (f) handleFile(f, input, drop, preview, nameEl, setFile);
  });
}

function handleFile(file, input, drop, preview, nameEl, setFile) {
  if (!file.type.startsWith("image/")) return;
  setFile(file);
  const url = URL.createObjectURL(file);
  preview.src = url;
  nameEl.textContent = file.name;
  drop.classList.add("has-image");
  updateButton();
  hideOutput();
}

function updateButton() {
  const ready = sourceFile && referenceFile;
  btn.disabled = !ready;
  hint.textContent = ready ? "Ready" : "Select both images to continue";
}

function hideOutput() {
  output.classList.add("hidden");
  errorBox.classList.add("hidden");
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove("hidden");
}

bindUpload(sourceInput, sourceDrop, sourcePreview, sourceName, (f) => { sourceFile = f; });
bindUpload(referenceInput, referenceDrop, referencePreview, referenceName, (f) => { referenceFile = f; });

btn.addEventListener("click", async () => {
  if (!sourceFile || !referenceFile) return;

  btn.disabled = true;
  hint.textContent = "";
  processing.classList.remove("hidden");
  output.classList.add("hidden");
  errorBox.classList.add("hidden");

  const fd = new FormData();
  fd.append("source", sourceFile);
  fd.append("reference", referenceFile);

  try {
    const res = await fetch("/api/register", { method: "POST", body: fd });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      throw new Error(j.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();
    outMatch.src = data.match_image;
    outRegistered.src = data.registered_image;
    mMatches.textContent = data.num_matches;
    mInliers.textContent = data.num_inliers;
    mRatio.textContent = data.inlier_ratio + "%";
    mMse.textContent = data.mse != null ? data.mse.toFixed(2) : "—";
    output.classList.remove("hidden");
    // scroll to output
    output.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    showError(e.message || String(e));
  } finally {
    processing.classList.add("hidden");
    btn.disabled = false;
  }
});
