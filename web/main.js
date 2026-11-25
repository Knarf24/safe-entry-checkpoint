// The page only handles the camera and drawing. All decisions are made in Python (door.py, checkpoint.py).
const $ = (id) => document.getElementById(id);
const PY_FILES = ["checkpoint.py", "door.py"];
const MODEL_FILES = [
  "models/face_detection_yunet_2023mar.onnx",
  "models/image_classification_mobilenetv2_2022apr.onnx",
  "models/mask_head.npz",
];

let py = null, door = null, stream = null, mode = "cam", uploaded = null;

function setLoading(text) { $("loadText").textContent = text; }

async function startPython() {
  try {
    setLoading("Starting Python…");
    py = await loadPyodide({ indexURL: new URL("pyodide/", location.href).href });
    setLoading("Loading OpenCV and NumPy…");
    await py.loadPackage(["numpy", "opencv-python"]);
    setLoading("Loading the face and mask models…");
    py.FS.mkdirTree("/home/pyodide/models");
    await Promise.all([...PY_FILES, ...MODEL_FILES].map(async (f) => {
      const r = await fetch(f);
      if (!r.ok) throw new Error(`Could not load ${f}`);
      py.FS.writeFile(`/home/pyodide/${f}`, new Uint8Array(await r.arrayBuffer()));
    }));
    setLoading("Getting the checkpoint ready…");
    door = py.pyimport("door");
    $("loading").classList.add("done");
    updateReady();
  } catch (e) {
    console.error(e);
    $("loading").classList.add("error");
    setLoading("Sorry, Python could not start in this browser. Try Chrome, Edge or Firefox. (" + e.message + ")");
    document.querySelector(".spinner").style.display = "none";
  }
}

function updateReady() {
  const haveImage = mode === "cam" ? !!stream : !!uploaded;
  $("checkBtn").disabled = !(door && haveImage);
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    $("video").srcObject = stream;
    await $("video").play();
    $("camHint").hidden = true;
    $("startCam").textContent = "Camera on";
    $("startCam").disabled = true;
  } catch (e) {
    $("camHint").innerHTML = "Couldn't open the camera. Allow camera access, or upload a photo instead.";
  }
  updateReady();
}

function setMode(m) {
  mode = m;
  $("tabCam").classList.toggle("active", m === "cam");
  $("tabUpload").classList.toggle("active", m === "upload");
  $("video").hidden = m !== "cam";
  $("preview").hidden = m !== "upload" || !uploaded;
  $("startCam").hidden = m !== "cam";
  $("camHint").hidden = m === "cam" ? !!stream : !!uploaded;
  if (m === "upload") {
    $("camHint").innerHTML = "Choose a photo of the people at the door.";
    if (!uploaded) $("file").click();
  } else if (!stream) {
    $("camHint").innerHTML = "Press <b>Start camera</b>, or upload a photo.";
  }
  updateReady();
}

function grabFrame() {
  const src = mode === "cam" ? $("video") : $("preview");
  const w0 = mode === "cam" ? src.videoWidth : src.naturalWidth;
  const h0 = mode === "cam" ? src.videoHeight : src.naturalHeight;
  const scale = Math.min(1, 1280 / Math.max(w0, h0));
  const w = Math.round(w0 * scale), h = Math.round(h0 * scale);
  const c = $("grab");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(src, 0, 0, w, h);
  return { data: ctx.getImageData(0, 0, w, h).data, w, h };
}

function showOccupancy(inside) {
  const cap = Math.max(1, parseInt($("capacity").value) || 1);
  $("inside").textContent = inside;
  $("cap").textContent = cap;
  const pct = Math.min(100, Math.round((100 * inside) / cap));
  $("barFill").style.width = pct + "%";
  $("barFill").style.background = pct < 80 ? "var(--ok)" : pct < 100 ? "var(--warn)" : "var(--bad)";
}

function showLog(rows) {
  if (!rows.length) {
    $("log").innerHTML = '<tr><td colspan="5" class="small">No one checked yet.</td></tr>';
    return;
  }
  $("log").innerHTML = rows.map((r) =>
    `<tr><td>${r.time}</td><td>${r.people}</td><td>${r.masks}</td>` +
    `<td class="${r.result === "Allowed" ? "ok" : "bad"}">${r.result}</td><td>${r.inside}</td></tr>`).join("");
}

async function checkEntry() {
  $("checkBtn").disabled = true;
  $("checkBtn").textContent = "Checking…";
  await new Promise((r) => setTimeout(r, 30)); // let the button repaint
  try {
    const { data, w, h } = grabFrame();
    const res = door.check(data, w, h, parseInt($("capacity").value) || 1, $("maskReq").checked, $("voice").checked);
    const [pixels, json] = res.toJs();
    res.destroy();
    const result = JSON.parse(json);
    const out = $("out");
    out.width = w; out.height = h;
    out.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(pixels.buffer, pixels.byteOffset, w * h * 4), w, h), 0, 0);
    out.hidden = false;
    const b = $("banner");
    b.className = "banner " + (result.allowed ? "ok" : "bad");
    b.textContent = (result.allowed ? "✅ " : "⛔ ") + result.headline;
    $("speech").textContent = "🔊 “" + result.speech + "”";
    showOccupancy(result.inside);
    showLog(result.log);
    window.lastResult = result; // handy for testing
  } catch (e) {
    console.error(e);
    $("banner").className = "banner bad";
    $("banner").textContent = "Something went wrong: " + e.message;
  }
  $("checkBtn").textContent = "Check entry";
  updateReady();
}

$("startCam").onclick = startCamera;
$("tabCam").onclick = () => setMode("cam");
$("tabUpload").onclick = () => setMode("upload");
$("file").onchange = () => {
  const f = $("file").files[0];
  if (!f) return;
  const img = $("preview");
  img.onload = () => { uploaded = true; setMode("upload"); };
  img.src = URL.createObjectURL(f);
};
$("checkBtn").onclick = checkEntry;
$("preview").onclick = () => $("file").click();
$("leftBtn").onclick = () => { if (door) showOccupancy(door.someone_left()); };
$("resetBtn").onclick = () => {
  if (!door) return;
  showOccupancy(door.reset());
  showLog([]);
  $("banner").className = "banner idle";
  $("banner").textContent = "📷 Ready";
  $("out").hidden = true;
  $("speech").textContent = "";
};
$("capacity").oninput = () => showOccupancy(parseInt($("inside").textContent) || 0);

startPython();
