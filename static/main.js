// ---------------------------------------------------------------------------
// Placeholders
// ---------------------------------------------------------------------------
// Data URIs for gray placeholder images shown before camera connects.
// Using SVG so there's no network request and no broken image icon.

const PLACEHOLDER_CAM = svgPlaceholder("No Camera", "#1a1d26");
const PLACEHOLDER_IMG = svgPlaceholder("No Image", "#13151e");

function svgPlaceholder(text, fill) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 360">
    <rect width="640" height="360" fill="${fill}"/>
    <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle"
      font-family="IBM Plex Mono, monospace" font-size="18" fill="#3a3f50">${text}</text>
  </svg>`;
  return "data:image/svg+xml;base64," + btoa(svg);
}

// ---------------------------------------------------------------------------
// On page load
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", async () => {
  // Set placeholders so the UI never shows broken image icons
  [1, 2].forEach((id) => {
    document.getElementById(`video-${id}`).src = PLACEHOLDER_CAM;
    document.getElementById(`captured-${id}`).src = PLACEHOLDER_IMG;
    document.getElementById(`processed-${id}`).src = PLACEHOLDER_IMG;
  });

  // Load available filters from the backend and populate both dropdowns.
  // This is the key connection: the dropdown reflects what process.py
  // actually has loaded, not a hardcoded list in the HTML.
  await loadFilters();
});

// ---------------------------------------------------------------------------
// Filter loader
// ---------------------------------------------------------------------------

async function loadFilters() {
  try {
    const res = await fetch("/filters");
    const data = await res.json(); // { filters: ["grayscale", "gaussian", ...] }

    [1, 2].forEach((id) => {
      const select = document.getElementById(`filter-${id}`);

      data.filters.forEach((name) => {
        const option = document.createElement("option");
        option.value = name;
        // Capitalise first letter for display: "grayscale" → "Grayscale"
        option.textContent = name.charAt(0).toUpperCase() + name.slice(1);
        select.appendChild(option);
      });
    });
  } catch (err) {
    console.warn("Could not load filters:", err);
    // App still works — dropdown just has the "none" option
  }
}

// ---------------------------------------------------------------------------
// Camera connect / disconnect
// ---------------------------------------------------------------------------

async function setSource(cam_id) {
  const src = document.getElementById(`src-${cam_id}`).value.trim();

  if (!src) {
    alert("Enter a camera source first (e.g. 0, or an RTSP/HTTP URL).");
    return;
  }

  const res = await fetch("/set_source", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cam_id, source: src }),
  });

  const j = await res.json();
  if (!j.ok) {
    alert("Connect failed: " + (j.detail || "unknown error"));
    return;
  }

  // Reload the stream img with a cache-buster so the browser picks up
  // the new source instead of serving the old cached stream.
  const vid = document.getElementById(`video-${cam_id}`);
  vid.src = `/video_feed/${cam_id}?t=${Date.now()}`;

  setStatus(cam_id, true);
}

async function stopSource(cam_id) {
  const res = await fetch("/set_source", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cam_id, source: "" }),
  });

  const j = await res.json();
  if (!j.ok) {
    alert("Stop failed: " + (j.detail || "unknown error"));
    return;
  }

  document.getElementById(`video-${cam_id}`).src = PLACEHOLDER_CAM;
  setStatus(cam_id, false);
}

// ---------------------------------------------------------------------------
// Capture + process
// ---------------------------------------------------------------------------

async function capture(cam_id) {
  // Read which filter the user selected in the dropdown
  const step = document.getElementById(`filter-${cam_id}`).value;

  // Disable the button while the request is in-flight (prevent double click)
  const btn = document.querySelector(`#col-${cam_id} .capture-btn`);
  btn.disabled = true;

  try {
    const res = await fetch("/capture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cam_id, step }), // step goes to FastAPI here
    });

    const j = await res.json();

    if (!j.ok) {
      alert("Capture failed: " + (j.detail || "unknown error"));
      return;
    }

    // Set both image panels from the base64 data URIs returned by the server
    document.getElementById(`captured-${cam_id}`).src = j.image;
    document.getElementById(`processed-${cam_id}`).src = j.processed;

    // Update process time display
    const timeEl = document.getElementById(`proc-time-${cam_id}`);
    if (timeEl && typeof j.process_time_ms === "number") {
      timeEl.textContent = `process time: ${j.process_time_ms.toFixed(2)} ms`;
    }
  } catch (err) {
    alert("Error: " + err);
  } finally {
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Preset fill
// ---------------------------------------------------------------------------

function fillPreset(cam_id, value) {
  // Fill the source input with a preset value.
  // For iPhone/RTSP, the user still needs to edit the IP — this just
  // saves them from typing the URL structure from scratch.
  document.getElementById(`src-${cam_id}`).value = value;
}

// ---------------------------------------------------------------------------
// Status dot helper
// ---------------------------------------------------------------------------

function setStatus(cam_id, connected) {
  const dot = document.getElementById(`status-${cam_id}`);
  if (connected) {
    dot.classList.add("connected");
  } else {
    dot.classList.remove("connected");
  }
}

function saveImg(cam_id, type) {
  //type = "captured" or "processed"
  const img = document.getElementById(`${type}-${cam_id}`);

  //if it is still the placeholder, then don't save
  if (!img.src.startsWith("data:image/jpeg")) {
    alert("No image to save yet - capture the image first");
    return;
  }

  // Create a temporary tag <a> and simulate a click to download
  const a = document.createElement("a");
  a.href = img.src;
  a.download = `cam${cam_id}_${type}_${Date.now()}.jpg`;
  a.click();
}
