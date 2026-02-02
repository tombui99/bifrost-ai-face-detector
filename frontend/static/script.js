const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const scanBtn = document.getElementById("scan-btn");
const loader = document.getElementById("loader");
const faceCountText = document.getElementById("face-count");
const resultModal = document.getElementById("result-modal");
const closeBtn = document.querySelector(".close-btn");
const resultCanvas = document.getElementById("result-canvas");
const resultList = document.getElementById("result-list");

// Configuration
// TODO: Replace with your actual Fly.io app URL
const API_URL =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1"
    ? "http://localhost:8080"
    : "https://ai-face-detector-backend.fly.dev"; // Replace this!

let isProcessing = false;

// Setup Camera
async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    video.srcObject = stream;

    return new Promise((resolve) => {
      video.onloadedmetadata = () => {
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
        resolve(video);
      };
    });
  } catch (err) {
    console.error("Error accessing camera:", err);
    alert(
      "Could not access camera. Please ensure you are using HTTPS or localhost.",
    );
  }
}

// Simple Face Detection (Visual Only for UI feeling)
// In a real browser app, we'd use face-api.js for real-time tracking,
// but for this demo, we'll focus on the backend snapshot recognition.
function drawLoop() {
  // Just clear for now, or add simple motion detection if needed
  // Real tracking will come from the snapshot result for this simplified demo
  requestAnimationFrame(drawLoop);
}

async function takeSnapshot() {
  if (isProcessing) return;
  isProcessing = true;
  loader.classList.remove("hidden");

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = video.videoWidth;
  tempCanvas.height = video.videoHeight;
  const tCtx = tempCanvas.getContext("2d");

  // Draw current frame to canvas
  tCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
  const imageData = tempCanvas.toDataURL("image/jpeg", 0.8);

  try {
    const response = await fetch(`${API_URL}/process_snapshot`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageData }),
    });

    const data = await response.json();
    if (data.status === "success") {
      showResults(data.detections, tempCanvas);
    } else {
      alert("Error: " + data.message);
    }
  } catch (err) {
    console.error("API Error:", err);
    alert("Failed to connect to backend server.");
  } finally {
    isProcessing = false;
    loader.classList.add("hidden");
  }
}

function showResults(detections, sourceCanvas) {
  resultCanvas.width = sourceCanvas.width;
  resultCanvas.height = sourceCanvas.height;
  const rCtx = resultCanvas.getContext("2d");

  // Draw the image
  rCtx.drawImage(sourceCanvas, 0, 0);

  // Clear list
  resultList.innerHTML = "";
  faceCountText.innerText = `Faces Detected: ${detections.length}`;

  detections.forEach((det) => {
    const color = det.name === "Unknown" ? "#ff3860" : "#00d1b2";

    // Draw box on result canvas
    rCtx.strokeStyle = color;
    rCtx.lineWidth = 4;
    rCtx.strokeRect(det.x, det.y, det.w, det.h);

    // Draw label
    rCtx.fillStyle = color;
    rCtx.font = "bold 24px Inter";
    rCtx.fillText(det.name, det.x, det.y - 10);

    // Add to list
    const item = document.createElement("div");
    item.className = `detection-item ${det.name === "Unknown" ? "unknown" : ""}`;
    item.innerHTML = `
            <span class="detection-name">${det.name}</span>
            <span class="detection-dist">${det.name !== "Unknown" ? "Match: " + (1 - det.distance).toFixed(2) : ""}</span>
        `;
    resultList.appendChild(item);
  });

  resultModal.classList.remove("hidden");
}

scanBtn.addEventListener("click", takeSnapshot);
closeBtn.addEventListener("click", () => resultModal.classList.add("hidden"));

// Start
setupCamera().then(() => {
  drawLoop();
});
