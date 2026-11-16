const preview = document.querySelector("#preview");
const startButton = document.querySelector("#start-camera");
const stopButton = document.querySelector("#stop-camera");
const cameraStatus = document.querySelector("#camera-status");
const expressionLabel = document.querySelector("#expression");
const bunnyArt = document.querySelector("#bunny-art");
const predictionPanel = document.querySelector("#prediction-panel");
const processingTime = document.querySelector("#processing-time");

const bunnyByExpression = {
  angry:    { file: "anger.png",     background: "#fdf1d7" },
  disgust:  { file: "disgust.png",   background: "#fdf2d8" },
  fear:     { file: "fear.png",      background: "#fdefd5" },
  happy:    { file: "happiness.png", background: "#fdf2da" },
  neutral:  { file: "neutral.png",   background: "#fdf1d7" },
  sad:      { file: "sadness.png",   background: "#fdf0d4" },
  surprise: { file: "surprise.png",  background: "#fdf1d6" },
};

const snapshot = document.createElement("canvas");
const snapshotContext = snapshot.getContext("2d");

async function requestPrediction(signal) {
  snapshot.width = preview.videoWidth;
  snapshot.height = preview.videoHeight;

  const startedAt = performance.now();
  snapshotContext.drawImage(preview, 0, 0);

  const jpeg = await new Promise((resolve, reject) => {
    snapshot.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Could not encode the snapshot."));
    }, "image/jpeg", 0.85);
  });

  const response = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "image/jpeg" },
    body: jpeg,
    signal,
  });

  if (!response.ok) {
    throw new Error(`Prediction request failed (${response.status}).`);
  }

  const { expression } = await response.json();
  return { expression, startedAt };
}

function clearBunny() {
  bunnyArt.replaceChildren();
  predictionPanel.style.removeProperty("--bunny-background");
}

async function showBunny(expression, signal) {
  const bunny = bunnyByExpression[expression];

  if (!bunny) {
    clearBunny();
    return;
  }

  const src = `/bunny/${bunny.file}`;
  if (bunnyArt.firstElementChild?.getAttribute("src") === src) return;

  const image = new Image();
  image.alt = `${expression} bunny illustration`;
  image.src = src;
  await image.decode();

  if (signal.aborted) return;

  bunnyArt.replaceChildren(image);
  predictionPanel.style.setProperty("--bunny-background", bunny.background);
}

async function showPrediction({ expression, startedAt }, signal) {
  await showBunny(expression, signal);
  if (signal.aborted) return;

  expressionLabel.textContent = expression ?? "No face detected";

  return new Promise((resolve) => {
    requestAnimationFrame(() => {
      if (!signal.aborted) {
        const elapsed = performance.now() - startedAt;
        processingTime.textContent = `${elapsed.toFixed(1)} ms`;
      }
      resolve();
    });
  });
}

let predictionController;
const snapshotPauseMs = 250;

async function runPredictions(signal) {
  while (!signal.aborted) {
    try {
      const prediction = await requestPrediction(signal);
      if (signal.aborted) return;

      await showPrediction(prediction, signal);
    } catch (error) {
      if (signal.aborted) return;

      expressionLabel.textContent = error.message;
      clearBunny();
      processingTime.textContent = "—";
    }

    if (signal.aborted) return;
    await new Promise((resolve) => setTimeout(resolve, snapshotPauseMs));
  }
}

function stopCamera() {
  predictionController?.abort();
  predictionController = undefined;
  expressionLabel.textContent = "Waiting for camera";
  clearBunny();
  processingTime.textContent = "—";

  const stream = preview.srcObject;
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    preview.srcObject = null;
  }

  startButton.disabled = false;
  stopButton.disabled = true;
  cameraStatus.textContent = "Camera is off";
}

async function startCamera() {
  startButton.disabled = true;
  cameraStatus.textContent = "Starting camera…";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    preview.srcObject = stream;
    await preview.play();

    stopButton.disabled = false;
    cameraStatus.textContent = "Camera is on";
    expressionLabel.textContent = "Waiting for prediction…";
    predictionController = new AbortController();
    void runPredictions(predictionController.signal);
  } catch (error) {
    stopCamera();
    cameraStatus.textContent = `Could not start camera: ${error.message}`;
  }
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
