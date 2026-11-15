const preview = document.querySelector("#preview");
const startButton = document.querySelector("#start-camera");
const stopButton = document.querySelector("#stop-camera");
const cameraStatus = document.querySelector("#camera-status");

function stopCamera() {
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
  } catch (error) {
    stopCamera();
    cameraStatus.textContent = `Could not start camera: ${error.message}`;
  }
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
