/**
 * PoseDetector - TensorFlow.js MoveNet for Push-Up Detection
 * v3.1 Fixed: More reliable ready detection & better tracking
 */

class PoseDetector {
  constructor() {
    this.detector = null;
    this.video = null;
    this.canvas = null;
    this.ctx = null;
    this.isRunning = false;
    this.animationId = null;
    this.stream = null;

    // Push-up tracking state
    this.repCount = 0;
    this.phase = "up"; // "up", "down", or "unknown"
    this.lastPhase = "up";
    this.phaseHistory = [];

    // Height tracking
    this.heightHistory = [];
    this.heightWindow = 15;

    // Ready detection
    this.isReady = false;
    this.readyFrames = 0;
    this.readyThreshold = 12; // ~0.4s at 30 FPS (lowered for faster response)
    this.readyLostThreshold = 8; // Frames before we say not ready

    // Smoothing (Exponential Moving Average)
    this.smoothedKeypoints = null;
    this.alpha = 0.5;

    // Confidence thresholds
    this.minConfidence = 0.30; // Slightly lowered for better detection in various lighting
    this.pushupDownAngle = 100;
    this.pushupUpAngle = 155;

    // Status tracking
    this.status = "idle";
    this.errorMessage = "";
  }

  async init(videoElement, canvasElement) {
    this.video = videoElement;
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext("2d");
    this.status = "loading";

    try {
      await this.loadTFJS();
      await this.createDetector();
      this.status = "ready";
      return true;
    } catch (err) {
      this.status = "error";
      this.errorMessage = err.message || "Failed to initialize PoseDetector";
      console.error("PoseDetector init failed:", err);
      throw err;
    }
  }

  async loadTFJS() {
    const scripts = [
      {
        src: "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core@4.10.0/dist/tf-core.min.js",
        name: "tf-core",
      },
      {
        src: "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter@4.10.0/dist/tf-converter.min.js",
        name: "tf-converter",
      },
      {
        src: "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl@4.10.0/dist/tf-backend-webgl.min.js",
        name: "tf-webgl",
      },
      {
        src: "https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection@2.1.0/dist/pose-detection.min.js",
        name: "pose-detection",
      },
    ];

    for (const script of scripts) {
      await this.loadScript(script.src, script.name);
    }

    let attempts = 0;
    while (!window.tf && attempts < 50) {
      await new Promise((r) => setTimeout(r, 100));
      attempts++;
    }
    if (!window.tf) throw new Error("TensorFlow.js failed to load");

    await tf.setBackend("webgl");
    await tf.ready();
  }

  loadScript(src, name) {
    return new Promise((resolve, reject) => {
      if (document.querySelector(`script[src="${src}"]`)) {
        resolve();
        return;
      }
      const script = document.createElement("script");
      script.src = src;
      script.async = false;
      script.onload = () => resolve();
      script.onerror = () => reject(new Error(`Failed to load script: ${name}`));
      document.head.appendChild(script);
    });
  }

  async createDetector() {
    if (!window.poseDetection) {
      throw new Error("Pose detection library not available");
    }

    const model = poseDetection.SupportedModels.MoveNet;
    const detectorConfig = {
      modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
      minPoseScore: 0.20, // Lowered to catch more poses
    };

    this.detector = await poseDetection.createDetector(model, detectorConfig);
  }

  async startCamera() {
    if (this.stream) this.stopCamera();

    const constraints = {
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
      audio: false,
    };

    this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    this.video.srcObject = this.stream;

    return new Promise((resolve, reject) => {
      this.video.onloadedmetadata = () => {
        this.video
          .play()
          .then(() => {
            this.canvas.width = this.video.videoWidth || 640;
            this.canvas.height = this.video.videoHeight || 480;
            resolve();
          })
          .catch(reject);
      };
      this.video.onerror = reject;
    });
  }

  stopCamera() {
    this.isRunning = false;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    if (this.video) this.video.srcObject = null;
  }

  async startDetection(onFrame) {
    if (!this.detector) throw new Error("Detector not initialized");

    this.isRunning = true;
    this.resetReps();

    const detect = async () => {
      if (!this.isRunning) return;

      if (this.video.readyState >= 2) {
        try {
          const poses = await this.detector.estimatePoses(this.video);

          if (poses.length > 0 && this.hasRequiredBodyParts(poses[0].keypoints)) {
            const kp = this.smoothKeypoints(poses[0].keypoints);
            this.detectReady(kp);

            // Only analyze push-ups if the user is in position
            if (this.isReady) {
              this.detectPushup(kp);
            } else {
              this.phase = "unknown";
            }

            this.drawFrame(kp);

            if (onFrame) {
              onFrame({
                reps: this.repCount,
                phase: this.phase,
                ready: this.isReady,
                keypoints: kp,
              });
            }
          } else {
            // Reset posture state when critical body parts are missing
            this.isReady = false;
            this.readyFrames = 0;
            this.heightHistory = [];
            this.phase = "unknown";
            this.drawFrame(null);

            if (onFrame) {
              onFrame({
                reps: this.repCount,
                phase: "unknown",
                ready: false,
                keypoints: null,
              });
            }
          }
        } catch (e) {
          console.error("Detection frame error:", e);
        }
      }

      this.animationId = requestAnimationFrame(detect);
    };

    detect();
  }

  /**
   * Ensures essential torso keypoints are present before running logic
   * Relaxed: only need one shoulder and one hip visible
   */
  hasRequiredBodyParts(keypoints) {
    const ls = this.getKP(keypoints, "left_shoulder");
    const rs = this.getKP(keypoints, "right_shoulder");
    const lh = this.getKP(keypoints, "left_hip");
    const rh = this.getKP(keypoints, "right_hip");

    // Must have at least one shoulder AND at least one hip
    const hasTorso = (ls || rs) && (lh || rh);
    return Boolean(hasTorso);
  }

  smoothKeypoints(keypoints) {
    if (!this.smoothedKeypoints) {
      this.smoothedKeypoints = keypoints.map((k) => ({ ...k }));
      return this.smoothedKeypoints;
    }
    for (let i = 0; i < keypoints.length; i++) {
      this.smoothedKeypoints[i].x =
        this.alpha * this.smoothedKeypoints[i].x +
        (1 - this.alpha) * keypoints[i].x;
      this.smoothedKeypoints[i].y =
        this.alpha * this.smoothedKeypoints[i].y +
        (1 - this.alpha) * keypoints[i].y;
      this.smoothedKeypoints[i].score = keypoints[i].score;
    }
    return this.smoothedKeypoints;
  }

  detectPushup(keypoints) {
    let newPhase = this.phase;

    const ls = this.getKP(keypoints, "left_shoulder");
    const rs = this.getKP(keypoints, "right_shoulder");
    const lh = this.getKP(keypoints, "left_hip");
    const rh = this.getKP(keypoints, "right_hip");

    // Estimate body scale using torso length (shoulder to hip)
    let torsoLength = 100;
    if ((ls || rs) && (lh || rh)) {
      const sY = ls && rs ? (ls.y + rs.y) / 2 : (ls || rs).y;
      const hY = lh && rh ? (lh.y + rh.y) / 2 : (lh || rh).y;
      torsoLength = Math.max(Math.abs(hY - sY), 40);
    }

    // ===== METHOD 1: Normalized Height-based Detection =====
    let currentHeight = null;
    if (ls && rs) {
      currentHeight = (ls.y + rs.y) / 2;
    } else if (ls || rs) {
      currentHeight = (ls || rs).y;
    }

    let heightPhase = null;
    if (currentHeight !== null) {
      this.heightHistory.push(currentHeight);
      if (this.heightHistory.length > this.heightWindow) {
        this.heightHistory.shift();
      }

      if (this.heightHistory.length >= 5) {
        const minH = Math.min(...this.heightHistory);
        const maxH = Math.max(...this.heightHistory);
        const range = maxH - minH;

        // Require vertical movement to be at least 30% of torso length (lowered from 35%)
        if (range > torsoLength * 0.30) {
          const ratio = (currentHeight - minH) / (range + 1e-6);
          if (ratio > 0.65) heightPhase = "down";
          else if (ratio < 0.35) heightPhase = "up";
        }
      }
    }

    // ===== METHOD 2: Angle-based Fallback =====
    const le = this.getKP(keypoints, "left_elbow");
    const re = this.getKP(keypoints, "right_elbow");
    const lw = this.getKP(keypoints, "left_wrist");
    const rw = this.getKP(keypoints, "right_wrist");

    let anglePhase = null;
    if (ls && le && lw) {
      const leftAngle = this.angle(ls, le, lw);
      if (leftAngle < this.pushupDownAngle) anglePhase = "down";
      else if (leftAngle > this.pushupUpAngle) anglePhase = "up";
    } else if (rs && re && rw) {
      const rightAngle = this.angle(rs, re, rw);
      if (rightAngle < this.pushupDownAngle) anglePhase = "down";
      else if (rightAngle > this.pushupUpAngle) anglePhase = "up";
    }

    // ===== Method Integration =====
    if (heightPhase && anglePhase) {
      newPhase = heightPhase === anglePhase ? heightPhase : anglePhase;
    } else {
      newPhase = heightPhase || anglePhase || this.phase;
    }

    // ===== Temporal Majority Filter (3 frames) =====
    this.phaseHistory.push(newPhase);
    if (this.phaseHistory.length > 3) this.phaseHistory.shift();

    const counts = {};
    this.phaseHistory.forEach((p) => {
      if (p) counts[p] = (counts[p] || 0) + 1;
    });

    let smoothed = this.phase;
    if (Object.keys(counts).length > 0) {
      smoothed = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
    }

    // Incremental rep counter on down -> up cycle completion
    if (this.lastPhase === "down" && smoothed === "up") {
      this.repCount++;
    }

    if (smoothed === "down" || smoothed === "up") {
      this.lastPhase = smoothed;
    }
    this.phase = smoothed;
  }

  detectReady(keypoints) {
    // Count visible keypoints with relaxed threshold
    const visible = keypoints.filter((k) => k.score > this.minConfidence);

    // Relaxed: need at least 5 visible keypoints (was 6)
    if (visible.length < 5) {
      this.readyFrames = Math.max(0, this.readyFrames - 1);
      if (this.readyFrames < this.readyLostThreshold) this.isReady = false;
      return;
    }

    const ls = this.getKP(keypoints, "left_shoulder");
    const rs = this.getKP(keypoints, "right_shoulder");
    const lh = this.getKP(keypoints, "left_hip");
    const rh = this.getKP(keypoints, "right_hip");

    if ((!ls && !rs) || (!lh && !rh)) {
      this.readyFrames = Math.max(0, this.readyFrames - 1);
      if (this.readyFrames < this.readyLostThreshold) this.isReady = false;
      return;
    }

    const h = this.canvas.height;
    const shoulderY = ls && rs ? (ls.y + rs.y) / 2 : (ls || rs).y;
    const hipY = lh && rh ? (lh.y + rh.y) / 2 : (lh || rh).y;

    // The shoulders and hips should be in a relatively horizontal alignment
    // Relaxed tolerance: 30% of canvas height (was 25%)
    const bodyHorizontal = Math.abs(shoulderY - hipY) < h * 0.30;

    if (bodyHorizontal) {
      this.readyFrames++;
      if (this.readyFrames >= this.readyThreshold) this.isReady = true;
    } else {
      this.readyFrames = Math.max(0, this.readyFrames - 1);
      if (this.readyFrames < this.readyLostThreshold) {
        this.isReady = false;
      }
    }
  }

  angle(a, b, c) {
    const ba = { x: a.x - b.x, y: a.y - b.y };
    const bc = { x: c.x - b.x, y: c.y - b.y };
    const dot = ba.x * bc.x + ba.y * bc.y;
    const magBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y);
    const magBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y);
    const cos = dot / (magBA * magBC + 1e-6);
    return Math.acos(Math.max(-1, Math.min(1, cos))) * (180 / Math.PI);
  }

  getKP(keypoints, name) {
    const kp = keypoints.find((k) => k.name === name);
    return kp && kp.score >= this.minConfidence ? kp : null;
  }

  drawFrame(keypoints) {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Draw video feed mirrored
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(this.video, 0, 0, w, h);
    ctx.restore();

    if (!keypoints || !this.isReady) {
      const overlayText = !keypoints
        ? "No Body Detected"
        : "Get into Push-Up Position";
      this.drawOverlay(overlayText, "rgba(239, 68, 68, 0.85)");
      return;
    }

    const mirrored = keypoints.map((k) => ({ ...k, x: w - k.x }));
    this.drawSkeleton(mirrored);

    // Rep Counter Display
    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.beginPath();
    ctx.roundRect(14, 14, 80, 52, 8);
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 26px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(this.repCount.toString(), 54, 44);
    ctx.font = "bold 9px sans-serif";
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.fillText("REPS", 54, 58);
    ctx.restore();

    // Ready Status Pill
    ctx.save();
    ctx.fillStyle = "rgba(34, 197, 94, 0.9)";
    ctx.beginPath();
    ctx.arc(w - 30, 30, 14, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 8px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("READY", w - 30, 33);
    ctx.restore();

    // Movement Phase Indicator
    const phaseColor =
      this.phase === "down"
        ? "#3b82f6"
        : this.phase === "up"
        ? "#22c55e"
        : "#a1a1aa";
    ctx.save();
    ctx.fillStyle = phaseColor;
    ctx.font = "bold 16px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(this.phase.toUpperCase(), 16, h - 20);
    ctx.restore();
  }

  drawSkeleton(keypoints) {
    const ctx = this.ctx;
    const connections = [
      ["left_shoulder", "right_shoulder"],
      ["left_shoulder", "left_elbow"],
      ["right_shoulder", "right_elbow"],
      ["left_elbow", "left_wrist"],
      ["right_elbow", "right_wrist"],
      ["left_shoulder", "left_hip"],
      ["right_shoulder", "right_hip"],
      ["left_hip", "right_hip"],
      ["left_hip", "left_knee"],
      ["right_hip", "right_knee"],
      ["left_knee", "left_ankle"],
      ["right_knee", "right_ankle"],
    ];

    connections.forEach(([aName, bName]) => {
      const a = keypoints.find((k) => k.name === aName);
      const b = keypoints.find((k) => k.name === bName);
      if (
        a &&
        b &&
        a.score >= this.minConfidence &&
        b.score >= this.minConfidence
      ) {
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.strokeStyle =
          this.phase === "down" ? "#60a5fa" : "rgba(255,255,255,0.65)";
        ctx.lineWidth = 3;
        ctx.lineCap = "round";
        ctx.stroke();
      }
    });

    keypoints.forEach((kp) => {
      if (kp.score >= this.minConfidence) {
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = "#3b82f6";
        ctx.fill();
      }
    });
  }

  drawOverlay(text, color) {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.save();
    ctx.fillStyle = "rgba(0, 0, 0, 0.35)";
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = color || "#ffffff";
    ctx.font = "600 16px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(text, w / 2, h / 2);
    ctx.restore();
  }

  resetReps() {
    this.repCount = 0;
    this.phase = "up";
    this.lastPhase = "up";
    this.phaseHistory = [];
    this.heightHistory = [];
    this.isReady = false;
    this.readyFrames = 0;
    this.smoothedKeypoints = null;
  }
}

window.PoseDetector = PoseDetector;
