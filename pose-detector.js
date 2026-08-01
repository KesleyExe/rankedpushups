/**
 * PoseDetector - TensorFlow.js MoveNet for Push-Up Detection
 * v3.2: Graceful tracking loss, visual depth gauge, consistent phase detection
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
    this.phaseLockFrames = 0; // Frames since last phase change
    this.minPhaseLock = 3; // Minimum frames before phase can change

    // Height tracking
    this.heightHistory = [];
    this.heightWindow = 12;
    this.baselineHeight = null; // Calibrated "up" position height

    // Ready detection
    this.isReady = false;
    this.readyFrames = 0;
    this.readyThreshold = 10;
    this.readyLostThreshold = 6;

    // Tracking resilience
    this.lostFrames = 0;
    this.maxLostFrames = 10;
    this.lastGoodKeypoints = null;
    this.trackingQuality = 1.0; // 0.0 - 1.0

    // Smoothing (Exponential Moving Average)
    this.smoothedKeypoints = null;
    this.alpha = 0.45;

    // Confidence thresholds
    this.minConfidence = 0.25; // Lowered for better low-position tracking
    this.minPoseScore = 0.18;
    this.pushupDownAngle = 95;  // Tighter down threshold
    this.pushupUpAngle = 150;   // Tighter up threshold

    // Depth gauge for visual feedback (0.0 = up, 1.0 = down)
    this.depthRatio = 0.0;

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
      minPoseScore: this.minPoseScore,
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
          let kp = null;
          let hasPose = false;

          if (poses.length > 0) {
            const rawKP = poses[0].keypoints;
            // Be lenient: if we have any body keypoints, try to use them
            const bodyParts = rawKP.filter(k => 
              ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 
               'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'].includes(k.name)
            );
            const visibleBody = bodyParts.filter(k => k.score > this.minConfidence);

            if (visibleBody.length >= 4) {
              hasPose = true;
              this.lastGoodKeypoints = rawKP.map(k => ({...k}));
              this.lostFrames = 0;
              kp = this.smoothKeypoints(rawKP);
            } else if (this.lostFrames < this.maxLostFrames && this.lastGoodKeypoints) {
              // Graceful degradation: use last known keypoints with decay
              this.lostFrames++;
              hasPose = true;
              // Blend last good with current (even if current is poor)
              const blended = rawKP.map((k, i) => {
                const last = this.lastGoodKeypoints[i];
                if (!last) return k;
                const decay = 1 - (this.lostFrames / this.maxLostFrames);
                return {
                  ...k,
                  x: k.x * (1 - decay) + last.x * decay,
                  y: k.y * (1 - decay) + last.y * decay,
                  score: Math.max(k.score, last.score * decay)
                };
              });
              kp = this.smoothKeypoints(blended);
            }
          }

          if (hasPose && kp) {
            this.trackingQuality = Math.max(0, 1 - (this.lostFrames / this.maxLostFrames));
            this.detectReady(kp);

            if (this.isReady) {
              this.detectPushup(kp);
            } else {
              this.phase = "unknown";
              this.depthRatio = 0.5;
            }

            this.drawFrame(kp);

            if (onFrame) {
              onFrame({
                reps: this.repCount,
                phase: this.phase,
                ready: this.isReady,
                keypoints: kp,
                depth: this.depthRatio,
                quality: this.trackingQuality,
              });
            }
          } else {
            this.lostFrames++;
            this.trackingQuality = 0;
            this.isReady = false;
            this.readyFrames = 0;
            this.heightHistory = [];
            this.phase = "unknown";
            this.depthRatio = 0.5;
            this.drawFrame(null);

            if (onFrame) {
              onFrame({
                reps: this.repCount,
                phase: "unknown",
                ready: false,
                keypoints: null,
                depth: 0.5,
                quality: 0,
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
   * Relaxed body detection - needs only 4 visible body keypoints
   */
  hasRequiredBodyParts(keypoints) {
    const bodyNames = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
                       'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'];
    const visible = keypoints.filter(k => bodyNames.includes(k.name) && k.score > this.minConfidence);
    return visible.length >= 4;
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
    const ls = this.getKP(keypoints, "left_shoulder");
    const rs = this.getKP(keypoints, "right_shoulder");
    const lh = this.getKP(keypoints, "left_hip");
    const rh = this.getKP(keypoints, "right_hip");
    const le = this.getKP(keypoints, "left_elbow");
    const re = this.getKP(keypoints, "right_elbow");
    const lw = this.getKP(keypoints, "left_wrist");
    const rw = this.getKP(keypoints, "right_wrist");

    // ===== Body scale estimation =====
    let torsoLength = 100;
    if ((ls || rs) && (lh || rh)) {
      const sY = ls && rs ? (ls.y + rs.y) / 2 : (ls || rs).y;
      const hY = lh && rh ? (lh.y + rh.y) / 2 : (lh || rh).y;
      torsoLength = Math.max(Math.abs(hY - sY), 40);
    }

    // ===== Height-based depth detection =====
    let currentHeight = null;
    if (ls && rs) {
      currentHeight = (ls.y + rs.y) / 2;
    } else if (ls || rs) {
      currentHeight = (ls || rs).y;
    }

    let heightPhase = null;
    let heightDepth = 0.5;

    if (currentHeight !== null) {
      this.heightHistory.push(currentHeight);
      if (this.heightHistory.length > this.heightWindow) {
        this.heightHistory.shift();
      }

      if (this.heightHistory.length >= 5) {
        const minH = Math.min(...this.heightHistory);
        const maxH = Math.max(...this.heightHistory);
        const range = maxH - minH;

        // Calibrate baseline on the fly
        if (range > torsoLength * 0.25) {
          this.baselineHeight = maxH;
          heightDepth = (currentHeight - minH) / (range + 1e-6);
          if (heightDepth > 0.60) heightPhase = "down";
          else if (heightDepth < 0.40) heightPhase = "up";
        } else if (this.baselineHeight) {
          // Not enough range yet, use baseline
          const distFromBase = Math.abs(currentHeight - this.baselineHeight);
          heightDepth = Math.min(distFromBase / (torsoLength * 0.5), 1.0);
          if (currentHeight > this.baselineHeight + torsoLength * 0.15) heightPhase = "down";
          else if (currentHeight < this.baselineHeight - torsoLength * 0.05) heightPhase = "up";
        }
      }
    }

    // ===== Angle-based detection =====
    let anglePhase = null;
    let angleDepth = 0.5;

    if (ls && le && lw) {
      const leftAngle = this.angle(ls, le, lw);
      angleDepth = Math.min(Math.max((this.pushupUpAngle - leftAngle) / (this.pushupUpAngle - this.pushupDownAngle), 1), 0);
      if (leftAngle < this.pushupDownAngle) anglePhase = "down";
      else if (leftAngle > this.pushupUpAngle) anglePhase = "up";
    } else if (rs && re && rw) {
      const rightAngle = this.angle(rs, re, rw);
      angleDepth = Math.min(Math.max((this.pushupUpAngle - rightAngle) / (this.pushupUpAngle - this.pushupDownAngle), 1), 0);
      if (rightAngle < this.pushupDownAngle) anglePhase = "down";
      else if (rightAngle > this.pushupUpAngle) anglePhase = "up";
    }

    // ===== Fuse methods =====
    let newPhase = this.phase;
    let newDepth = 0.5;

    if (heightPhase && anglePhase) {
      newPhase = heightPhase === anglePhase ? heightPhase : 
                 (heightDepth > 0.5 ? "down" : "up");
      newDepth = (heightDepth + angleDepth) / 2;
    } else {
      newPhase = heightPhase || anglePhase || this.phase;
      newDepth = heightDepth !== 0.5 ? heightDepth : angleDepth;
    }

    // ===== Phase locking to prevent jitter =====
    if (newPhase !== this.phase && this.phaseLockFrames < this.minPhaseLock) {
      this.phaseLockFrames++;
      newPhase = this.phase; // Stay in current phase until lock expires
    } else {
      this.phaseLockFrames = 0;
    }

    // ===== Temporal majority filter (3 frames) =====
    this.phaseHistory.push(newPhase);
    if (this.phaseHistory.length > 3) this.phaseHistory.shift();

    const counts = {};
    this.phaseHistory.forEach((p) => {
      if (p && p !== "unknown") counts[p] = (counts[p] || 0) + 1;
    });

    let smoothed = this.phase;
    if (Object.keys(counts).length > 0) {
      const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
      if (sorted[0][1] >= 2) smoothed = sorted[0][0];
    }

    // ===== Rep counting =====
    if (this.lastPhase === "down" && smoothed === "up") {
      this.repCount++;
      this.phaseLockFrames = 0;
    }

    if (smoothed === "down" || smoothed === "up") {
      this.lastPhase = smoothed;
    }
    this.phase = smoothed;
    this.depthRatio = newDepth;
  }

  detectReady(keypoints) {
    const visible = keypoints.filter((k) => k.score > this.minConfidence);

    if (visible.length < 4) {
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

    // Relaxed horizontal alignment check
    const bodyHorizontal = Math.abs(shoulderY - hipY) < h * 0.32;

    if (bodyHorizontal) {
      this.readyFrames++;
      if (this.readyFrames >= this.readyThreshold) this.isReady = true;
    } else {
      this.readyFrames = Math.max(0, this.readyFrames - 1);
      if (this.readyFrames < Math.floor(this.readyThreshold * 0.4)) {
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
      this.drawTrackingQuality(w, h, 0);
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

    // ===== DEPTH GAUGE (right side) =====
    const gaugeX = w - 24;
    const gaugeY = h / 2 - 60;
    const gaugeH = 120;
    const gaugeW = 12;

    // Background
    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.beginPath();
    ctx.roundRect(gaugeX - gaugeW/2, gaugeY, gaugeW, gaugeH, 6);
    ctx.fill();

    // Fill
    const fillH = gaugeH * this.depthRatio;
    const fillY = gaugeY + gaugeH - fillH;
    const fillColor = this.phase === "down" ? "#3b82f6" : 
                      this.phase === "up" ? "#22c55e" : "#a1a1aa";
    ctx.fillStyle = fillColor;
    ctx.beginPath();
    ctx.roundRect(gaugeX - gaugeW/2, fillY, gaugeW, fillH, 6);
    ctx.fill();

    // Labels
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.font = "bold 9px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("UP", gaugeX, gaugeY - 6);
    ctx.fillText("DOWN", gaugeX, gaugeY + gaugeH + 12);
    ctx.restore();

    // ===== COACHING ARROW =====
    const arrowText = this.phase === "up" ? "▼ PUSH DOWN" : 
                      this.phase === "down" ? "▲ PUSH UP" : "";
    if (arrowText) {
      ctx.save();
      const arrowColor = this.phase === "up" ? "#3b82f6" : "#22c55e";
      ctx.fillStyle = arrowColor;
      ctx.font = "bold 18px sans-serif";
      ctx.textAlign = "center";
      ctx.shadowColor = "rgba(0,0,0,0.8)";
      ctx.shadowBlur = 8;
      ctx.fillText(arrowText, w / 2, h / 2 + 40);
      ctx.restore();
    }

    // Tracking quality indicator
    this.drawTrackingQuality(w, h, this.trackingQuality);

    // Phase text
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

  drawTrackingQuality(w, h, quality) {
    const ctx = this.ctx;
    const barW = 60;
    const barH = 4;
    const x = w - barW - 14;
    const y = h - 14;

    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.beginPath();
    ctx.roundRect(x, y, barW, barH, 2);
    ctx.fill();

    const color = quality > 0.7 ? "#22c55e" : quality > 0.4 ? "#eab308" : "#ef4444";
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(x, y, barW * quality, barH, 2);
    ctx.fill();
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

    const lineColor = this.phase === "down" ? "#60a5fa" : 
                      this.phase === "up" ? "rgba(255,255,255,0.65)" : "rgba(255,255,255,0.3)";

    connections.forEach(([aName, bName]) => {
      const a = keypoints.find((k) => k.name === aName);
      const b = keypoints.find((k) => k.name === bName);
      if (a && b && a.score >= this.minConfidence && b.score >= this.minConfidence) {
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.strokeStyle = lineColor;
        ctx.lineWidth = 3;
        ctx.lineCap = "round";
        ctx.stroke();
      }
    });

    keypoints.forEach((kp) => {
      if (kp.score >= this.minConfidence) {
        ctx.beginPath();
        ctx.arc(kp.x, kp.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = this.phase === "down" ? "#60a5fa" : "#3b82f6";
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
    this.phaseLockFrames = 0;
    this.heightHistory = [];
    this.baselineHeight = null;
    this.isReady = false;
    this.readyFrames = 0;
    this.lostFrames = 0;
    this.lastGoodKeypoints = null;
    this.trackingQuality = 1.0;
    this.depthRatio = 0.0;
    this.smoothedKeypoints = null;
  }
}

window.PoseDetector = PoseDetector;
