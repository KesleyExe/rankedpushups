/**
 * PoseDetector - TensorFlow.js MoveNet for Push-Up Detection
 * v3.4: Added jitter debouncing, strict pixel minimums, and longer calibration
 *       windows to prevent ghost reps while laying still.
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
    this.phase = "up";         // "up", "down", or "unknown"
    this.lastPhase = "up";     // Last confirmed phase (for rep counting)
    
    // Height tracking
    this.heightHistory = [];
    this.heightWindow = 90;    // Increased to ~3 seconds to prevent instant false calibration
    this.upPositionY = null;   // Calibrated UP position (smallest y = highest on screen)
    this.downPositionY = null; // Calibrated DOWN position (largest y = lowest on screen)
    
    // Ready detection
    this.isReady = false;
    this.readyFrames = 0;
    this.readyThreshold = 6;      // ~0.2s at 30fps
    this.readyLostThreshold = 4;  // Brief dropouts allowed

    // Tracking resilience
    this.lostFrames = 0;
    this.maxLostFrames = 12;
    this.lastGoodKeypoints = null;

    // Smoothing
    this.smoothedKeypoints = null;
    this.alpha = 0.35;  // Slightly more smoothing for stability

    // Confidence thresholds
    this.minConfidence = 0.18;    // For logic (pushup counting, ready check)
    this.drawConfidence = 0.10; // For visual skeleton (draw even faint detections)
    this.minPoseScore = 0.15;   // Detector-level threshold
    
    // Pushup thresholds (as fraction of torso length)
    this.downDropRatio = 0.38;  // Must drop 38% of torso to register "down"
    this.upDropRatio = 0.12;    // Must be within 12% of torso from top to register "up"
    
    // Angle thresholds (degrees)
    this.pushupDownAngle = 95;
    this.pushupUpAngle = 150;

    // Depth gauge (0.0 = up, 1.0 = down)
    this.depthRatio = 0.0;

    // Debouncing (Anti-jitter)
    this.pendingPhase = "unknown";
    this.pendingFrames = 0;
    this.requiredHoldFrames = 4; // Must hold a pose for 4 consecutive frames to register

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
      { src: "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core@4.10.0/dist/tf-core.min.js", name: "tf-core" },
      { src: "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter@4.10.0/dist/tf-converter.min.js", name: "tf-converter" },
      { src: "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl@4.10.0/dist/tf-backend-webgl.min.js", name: "tf-webgl" },
      { src: "https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection@2.1.0/dist/pose-detection.min.js", name: "pose-detection" },
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
      if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
      const script = document.createElement("script");
      script.src = src;
      script.async = false;
      script.onload = () => resolve();
      script.onerror = () => reject(new Error(`Failed to load script: ${name}`));
      document.head.appendChild(script);
    });
  }

  async createDetector() {
    if (!window.poseDetection) throw new Error("Pose detection library not available");
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
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    };
    this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    this.video.srcObject = this.stream;
    return new Promise((resolve, reject) => {
      this.video.onloadedmetadata = () => {
        this.video.play().then(() => {
          this.canvas.width = this.video.videoWidth || 640;
          this.canvas.height = this.video.videoHeight || 480;
          resolve();
        }).catch(reject);
      };
      this.video.onerror = reject;
    });
  }

  stopCamera() {
    this.isRunning = false;
    if (this.animationId) { cancelAnimationFrame(this.animationId); this.animationId = null; }
    if (this.stream) { this.stream.getTracks().forEach((t) => t.stop()); this.stream = null; }
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
            // Need at least 3 body keypoints to attempt tracking
            const bodyNames = ['nose','left_shoulder','right_shoulder','left_hip','right_hip',
                               'left_elbow','right_elbow','left_wrist','right_wrist'];
            const visibleBody = rawKP.filter(k => bodyNames.includes(k.name) && k.score > this.minConfidence);

            if (visibleBody.length >= 3) {
              hasPose = true;
              this.lastGoodKeypoints = rawKP.map(k => ({...k}));
              this.lostFrames = 0;
              kp = this.smoothKeypoints(rawKP);
            } else if (this.lostFrames < this.maxLostFrames && this.lastGoodKeypoints) {
              // Graceful degradation: blend last good pose with current
              this.lostFrames++;
              hasPose = true;
              const decay = Math.max(0, 1 - (this.lostFrames / this.maxLostFrames));
              const blended = rawKP.map((k, i) => {
                const last = this.lastGoodKeypoints[i];
                if (!last) return k;
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
            const quality = Math.max(0, 1 - (this.lostFrames / this.maxLostFrames));
            this.detectReady(kp);

            if (this.isReady) {
              this.detectPushup(kp);
            } else {
              this.phase = "unknown";
              this.depthRatio = 0.0;
            }

            this.drawFrame(kp);

            if (onFrame) {
              onFrame({
                reps: this.repCount,
                phase: this.phase,
                ready: this.isReady,
                keypoints: kp,
                depth: this.depthRatio,
                quality: quality,
              });
            }
          } else {
            this.lostFrames++;
            this.isReady = false;
            this.readyFrames = 0;
            this.phase = "unknown";
            this.depthRatio = 0.0;
            this.drawFrame(null);
            if (onFrame) {
              onFrame({ reps: this.repCount, phase: "unknown", ready: false, keypoints: null, depth: 0, quality: 0 });
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

  smoothKeypoints(keypoints) {
    if (!this.smoothedKeypoints) {
      this.smoothedKeypoints = keypoints.map((k) => ({ ...k }));
      return this.smoothedKeypoints;
    }
    for (let i = 0; i < keypoints.length; i++) {
      this.smoothedKeypoints[i].x = this.alpha * this.smoothedKeypoints[i].x + (1 - this.alpha) * keypoints[i].x;
      this.smoothedKeypoints[i].y = this.alpha * this.smoothedKeypoints[i].y + (1 - this.alpha) * keypoints[i].y;
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

    // Estimate torso length for scale
    let torsoLength = 100;
    const hasShoulder = ls || rs;
    const hasHip = lh || rh;
    if (hasShoulder && hasHip) {
      const sY = ls && rs ? (ls.y + rs.y) / 2 : (ls || rs).y;
      const hY = lh && rh ? (lh.y + rh.y) / 2 : (lh || rh).y;
      // Increased minimum torso length from 50 to 100 to prevent tiny pixel thresholds
      torsoLength = Math.max(Math.abs(hY - sY), 100); 
    }

    // ===== Current shoulder height =====
    let currentHeight = null;
    if (ls && rs) currentHeight = (ls.y + rs.y) / 2;
    else if (ls || rs) currentHeight = (ls || rs).y;

    // ===== Calibrate up/down positions from history =====
    if (currentHeight !== null) {
      this.heightHistory.push(currentHeight);
      if (this.heightHistory.length > this.heightWindow) this.heightHistory.shift();

      if (this.heightHistory.length >= 10) { // Require slightly more history before math
        const sorted = [...this.heightHistory].sort((a, b) => a - b);
        this.upPositionY = sorted[Math.floor(sorted.length * 0.15)];
        this.downPositionY = sorted[Math.floor(sorted.length * 0.85)];
      } else {
        this.upPositionY = Math.min(this.upPositionY ?? Infinity, currentHeight);
      }
    }

    // ===== Height-based phase detection with hysteresis =====
    let heightPhase = null;
    let heightDepth = 0.0;

    if (currentHeight !== null && this.upPositionY !== null) {
      const drop = currentHeight - this.upPositionY;
      const range = Math.max((this.downPositionY || currentHeight) - this.upPositionY, torsoLength * 0.25);
      
      heightDepth = Math.max(0, Math.min(1, drop / range));

      // Absolute pixel minimums (e.g., must drop at least 35 pixels no matter what)
      const downThreshold = Math.max(torsoLength * this.downDropRatio, 35);
      const upThreshold = torsoLength * this.upDropRatio;

      if (drop > downThreshold) {
        heightPhase = "down";
      } else if (drop < upThreshold) {
        heightPhase = "up";
      }
    }

    // ===== Angle-based detection (fallback / confirmation) =====
    let anglePhase = null;
    let angleDepth = 0.0;

    const calcAngleDepth = (angle) => {
      const raw = (this.pushupUpAngle - angle) / (this.pushupUpAngle - this.pushupDownAngle);
      return Math.max(0, Math.min(1, raw));
    };

    if (ls && le && lw) {
      const leftAngle = this.angle(ls, le, lw);
      angleDepth = calcAngleDepth(leftAngle);
      if (leftAngle < this.pushupDownAngle) anglePhase = "down";
      else if (leftAngle > this.pushupUpAngle) anglePhase = "up";
    } else if (rs && re && rw) {
      const rightAngle = this.angle(rs, re, rw);
      angleDepth = calcAngleDepth(rightAngle);
      if (rightAngle < this.pushupDownAngle) anglePhase = "down";
      else if (rightAngle > this.pushupUpAngle) anglePhase = "up";
    }

    // ===== Fuse methods =====
    let rawNewPhase = this.phase;
    let newDepth = 0.0;

    if (heightPhase) {
      rawNewPhase = heightPhase;
      newDepth = heightDepth;
    } else if (anglePhase) {
      rawNewPhase = anglePhase;
      newDepth = angleDepth;
    }

    // ===== Debouncing (Hold to Confirm Phase) =====
    // Prevents a single frame of jitter from counting as a pushup
    if (rawNewPhase !== this.phase && rawNewPhase !== "unknown") {
      if (rawNewPhase === this.pendingPhase) {
        this.pendingFrames++;
        if (this.pendingFrames >= this.requiredHoldFrames) {
          
          // Phase is confirmed, check for rep
          if (this.lastPhase === "down" && rawNewPhase === "up") {
            this.repCount++;
          }
          
          this.lastPhase = rawNewPhase;
          this.phase = rawNewPhase;
          this.pendingFrames = 0; // Reset
        }
      } else {
        // Started seeing a new phase
        this.pendingPhase = rawNewPhase;
        this.pendingFrames = 1;
      }
    } else {
      // Either phase hasn't changed, or phase is unknown
      this.pendingFrames = 0; 
    }

    // Update depth instantly for a smooth UI, even if the phase is waiting to debounce
    this.depthRatio = newDepth;
  }

  detectReady(keypoints) {
    const visible = keypoints.filter((k) => k.score > this.minConfidence);
    if (visible.length < 3) {
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

    // Relaxed: shoulders and hips within 40% of canvas height
    const bodyHorizontal = Math.abs(shoulderY - hipY) < h * 0.40;

    if (bodyHorizontal) {
      this.readyFrames++;
      if (this.readyFrames >= this.readyThreshold) this.isReady = true;
    } else {
      this.readyFrames = Math.max(0, this.readyFrames - 1);
      if (this.readyFrames < Math.floor(this.readyThreshold * 0.5)) {
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

  // Get keypoint for drawing (lower threshold)
  getKPDraw(keypoints, name) {
    const kp = keypoints.find((k) => k.name === name);
    return kp && kp.score >= this.drawConfidence ? kp : null;
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

    if (!keypoints) {
      this.drawOverlay("No Body Detected", "rgba(239, 68, 68, 0.85)");
      return;
    }

    const mirrored = keypoints.map((k) => ({ ...k, x: w - k.x }));
    this.drawSkeleton(mirrored);

    // Rep Counter
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

    if (!this.isReady) {
      this.drawOverlay("Get into Push-Up Position", "rgba(234, 179, 8, 0.85)");
      return;
    }

    // Ready pill
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
    const gaugeY = h / 2 - 70;
    const gaugeH = 140;
    const gaugeW = 14;

    ctx.save();
    // Background track
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.beginPath();
    ctx.roundRect(gaugeX - gaugeW/2, gaugeY, gaugeW, gaugeH, 7);
    ctx.fill();

    // Fill from bottom up
    const fillH = gaugeH * this.depthRatio;
    const fillY = gaugeY + gaugeH - fillH;
    const fillColor = this.phase === "down" ? "#3b82f6" : 
                      this.phase === "up" ? "#22c55e" : "#71717a";
    ctx.fillStyle = fillColor;
    ctx.beginPath();
    ctx.roundRect(gaugeX - gaugeW/2, fillY, gaugeW, fillH, 7);
    ctx.fill();

    // Target line for "down"
    const downLineY = gaugeY + gaugeH * 0.62;
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(gaugeX - gaugeW/2 - 4, downLineY);
    ctx.lineTo(gaugeX + gaugeW/2 + 4, downLineY);
    ctx.stroke();

    // Labels
    ctx.fillStyle = "rgba(255,255,255,0.8)";
    ctx.font = "bold 9px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("UP", gaugeX, gaugeY - 8);
    ctx.fillText("DOWN", gaugeX, gaugeY + gaugeH + 14);
    ctx.restore();

    // ===== COACHING TEXT =====
    const arrowText = this.phase === "up" ? "▼ PUSH DOWN" : 
                      this.phase === "down" ? "▲ PUSH UP" : "";
    if (arrowText) {
      ctx.save();
      const arrowColor = this.phase === "up" ? "#60a5fa" : "#4ade80";
      ctx.fillStyle = arrowColor;
      ctx.font = "bold 20px sans-serif";
      ctx.textAlign = "center";
      ctx.shadowColor = "rgba(0,0,0,0.9)";
      ctx.shadowBlur = 10;
      ctx.fillText(arrowText, w / 2, h / 2 + 50);
      ctx.restore();
    }

    // Phase text
    const phaseColor = this.phase === "down" ? "#3b82f6" : 
                       this.phase === "up" ? "#22c55e" : "#a1a1aa";
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

    const lineColor = this.phase === "down" ? "rgba(96, 165, 250, 0.9)" : 
                      this.phase === "up" ? "rgba(255, 255, 255, 0.7)" : "rgba(255, 255, 255, 0.35)";

    connections.forEach(([aName, bName]) => {
      const a = this.getKPDraw(keypoints, aName);
      const b = this.getKPDraw(keypoints, bName);
      if (a && b) {
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
      if (kp.score >= this.drawConfidence) {
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
    ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
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
    this.heightHistory = [];
    this.upPositionY = null;
    this.downPositionY = null;
    this.isReady = false;
    this.readyFrames = 0;
    this.lostFrames = 0;
    this.lastGoodKeypoints = null;
    this.depthRatio = 0.0;
    this.smoothedKeypoints = null;
    this.pendingPhase = "unknown";
    this.pendingFrames = 0;
  }
}

window.PoseDetector = PoseDetector;
