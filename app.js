// ===================== WEBRTC & GLOBAL STATE =====================
let socket = null;
let currentMatchId = null;
let currentRole = null; // "p1" or "p2"
let peerSocketId = null;
let pc = null;

let localStream = null;
let myName = localStorage.getItem("pushup_username") || "Player";
let myElo = parseInt(localStorage.getItem("pushup_elo")) || 800;

const rtcConfig = {
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" },
    { urls: "stun:stun1.l.google.com:19302" }
  ]
};

// ===================== DOM ELEMENTS =====================
const screens = {
  home: document.getElementById("screen-home"),
  matching: document.getElementById("screen-matching"),
  ready: document.getElementById("screen-ready"),
  compete: document.getElementById("screen-compete"),
  result: document.getElementById("screen-result"),
};

const sidebarVideo = document.getElementById("sidebar-video");
const localVideo = document.getElementById("local-video");
const remoteVideo = document.getElementById("remote-video");
const remotePlaceholder = document.getElementById("remote-placeholder");

// ===================== CAMERA SETUP =====================
async function initCamera() {
  try {
    localStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false
    });
    sidebarVideo.srcObject = localStream;
    localVideo.srcObject = localStream;
  } catch (err) {
    console.warn("Camera access failed or denied:", err);
  }
}

function showScreen(screenName) {
  Object.keys(screens).forEach(name => {
    screens[name].classList.toggle("active", name === screenName);
  });
}

// ===================== WEBRTC FUNCTIONS =====================
function createPeerConnection() {
  closePeerConnection();
  pc = new RTCPeerConnection(rtcConfig);

  if (localStream) {
    localStream.getTracks().forEach(track => pc.addTrack(track, localStream));
  }

  pc.ontrack = (event) => {
    if (event.streams && event.streams[0]) {
      remoteVideo.srcObject = event.streams[0];
      if (remotePlaceholder) remotePlaceholder.style.display = "none";
    }
  };

  pc.onicecandidate = (event) => {
    if (event.candidate && peerSocketId) {
      socket.emit("signal:ice-candidate", { target: peerSocketId, candidate: event.candidate });
    }
  };

  pc.onconnectionstatechange = () => {
    if (pc && (pc.connectionState === "disconnected" || pc.connectionState === "failed")) {
      if (remotePlaceholder) remotePlaceholder.style.display = "flex";
    }
  };
}

async function startWebRTCAsOfferer() {
  createPeerConnection();
  try {
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    socket.emit("signal:offer", { target: peerSocketId, offer });
  } catch (err) {
    console.error("Error creating WebRTC offer:", err);
  }
}

function closePeerConnection() {
  if (pc) {
    pc.close();
    pc = null;
  }
  if (remoteVideo) remoteVideo.srcObject = null;
  if (remotePlaceholder) remotePlaceholder.style.display = "flex";
}

// ===================== SOCKET INITIALIZATION =====================
function initSocket() {
  socket = io({ transports: ["websocket", "polling"] });

  socket.on("connect", () => {
    document.getElementById("status-dot").classList.add("online");
    document.getElementById("status-text").innerText = "Connected";
    socket.emit("user:register", { name: myName, elo: myElo });
  });

  socket.on("user:registered", (data) => {
    myName = data.name;
    myElo = data.elo;
    document.getElementById("sidebar-name").innerText = myName;
    document.getElementById("sidebar-elo").innerText = `${myElo} ELO`;
    document.getElementById("home-elo").innerText = myElo;
    document.getElementById("sidebar-avatar").innerText = myName.charAt(0).toUpperCase();
    document.getElementById("btn-match").disabled = false;
    document.getElementById("btn-match").innerText = "Start matching";
  });

  socket.on("queue:joined", () => {
    showScreen("matching");
  });

  socket.on("queue:left", () => {
    showScreen("home");
  });

  socket.on("match:ready_check", (data) => {
    currentMatchId = data.matchId;
    currentRole = data.youAre;
    peerSocketId = data.peerSocketId;

    document.getElementById("ready-opp-name").innerText = data.opponent.name;
    document.getElementById("ready-opp-name2").innerText = data.opponent.name;
    document.getElementById("compete-opp-name").innerText = data.opponent.name;

    showScreen("ready");
    socket.emit("match:ready", { matchId: currentMatchId, ready: true });

    // Establish WebRTC connection (P1 initiates)
    if (currentRole === "p1") {
      startWebRTCAsOfferer();
    }
  });

  socket.on("signal:offer", async ({ sender, offer }) => {
    if (!pc) createPeerConnection();
    try {
      await pc.setRemoteDescription(new RTCSessionDescription(offer));
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      socket.emit("signal:answer", { target: sender, answer });
    } catch (err) {
      console.error("Error handling offer:", err);
    }
  });

  socket.on("signal:answer", async ({ answer }) => {
    try {
      if (pc) await pc.setRemoteDescription(new RTCSessionDescription(answer));
    } catch (err) {
      console.error("Error handling answer:", err);
    }
  });

  socket.on("signal:ice-candidate", async ({ candidate }) => {
    try {
      if (pc) await pc.addIceCandidate(new RTCIceCandidate(candidate));
    } catch (err) {
      console.error("Error adding ICE candidate:", err);
    }
  });

  socket.on("match:countdown_start", () => {
    showScreen("compete");
  });

  socket.on("match:tick", (data) => {
    const yourReps = currentRole === "p1" ? data.p1Reps : data.p2Reps;
    const oppReps = currentRole === "p1" ? data.p2Reps : data.p1Reps;

    document.getElementById("compete-your-reps").innerText = yourReps;
    document.getElementById("compete-opp-reps").innerText = oppReps;
    document.getElementById("compete-timer").innerText = data.timeLeft;
  });

  socket.on("match:end", (data) => {
    closePeerConnection();

    document.getElementById("result-badge").innerText = data.result.toUpperCase();
    document.getElementById("result-title").innerText = data.result === "win" ? "Victory!" : (data.result === "loss" ? "Defeat" : "Draw");
    document.getElementById("result-your-reps").innerText = data.yourReps;
    document.getElementById("result-opp-reps").innerText = data.oppReps;
    document.getElementById("result-elo-change").innerText = (data.yourEloChange >= 0 ? "+" : "") + data.yourEloChange;
    document.getElementById("result-new-elo").innerText = data.newElo;

    localStorage.setItem("pushup_elo", data.newElo);
    document.getElementById("home-elo").innerText = data.newElo;
    document.getElementById("sidebar-elo").innerText = `${data.newElo} ELO`;

    showScreen("result");
    currentMatchId = null;
  });
}

// ===================== EVENT LISTENERS =====================
document.getElementById("btn-match").addEventListener("click", () => {
  if (socket) socket.emit("queue:join");
});

document.getElementById("btn-cancel").addEventListener("click", () => {
  if (socket) socket.emit("queue:leave");
});

document.getElementById("btn-forfeit").addEventListener("click", () => {
  if (socket && currentMatchId) {
    socket.emit("match:forfeit", { matchId: currentMatchId });
  }
});

document.getElementById("btn-play-again").addEventListener("click", () => {
  showScreen("home");
});

// Fetch Stats & Leaderboard
async function fetchStats() {
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();
    document.getElementById("stat-online").innerText = data.online;
    document.getElementById("stat-queue").innerText = data.inQueue;
    document.getElementById("stat-active").innerText = data.activeMatches;
    document.getElementById("stat-total").innerText = data.totalMatches;
  } catch (err) {}
}

async function fetchLeaderboard() {
  try {
    const res = await fetch("/api/leaderboard");
    const data = await res.json();
    const list = document.getElementById("leaderboard-list");
    list.innerHTML = data.map((item, idx) => `
      <div class="leaderboard-item">
        <span class="rank">#${idx + 1}</span>
        <span class="name">${item.name}</span>
        <span class="elo">${item.elo} ELO</span>
      </div>
    `).join("");
  } catch (err) {}
}

window.addEventListener("DOMContentLoaded", () => {
  initCamera();
  initSocket();
  fetchStats();
  fetchLeaderboard();
  setInterval(() => { fetchStats(); fetchLeaderboard(); }, 5000);
});
