const express = require("express");
const http = require("http");
const os = require("os");
const { Server } = require("socket.io");
const { v4: uuidv4 } = require("uuid");
const path = require("path");

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*" },
  pingTimeout: 60000,
  pingInterval: 25000,
  transports: ["websocket", "polling"]
});

app.use(express.static(path.join(__dirname, "public")));

// Redirect root domain "/" to "/app" to prevent "Cannot GET /" on Render
app.get("/", (req, res) => {
  res.redirect("/app");
});

app.get("/app", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "app.html"));
});

// ===================== DATA STORES =====================
const users = new Map();
const matches = new Map();
const matchHistory = [];
const queue = [];

// ===================== ELO SYSTEM =====================
function expectedScore(a, b) {
  return 1 / (1 + Math.pow(10, (b - a) / 400));
}

function calculateElo(youElo, oppElo, yourReps, oppReps) {
  const diff = yourReps - oppReps;
  let actual, result;
  if (diff > 0) { actual = 1; result = "win"; }
  else if (diff < 0) { actual = 0; result = "loss"; }
  else { actual = 0.5; result = "tie"; }

  const K_BASE = 16;
  const bonus = Math.min(Math.abs(diff) / 50, 1);
  const K_eff = Math.round(K_BASE * (1 + bonus));
  const expected = expectedScore(youElo, oppElo);
  const change = Math.round(K_eff * (actual - expected));

  return { change, result, diff, K_eff, expected, actual };
}

function validateReps(reps, durationSec) {
  const maxReps = durationSec * 5;
  return Math.max(0, Math.min(reps, maxReps));
}

// ===================== MATCH CLASS =====================
class Match {
  constructor(p1, p2) {
    this.id = uuidv4();
    this.p1 = p1;
    this.p2 = p2;
    this.p1Reps = 0;
    this.p2Reps = 0;
    this.p1Ready = false;
    this.p2Ready = false;
    this.phase = "found";
    this.countdown = 5;
    this.duration = 60;
    this.timeLeft = 0;
    this.startedAt = null;
    this.endedAt = null;
    this.timer = null;
    this.tickInterval = null;
  }

  startReadyCheck() {
    this.phase = "ready_check";
    this.sendTo(this.p1.socketId, "match:ready_check", {
      matchId: this.id,
      opponent: this.getOpponentData(this.p1),
      youAre: "p1",
      peerSocketId: this.p2.socketId
    });
    this.sendTo(this.p2.socketId, "match:ready_check", {
      matchId: this.id,
      opponent: this.getOpponentData(this.p2),
      youAre: "p2",
      peerSocketId: this.p1.socketId
    });
    console.log(`[Match ${this.id.slice(0,8)}] Ready check: ${this.p1.name} vs ${this.p2.name}`);
  }

  setReady(socketId, isReady) {
    if (socketId === this.p1.socketId) this.p1Ready = isReady;
    else if (socketId === this.p2.socketId) this.p2Ready = isReady;

    this.broadcast("match:ready_update", {
      p1Ready: this.p1Ready,
      p2Ready: this.p2Ready,
      p1Name: this.p1.name,
      p2Name: this.p2.name,
    });

    if (this.p1Ready && this.p2Ready && this.phase === "ready_check") {
      this.startCountdown();
    }
  }

  startCountdown() {
    this.phase = "countdown";
    this.timeLeft = this.countdown;
    this.broadcast("match:countdown_start", { seconds: this.countdown });
    console.log(`[Match ${this.id.slice(0,8)}] Countdown started`);

    this.timer = setInterval(() => {
      this.timeLeft--;
      this.broadcast("match:countdown_tick", { timeLeft: this.timeLeft });
      if (this.timeLeft <= 0) {
        clearInterval(this.timer);
        this.beginActivePhase();
      }
    }, 1000);
  }

  beginActivePhase() {
    this.phase = "active";
    this.timeLeft = this.duration;
    this.startedAt = Date.now();
    this.broadcast("match:start", { duration: this.duration });
    console.log(`[Match ${this.id.slice(0,8)}] Active phase started`);

    this.tickInterval = setInterval(() => {
      this.broadcast("match:tick", {
        p1Reps: this.p1Reps,
        p2Reps: this.p2Reps,
        timeLeft: this.timeLeft,
      });
    }, 500);

    this.timer = setInterval(() => {
      this.timeLeft--;
      if (this.timeLeft <= 0) this.endMatch();
    }, 1000);
  }

  recordRep(playerSocketId, count) {
    if (this.phase !== "active") return;
    const validated = validateReps(count, this.duration);
    if (playerSocketId === this.p1.socketId) this.p1Reps = validated;
    else if (playerSocketId === this.p2.socketId) this.p2Reps = validated;
  }

  forfeit(playerSocketId) {
    if (this.phase === "ended") return;
    if (playerSocketId === this.p1.socketId) this.p1Reps = 0;
    else this.p2Reps = 0;
    this.endMatch("forfeit");
  }

  endMatch(reason = "time") {
    if (this.phase === "ended") return;
    this.phase = "ended";
    this.endedAt = Date.now();
    clearInterval(this.timer);
    clearInterval(this.tickInterval);

    const p1Result = calculateElo(this.p1.elo, this.p2.elo, this.p1Reps, this.p2Reps);
    const p2Result = calculateElo(this.p2.elo, this.p1.elo, this.p2Reps, this.p1Reps);

    this.p1.elo = Math.max(100, this.p1.elo + p1Result.change);
    this.p2.elo = Math.max(100, this.p2.elo + p2Result.change);
    this.p1.inMatch = false;
    this.p2.inMatch = false;

    const record = {
      id: this.id,
      p1: { name: this.p1.name, eloBefore: this.p1.elo - p1Result.change, eloAfter: this.p1.elo, reps: this.p1Reps, change: p1Result.change },
      p2: { name: this.p2.name, eloBefore: this.p2.elo - p2Result.change, eloAfter: this.p2.elo, reps: this.p2Reps, change: p2Result.change },
      result: { p1: p1Result.result, p2: p2Result.result },
      diff: p1Result.diff,
      endedAt: this.endedAt,
      reason,
    };
    matchHistory.push(record);

    this.sendTo(this.p1.socketId, "match:end", {
      yourReps: this.p1Reps, oppReps: this.p2Reps,
      yourEloChange: p1Result.change, newElo: this.p1.elo,
      result: p1Result.result, diff: p1Result.diff,
      opponentName: this.p2.name, reason,
    });
    this.sendTo(this.p2.socketId, "match:end", {
      yourReps: this.p2Reps, oppReps: this.p1Reps,
      yourEloChange: p2Result.change, newElo: this.p2.elo,
      result: p2Result.result, diff: p2Result.diff,
      opponentName: this.p1.name, reason,
    });

    console.log(`[Match ${this.id.slice(0,8)}] Ended: ${this.p1.name} ${this.p1Reps} vs ${this.p2.name} ${this.p2Reps} (${reason})`);
    setTimeout(() => matches.delete(this.id), 30000);
  }

  getOpponentData(forPlayer) {
    const opp = forPlayer.socketId === this.p1.socketId ? this.p2 : this.p1;
    return { name: opp.name, elo: opp.elo };
  }

  broadcast(event, data) {
    io.to(this.p1.socketId).emit(event, data);
    io.to(this.p2.socketId).emit(event, data);
  }

  sendTo(socketId, event, data) {
    io.to(socketId).emit(event, data);
  }
}

// ===================== MATCHMAKING =====================
function tryMatchmaking() {
  if (queue.length < 2) return;
  const sorted = queue
    .map(sid => users.get(sid))
    .filter(u => u && u.inQueue && !u.inMatch)
    .sort((a, b) => a.elo - b.elo);

  const matched = new Set();
  for (let i = 0; i < sorted.length; i++) {
    if (matched.has(sorted[i].socketId)) continue;
    const queueTime = Date.now() - (sorted[i].queueJoinedAt || Date.now());
    const tolerance = Math.min(50 + Math.floor(queueTime / 2000) * 25, 300);

    for (let j = i + 1; j < sorted.length; j++) {
      if (matched.has(sorted[j].socketId)) continue;
      if (Math.abs(sorted[i].elo - sorted[j].elo) <= tolerance) {
        matched.add(sorted[i].socketId);
        matched.add(sorted[j].socketId);
        sorted[i].inQueue = false;
        sorted[j].inQueue = false;
        sorted[i].inMatch = true;
        sorted[j].inMatch = true;
        const idx1 = queue.indexOf(sorted[i].socketId);
        const idx2 = queue.indexOf(sorted[j].socketId);
        if (idx1 > -1) queue.splice(idx1, 1);
        if (idx2 > -1) queue.splice(idx2, 1);

        const match = new Match(sorted[i], sorted[j]);
        matches.set(match.id, match);
        match.startReadyCheck();
        console.log(`[Matchmaking] Matched: ${sorted[i].name}(${sorted[i].elo}) vs ${sorted[j].name}(${sorted[j].elo})`);
        break;
      }
    }
  }
}

setInterval(tryMatchmaking, 1500);

// ===================== SOCKET HANDLERS =====================
io.on("connection", (socket) => {
  console.log(`[Socket] Connected: ${socket.id}`);

  socket.on("user:register", ({ name, elo }) => {
    const storedElo = parseInt(elo);
    const user = {
      id: uuidv4(),
      name: name?.trim()?.slice(0, 16) || "Anonymous",
      elo: (!isNaN(storedElo) && storedElo >= 100 && storedElo <= 2000) ? storedElo : 800,
      socketId: socket.id,
      inQueue: false,
      inMatch: false,
      queueJoinedAt: null,
    };
    users.set(socket.id, user);
    socket.emit("user:registered", { name: user.name, elo: user.elo });
    console.log(`[User] Registered: ${user.name} (${socket.id}) ELO: ${user.elo}`);
  });

  socket.on("queue:join", () => {
    const user = users.get(socket.id);
    if (!user || user.inQueue || user.inMatch) return;
    user.inQueue = true;
    user.queueJoinedAt = Date.now();
    queue.push(socket.id);
    socket.emit("queue:joined");
  });

  socket.on("queue:leave", () => {
    const user = users.get(socket.id);
    if (!user || !user.inQueue) return;
    user.inQueue = false;
    user.queueJoinedAt = null;
    const idx = queue.indexOf(socket.id);
    if (idx > -1) queue.splice(idx, 1);
    socket.emit("queue:left");
  });

  socket.on("match:ready", ({ matchId, ready }) => {
    const match = matches.get(matchId);
    if (match) match.setReady(socket.id, ready);
  });

  socket.on("match:rep", ({ matchId, count }) => {
    const match = matches.get(matchId);
    if (match) match.recordRep(socket.id, count);
  });

  socket.on("match:forfeit", ({ matchId }) => {
    const match = matches.get(matchId);
    if (match) match.forfeit(socket.id);
  });

  // --- WebRTC Signaling ---
  socket.on("signal:offer", ({ target, offer }) => {
    io.to(target).emit("signal:offer", { sender: socket.id, offer });
  });

  socket.on("signal:answer", ({ target, answer }) => {
    io.to(target).emit("signal:answer", { sender: socket.id, answer });
  });

  socket.on("signal:ice-candidate", ({ target, candidate }) => {
    io.to(target).emit("signal:ice-candidate", { sender: socket.id, candidate });
  });

  socket.on("disconnect", () => {
    const user = users.get(socket.id);
    if (user) {
      for (const [_, match] of matches) {
        if (match.phase !== "ended") {
          if (match.p1.socketId === socket.id || match.p2.socketId === socket.id) {
            match.forfeit(socket.id);
          }
        }
      }
      if (user.inQueue) {
        const idx = queue.indexOf(socket.id);
        if (idx > -1) queue.splice(idx, 1);
      }
      users.delete(socket.id);
    }
  });
});

// ===================== HTTP ENDPOINTS =====================
app.get("/api/leaderboard", (req, res) => {
  const allUsers = Array.from(users.values()).map(u => ({ name: u.name, elo: u.elo, online: true }));
  const seen = new Set(allUsers.map(u => u.name));
  for (const m of matchHistory.slice(-50)) {
    if (!seen.has(m.p1.name)) { allUsers.push({ name: m.p1.name, elo: m.p1.eloAfter, online: false }); seen.add(m.p1.name); }
    if (!seen.has(m.p2.name)) { allUsers.push({ name: m.p2.name, elo: m.p2.eloAfter, online: false }); seen.add(m.p2.name); }
  }
  allUsers.sort((a, b) => b.elo - a.elo);
  res.json(allUsers.slice(0, 20));
});

app.get("/api/history", (req, res) => {
  res.json(matchHistory.slice(-20).reverse());
});

app.get("/api/stats", (req, res) => {
  res.json({
    online: users.size,
    inQueue: queue.length,
    activeMatches: Array.from(matches.values()).filter(m => m.phase === "active").length,
    totalMatches: matchHistory.length,
  });
});

const PORT = process.env.PORT || 3000;

function getLocalIp() {
  const interfaces = os.networkInterfaces();
  for (const name of Object.keys(interfaces)) {
    for (const iface of interfaces[name]) {
      if (iface.family === 'IPv4' && !iface.internal) {
        return iface.address;
      }
    }
  }
  return 'YOUR_IP';
}

server.listen(PORT, '0.0.0.0', () => {
  const localIp = getLocalIp();
  console.log(`[Server] Push-up Arena running on port ${PORT}`);
  console.log(`[Server] Local:    http://localhost:${PORT}`);
  console.log(`[Server] Network:  http://${localIp}:${PORT}`);
});
