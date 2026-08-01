// ===================== STATE =====================
const state = {
  socket: null,
  screen: 'home',
  user: null,
  match: null,
  phase: 'idle',
  poseDetector: null,
  cameraStream: null,
  aiLoaded: false,
  isRegistered: false,
  repThrottle: null,
  queueStartTime: null,
  botMode: false,
  manualReps: 0,
  connectionAttempts: 0,
  maxConnectionAttempts: 10,
  peerConnection: null,
  remoteStream: null,
  webrtcReady: false,
  readyCheckInterval: null,
  botTimer: null,
  botTimeouts: []
};

// WebRTC ICE Servers Configuration
const rtcConfig = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' }
  ]
};

// ===================== WEBRTC STREAMING =====================
function initPeerConnection() {
  closePeerConnection();

  const pc = new RTCPeerConnection(rtcConfig);
  state.peerConnection = pc;

  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(track => {
      try {
        pc.addTrack(track, state.cameraStream);
      } catch (e) {
        console.warn('[WebRTC] Track add failed:', e);
      }
    });
  }

  pc.ontrack = (event) => {
    console.log('[WebRTC] Received remote track');
    state.remoteStream = event.streams[0];
    const oppVideo = $('compete-opp-video');
    const oppPlaceholder = $('compete-opp-placeholder');
    if (oppVideo && state.remoteStream) {
      oppVideo.srcObject = state.remoteStream;
      oppVideo.play().catch(e => console.warn('[WebRTC] Remote video play error:', e));
    }
    if (oppPlaceholder) oppPlaceholder.style.display = 'none';
  };

  pc.onicecandidate = (event) => {
    if (event.candidate && state.match && state.socket) {
      state.socket.emit('webrtc:signal', {
        matchId: state.match.id,
        signal: { type: 'candidate', candidate: event.candidate }
      });
    }
  };

  pc.onconnectionstatechange = () => {
    console.log('[WebRTC] Connection state:', pc.connectionState);
    if (pc.connectionState === 'connected') {
      state.webrtcReady = true;
    }
  };

  return pc;
}

async function createWebRTCOffer() {
  try {
    if (!state.peerConnection) initPeerConnection();
    const pc = state.peerConnection;

    if (state.cameraStream) {
      const senders = pc.getSenders().map(s => s.track);
      state.cameraStream.getTracks().forEach(track => {
        if (!senders.includes(track)) {
          pc.addTrack(track, state.cameraStream);
        }
      });
    }

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    if (state.socket && state.match) {
      state.socket.emit('webrtc:signal', {
        matchId: state.match.id,
        signal: { type: 'offer', offer }
      });
      console.log('[WebRTC] Offer sent');
    }
  } catch (err) {
    console.error('[WebRTC] Offer creation failed:', err);
  }
}

async function handleWebRTCSignal({ signal }) {
  if (!signal) return;

  if (!state.peerConnection) {
    initPeerConnection();
  }
  const pc = state.peerConnection;

  try {
    if (signal.type === 'offer') {
      console.log('[WebRTC] Received offer');
      await pc.setRemoteDescription(new RTCSessionDescription(signal.offer));

      if (state.cameraStream) {
        const senders = pc.getSenders().map(s => s.track);
        state.cameraStream.getTracks().forEach(track => {
          if (!senders.includes(track)) {
            pc.addTrack(track, state.cameraStream);
          }
        });
      }

      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);

      if (state.socket && state.match) {
        state.socket.emit('webrtc:signal', {
          matchId: state.match.id,
          signal: { type: 'answer', answer }
        });
        console.log('[WebRTC] Answer sent');
      }
    } else if (signal.type === 'answer') {
      console.log('[WebRTC] Received answer');
      await pc.setRemoteDescription(new RTCSessionDescription(signal.answer));
    } else if (signal.type === 'candidate') {
      await pc.addIceCandidate(new RTCIceCandidate(signal.candidate));
    }
  } catch (err) {
    console.error('[WebRTC] Signaling error:', err);
  }
}

function closePeerConnection() {
  if (state.peerConnection) {
    state.peerConnection.close();
    state.peerConnection = null;
  }
  state.remoteStream = null;
  state.webrtcReady = false;
  const oppVideo = $('compete-opp-video');
  const oppPlaceholder = $('compete-opp-placeholder');
  if (oppVideo) oppVideo.srcObject = null;
  if (oppPlaceholder) oppPlaceholder.style.display = 'flex';
}

// ===================== DOM HELPERS =====================
function $(id) { return document.getElementById(id); }
function showScreen(name) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  const el = $('screen-' + name);
  if (el) el.classList.add('active');
  state.screen = name;
  console.log('[Screen]', name);
}

function fmtElo(n) { return (n || 0).toLocaleString(); }

function showError(msg, duration) {
  console.error('[Error]', msg);
  const toast = $('error-toast');
  const text = $('error-text');
  if (!toast || !text) return;
  text.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), duration || 8000);
}

function showLoading(text) {
  const overlay = $('loading-overlay');
  const txt = $('loading-text');
  if (overlay && txt) {
    txt.textContent = text || 'Loading...';
    overlay.classList.add('show');
  }
}

function hideLoading() {
  const overlay = $('loading-overlay');
  if (overlay) overlay.classList.remove('show');
}

function setStatus(text, type) {
  const st = $('status-text');
  const dot = $('status-dot');
  if (st) st.textContent = text;
  if (dot) {
    dot.className = 'status-dot';
    if (type) dot.classList.add(type);
  }
}

function setButtonState(id, text, disabled) {
  const btn = $(id);
  if (!btn) return;
  btn.textContent = text;
  btn.disabled = !!disabled;
}

// ===================== CAMERA =====================
async function startCamera() {
  if (state.cameraStream) {
    try {
      const video = $('sidebar-video');
      if (video && video.srcObject !== state.cameraStream) {
        video.srcObject = state.cameraStream;
        await video.play();
      }
      return true;
    } catch (e) {
      console.warn('[Camera] Reuse failed, requesting fresh');
      stopCamera();
    }
  }

  try {
    console.log('[Camera] Requesting access...');
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });
    state.cameraStream = stream;
    console.log('[Camera] Access granted');

    const video = $('sidebar-video');
    if (video) {
      video.srcObject = stream;
      await video.play();
    }

    const mainYourVideo = $('compete-your-video');
    if (mainYourVideo) {
      mainYourVideo.srcObject = stream;
      await mainYourVideo.play();
    }

    const canvas = $('sidebar-canvas');
    if (canvas && video) {
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
    }

    setStatus('Camera on', 'ready');
    return true;
  } catch (err) {
    console.error('[Camera] Failed:', err.name, err.message);
    showError('Camera blocked. You can still play — tap the screen or press spacebar to count reps.');
    setStatus('Camera blocked', 'error');
    return false;
  }
}

function stopCamera() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(t => t.stop());
    state.cameraStream = null;
  }
  const video = $('sidebar-video');
  if (video) video.srcObject = null;
  const mainYourVideo = $('compete-your-video');
  if (mainYourVideo) mainYourVideo.srcObject = null;
}

// ===================== AI / POSE DETECTOR =====================
async function initPoseDetector() {
  if (state.poseDetector && state.aiLoaded) return true;

  showLoading('Loading AI... (10-20s first time)');

  try {
    state.poseDetector = new PoseDetector();
    const video = $('sidebar-video');
    const canvas = $('sidebar-canvas');
    if (!video || !canvas) throw new Error('Video/canvas not found');

    await state.poseDetector.init(video, canvas);
    state.aiLoaded = true;
    hideLoading();
    setStatus('AI ready', 'ready');
    console.log('[AI] Pose detector loaded');
    return true;
  } catch (err) {
    hideLoading();
    console.error('[AI] Failed:', err);
    showError('AI model failed to load. Playing in manual mode — tap screen or press spacebar.');
    setStatus('Manual mode', 'warning');
    state.aiLoaded = false;
    return false;
  }
}

function startPoseDetection() {
  if (!state.poseDetector || !state.aiLoaded) return;
  state.poseDetector.startDetection((data) => {
    $('sidebar-rep-count').textContent = data.reps;
    const phaseEl = $('sidebar-phase');
    if (phaseEl) {
      phaseEl.textContent = data.phase === 'down' ? '▼ DOWN' : (data.phase === 'up' ? '▲ UP' : '--');
      phaseEl.className = 'sidebar-phase ' + data.phase;
    }

    // Update coaching indicator
    updateCoachingIndicator(data.phase, data.depth);

    if (state.phase === 'active' && state.match) {
      sendRepsToServer(data.reps);
      const yourRepsEl = $('compete-your-reps');
      if (yourRepsEl) yourRepsEl.textContent = data.reps;
      updateCompeteProgress();
    }

    // Auto-ready detection
    if (state.phase === 'ready_check' && data.ready && state.socket && state.match) {
      const dot = $('ready-dot-you');
      if (dot && !dot.classList.contains('ready')) {
        console.log('[Ready] Auto-detected position');
        state.socket.emit('match:ready', { matchId: state.match.id, ready: true });
      }
    }
  });
}

// ===================== COACHING INDICATOR =====================
function updateCoachingIndicator(phase, depth) {
  const coachEl = $('coaching-indicator');
  const depthBar = $('depth-bar-fill');
  const depthLabel = $('depth-label');

  if (!coachEl) return;

  if (phase === 'up') {
    coachEl.textContent = '▼ PUSH DOWN';
    coachEl.className = 'coaching-indicator coach-down';
  } else if (phase === 'down') {
    coachEl.textContent = '▲ PUSH UP';
    coachEl.className = 'coaching-indicator coach-up';
  } else {
    coachEl.textContent = '';
    coachEl.className = 'coaching-indicator';
  }

  if (depthBar) {
    const pct = Math.round((depth || 0.5) * 100);
    depthBar.style.width = pct + '%';
    depthBar.className = 'depth-bar-fill ' + (phase === 'down' ? 'depth-down' : phase === 'up' ? 'depth-up' : '');
  }

  if (depthLabel) {
    depthLabel.textContent = phase === 'down' ? 'BOTTOM' : phase === 'up' ? 'TOP' : '—';
  }
}

// ===================== SOCKET SETUP =====================
function connect() {
  if (state.socket) {
    state.socket.removeAllListeners();
    state.socket.close();
  }

  console.log('[Socket] Connecting... attempt', state.connectionAttempts + 1);
  setStatus('Connecting...', 'warning');
  setButtonState('btn-match', 'Connecting...', true);

  const socket = io({
    transports: ['websocket', 'polling'],
    reconnection: true,
    reconnectionAttempts: state.maxConnectionAttempts,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
    timeout: 10000,
  });

  state.socket = socket;

  const connectionTimeout = setTimeout(() => {
    if (!state.isRegistered) {
      console.warn('[Socket] Connection timeout');
      setStatus('Server unreachable', 'error');
      setButtonState('btn-match', 'Retry connection', false);
      showError('Cannot reach server. Make sure "npm start" is running.');
    }
  }, 8000);

  socket.on('connect', () => {
    console.log('[Socket] Connected:', socket.id);
    clearTimeout(connectionTimeout);
    setStatus('Connected', 'ready');
    state.connectionAttempts = 0;

    const name = localStorage.getItem('arena_name');
    const storedElo = localStorage.getItem('arena_elo');
    if (name && !state.isRegistered) {
      console.log('[Socket] Registering as', name, storedElo ? `(ELO: ${storedElo})` : '');
      socket.emit('user:register', { name, elo: storedElo ? parseInt(storedElo) : undefined });
    }
  });

  socket.on('disconnect', (reason) => {
    console.log('[Socket] Disconnected:', reason);
    state.isRegistered = false;
    closePeerConnection();
    if (state.phase !== 'ended' && state.phase !== 'idle') {
      state.phase = 'idle';
      showScreen('home');
      showError('Connection lost. Reconnecting...');
    }
    setStatus('Disconnected', 'error');
    setButtonState('btn-match', 'Reconnecting...', true);
  });

  socket.on('connect_error', (err) => {
    state.connectionAttempts++;
    console.error('[Socket] Connection error:', err.message);
    setStatus('Connection failed', 'error');
    setButtonState('btn-match', 'Retry connection', false);
    if (state.connectionAttempts >= state.maxConnectionAttempts) {
      showError('Server unreachable after ' + state.maxConnectionAttempts + ' tries. Is "npm start" running?');
    }
  });

  socket.on('user:registered', ({ name, elo }) => {
    console.log('[Socket] Registered:', name, elo);
    state.user = { name, elo };
    state.isRegistered = true;
    localStorage.setItem('arena_name', name);
    localStorage.setItem('arena_elo', elo);

    const sn = $('sidebar-name');
    const se = $('sidebar-elo');
    const sa = $('sidebar-avatar');
    const he = $('home-elo');
    const cyn = $('compete-your-name');

    if (sn) sn.textContent = name;
    if (se) se.textContent = elo + ' ELO';
    if (sa) sa.textContent = name.slice(0, 2).toUpperCase();
    if (he) he.textContent = fmtElo(elo);
    if (cyn) cyn.textContent = name;

    setButtonState('btn-match', 'Start matching', false);
    setStatus('Ready to play', 'ready');
    showScreen('home');
    fetchLeaderboard();
    fetchStats();
  });

  socket.on('queue:joined', () => {
    console.log('[Queue] Joined');
    state.phase = 'queue';
    state.queueStartTime = Date.now();
    showScreen('matching');
    startQueueAnimation();
  });

  socket.on('queue:left', () => {
    console.log('[Queue] Left');
    state.phase = 'idle';
    state.queueStartTime = null;
    showScreen('home');
  });

  socket.on('match:ready_check', ({ matchId, opponent, youAre }) => {
    console.log('[Match] Ready check vs', opponent.name);
    state.match = { id: matchId, opponent, youAre };
    state.phase = 'ready_check';
    state.botMode = false;

    const ron = $('ready-opp-name');
    const ron2 = $('ready-opp-name2');
    const con = $('compete-opp-name');
    const cor = $('compete-opp-reps');
    const vLabel = $('video-opp-name');

    if (ron) ron.textContent = opponent.name;
    if (ron2) ron2.textContent = opponent.name;
    if (con) con.textContent = opponent.name;
    if (vLabel) vLabel.textContent = opponent.name;
    if (cor) cor.textContent = '0';

    if (state.poseDetector) state.poseDetector.resetReps();
    state.manualReps = 0;

    initPeerConnection();

    if (youAre === 'p1') {
      setTimeout(() => {
        if (state.phase === 'ready_check') {
          createWebRTCOffer();
        }
      }, 800);
    }

    showScreen('ready');
  });

  socket.on('webrtc:signal', (data) => {
    handleWebRTCSignal(data);
  });

  socket.on('match:ready_update', ({ p1Ready, p2Ready, p1Name, p2Name }) => {
    const isP1 = state.match?.youAre === 'p1';
    const youReady = isP1 ? p1Ready : p2Ready;
    const oppReady = isP1 ? p2Ready : p1Ready;

    const rdy = $('ready-dot-you');
    const rdo = $('ready-dot-opp');
    const rs = $('ready-status');
    const brc = $('btn-ready-confirm');

    if (rdy) rdy.classList.toggle('ready', youReady);
    if (rdo) rdo.classList.toggle('ready', oppReady);

    if (youReady && oppReady) {
      if (rs) { rs.textContent = 'Both players ready!'; rs.classList.add('both-ready'); }
      if (brc) { brc.textContent = 'Ready!'; brc.disabled = true; }
    } else if (youReady) {
      if (rs) rs.textContent = 'Waiting for opponent...';
      if (brc) { brc.textContent = 'Ready!'; brc.disabled = true; }
    } else {
      if (rs) rs.textContent = state.aiLoaded ? 'Get into push-up position...' : 'Click ready when set...';
      if (brc) {
        brc.textContent = state.aiLoaded ? 'Waiting for position...' : 'I am ready';
        brc.disabled = state.aiLoaded;
      }
    }
  });

  socket.on('match:countdown_start', ({ seconds }) => {
    console.log('[Match] Countdown start', seconds);
    state.phase = 'countdown';
    const rs = $('ready-status');
    if (rs) rs.textContent = 'Starting in ' + seconds + '...';
  });

  socket.on('match:countdown_tick', ({ timeLeft }) => {
    const rs = $('ready-status');
    if (rs) rs.textContent = 'Starting in ' + timeLeft + '...';
  });

  socket.on('match:start', ({ duration }) => {
    console.log('[Match] Start!', duration);
    state.phase = 'active';
    showScreen('compete');
    const ct = $('compete-timer');
    const cpd = $('compete-phase-display');
    const cyr = $('compete-your-reps');
    if (ct) ct.textContent = duration;
    if (cpd) cpd.textContent = 'GO!';
    if (cyr) cyr.textContent = state.poseDetector?.repCount || state.manualReps || 0;
  });

  socket.on('match:tick', ({ p1Reps, p2Reps, timeLeft }) => {
    if (state.phase !== 'active') return;
    const ct = $('compete-timer');
    if (ct) ct.textContent = timeLeft;

    const isP1 = state.match?.youAre === 'p1';
    const oppReps = isP1 ? p2Reps : p1Reps;
    const myReps = state.poseDetector?.repCount || state.manualReps || 0;

    const cor = $('compete-opp-reps');
    if (cor) cor.textContent = oppReps;
    updateCompeteProgress(myReps, oppReps);
  });

  socket.on('match:end', ({ yourReps, oppReps, yourEloChange, newElo, result, diff, opponentName, reason }) => {
    console.log('[Match] End', result, yourReps, 'vs', oppReps);
    state.phase = 'ended';
    closePeerConnection();
    if (state.user) state.user.elo = newElo;
    localStorage.setItem('arena_elo', newElo);

    const ry = $('res-your');
    const ro = $('res-opp');
    const rd = $('res-diff');
    const rne = $('res-new-elo');
    const he = $('home-elo');
    const se = $('sidebar-elo');
    const rec = $('res-elo-change');
    const ri = $('result-icon');
    const rt = $('result-title');
    const rs = $('result-sub');

    if (ry) ry.textContent = yourReps;
    if (ro) ro.textContent = oppReps;
    if (rd) rd.textContent = (diff >= 0 ? '+' : '') + diff;
    if (rne) rne.textContent = fmtElo(newElo);
    if (he) he.textContent = fmtElo(newElo);
    if (se) se.textContent = newElo + ' ELO';

    if (rec) {
      rec.textContent = (yourEloChange >= 0 ? '+' : '') + yourEloChange;
      rec.className = yourEloChange > 0 ? 'elo-positive' : (yourEloChange < 0 ? 'elo-negative' : '');
    }

    if (result === 'win') {
      if (ri) ri.textContent = '🏆';
      if (rt) rt.textContent = reason === 'forfeit' ? 'Win by forfeit' : 'Victory!';
      if (rs) rs.textContent = reason === 'forfeit' ? opponentName + ' left' : 'You out-repped ' + opponentName + ' by ' + Math.abs(diff);
    } else if (result === 'loss') {
      if (ri) ri.textContent = '💪';
      if (rt) rt.textContent = reason === 'forfeit' ? 'Forfeited' : 'Defeat';
      if (rs) rs.textContent = reason === 'forfeit' ? 'You left' : opponentName + ' out-repped you by ' + Math.abs(diff);
    } else {
      if (ri) ri.textContent = '🤝';
      if (rt) rt.textContent = 'Tie!';
      if (rs) rs.textContent = 'Perfectly matched';
    }

    showScreen('result');
    fetchLeaderboard();
    fetchStats();
  });
}

// ===================== BOT MATCH =====================
function startBotMatch() {
  console.log('[Bot] Starting bot match');
  state.botMode = true;
  state.phase = 'ready_check';

  const botNames = ['IronChest', 'PushKing', 'RepMaster', 'SwolePatrol', 'GymRat', 'BeastMode'];
  const botName = botNames[Math.floor(Math.random() * botNames.length)];
  const botElo = 800 + Math.floor(Math.random() * 200 - 100);

  state.match = {
    id: 'bot-' + Date.now(),
    opponent: { name: botName, elo: botElo },
    youAre: 'p1',
  };

  const ron = $('ready-opp-name');
  const ron2 = $('ready-opp-name2');
  const con = $('compete-opp-name');
  const cor = $('compete-opp-reps');
  const vLabel = $('video-opp-name');

  if (ron) ron.textContent = botName;
  if (ron2) ron2.textContent = botName;
  if (con) con.textContent = botName;
  if (vLabel) vLabel.textContent = botName;
  if (cor) cor.textContent = '0';

  if (state.poseDetector) state.poseDetector.resetReps();
  state.manualReps = 0;

  showScreen('ready');

  setTimeout(() => {
    const rdo = $('ready-dot-opp');
    const rs = $('ready-status');
    if (rdo) rdo.classList.add('ready');
    if (rs) rs.textContent = 'Opponent ready!';

    setTimeout(() => {
      const rdy = $('ready-dot-you');
      if (rdy) rdy.classList.add('ready');
      if (rs) { rs.textContent = 'Both ready! Starting...'; rs.classList.add('both-ready'); }

      let count = 5;
      const countdownInterval = setInterval(() => {
        count--;
        if (rs) rs.textContent = 'Starting in ' + count + '...';
        if (count <= 0) {
          clearInterval(countdownInterval);
          startBotRound(botElo);
        }
      }, 1000);
    }, 1500);
  }, 1000);
}

function startBotRound(botElo) {
  state.phase = 'active';
  showScreen('compete');
  const ct = $('compete-timer');
  const cpd = $('compete-phase-display');
  if (ct) ct.textContent = '60';
  if (cpd) cpd.textContent = 'GO!';

  let timeLeft = 60;
  let botReps = 0;
  const botSkill = 0.5 + Math.random() * 1.5;
  const targetReps = Math.floor(15 + botSkill * 25); // 15-50 target reps

  // Clear any existing bot timeouts
  state.botTimeouts.forEach(t => clearTimeout(t));
  state.botTimeouts = [];

  // Game timer
  state.botTimer = setInterval(() => {
    timeLeft--;
    const ct2 = $('compete-timer');
    if (ct2) ct2.textContent = timeLeft;

    const myReps = state.poseDetector?.repCount || state.manualReps || 0;
    updateCompeteProgress(myReps, botReps);

    if (timeLeft <= 0) {
      clearInterval(state.botTimer);
      state.botTimeouts.forEach(t => clearTimeout(t));
      state.botTimeouts = [];
      endBotMatch(myReps, botReps, botElo);
    }
  }, 1000);

  // Bot rep scheduler — decoupled from user's actions
  // Distribute bot reps across the match duration with realistic gaps
  const avgGap = 60000 / (targetReps + 1);

  function scheduleNextBotRep() {
    if (timeLeft <= 0 || botReps >= targetReps) return;
    const jitter = avgGap * 0.35;
    const delay = Math.max(800, avgGap - jitter + Math.random() * jitter * 2);

    const timeout = setTimeout(() => {
      if (timeLeft > 0) {
        botReps++;
        const cor = $('compete-opp-reps');
        if (cor) cor.textContent = botReps;

        const myReps = state.poseDetector?.repCount || state.manualReps || 0;
        updateCompeteProgress(myReps, botReps);
      }
      scheduleNextBotRep();
    }, delay);

    state.botTimeouts.push(timeout);
  }

  // Start bot reps after a small initial delay
  const initialDelay = setTimeout(scheduleNextBotRep, 1500);
  state.botTimeouts.push(initialDelay);
}

function endBotMatch(myReps, botReps, botElo) {
  state.phase = 'ended';
  closePeerConnection();
  const diff = myReps - botReps;
  let result;
  if (diff > 0) result = 'win';
  else if (diff < 0) result = 'loss';
  else result = 'tie';

  // Bot matches NEVER affect ELO
  const currentElo = state.user?.elo || parseInt(localStorage.getItem('arena_elo')) || 800;

  const ry = $('res-your');
  const ro = $('res-opp');
  const rd = $('res-diff');
  const rne = $('res-new-elo');
  const he = $('home-elo');
  const se = $('sidebar-elo');
  const rec = $('res-elo-change');
  const ri = $('result-icon');
  const rt = $('result-title');
  const rs = $('result-sub');

  if (ry) ry.textContent = myReps;
  if (ro) ro.textContent = botReps;
  if (rd) rd.textContent = (diff >= 0 ? '+' : '') + diff;
  if (rne) rne.textContent = fmtElo(currentElo);
  if (he) he.textContent = fmtElo(currentElo);
  if (se) se.textContent = currentElo + ' ELO';

  if (rec) {
    rec.textContent = '—';
    rec.className = '';
  }

  if (result === 'win') {
    if (ri) ri.textContent = '🏆';
    if (rt) rt.textContent = 'Victory!';
    if (rs) rs.textContent = 'You out-repped ' + state.match.opponent.name + ' by ' + Math.abs(diff) + ' (Practice — No ELO change)';
  } else if (result === 'loss') {
    if (ri) ri.textContent = '💪';
    if (rt) rt.textContent = 'Defeat';
    if (rs) rs.textContent = state.match.opponent.name + ' out-repped you by ' + Math.abs(diff) + ' (Practice — No ELO change)';
  } else {
    if (ri) ri.textContent = '🤝';
    if (rt) rt.textContent = 'Tie!';
    if (rs) rs.textContent = 'Perfectly matched (Practice — No ELO change)';
  }

  showScreen('result');
}

// ===================== UI HELPERS =====================
function updateCompeteProgress(you, opp) {
  you = you || 0;
  opp = opp || 0;
  const total = Math.max(you + opp, 1);
  const youPct = (you / total) * 100;
  const oppPct = (opp / total) * 100;
  const py = $('compete-progress-you');
  const po = $('compete-progress-opp');
  if (py) py.style.width = youPct + '%';
  if (po) po.style.width = oppPct + '%';
}

function sendRepsToServer(count) {
  if (state.botMode) return;
  if (!state.match || !state.socket || state.phase !== 'active') return;
  if (!state.repThrottle) {
    state.repThrottle = setTimeout(() => { state.repThrottle = null; }, 150);
    state.socket.emit('match:rep', { matchId: state.match.id, count });
  }
}

let queueAnimInterval = null;
function startQueueAnimation() {
  if (queueAnimInterval) clearInterval(queueAnimInterval);
  let elapsed = 0;
  queueAnimInterval = setInterval(() => {
    if (state.phase !== 'queue') { clearInterval(queueAnimInterval); return; }
    elapsed += 1;
    const tolerance = 50 + Math.floor(elapsed / 2) * 25;
    const mr = $('match-range');
    if (mr) mr.textContent = '±' + Math.min(tolerance, 300);
    const mpb = $('match-progress-bar');
    if (mpb) mpb.style.width = Math.min((elapsed / 30) * 100, 100) + '%';
  }, 1000);
}

// ===================== API =====================
async function fetchLeaderboard() {
  try {
    const res = await fetch('/api/leaderboard');
    const data = await res.json();
    renderLeaderboard(data);
  } catch (e) { console.error('[API] Leaderboard error', e); }
}

async function fetchStats() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    const so = $('stat-online');
    const sq = $('stat-queue');
    const sa = $('stat-active');
    const st = $('stat-total');
    const qc = $('queue-count');
    const am = $('active-matches');
    if (so) so.textContent = data.online;
    if (sq) sq.textContent = data.inQueue;
    if (sa) sa.textContent = data.activeMatches;
    if (st) st.textContent = data.totalMatches;
    if (qc) qc.textContent = data.inQueue + ' in queue';
    if (am) am.textContent = data.activeMatches + ' active';
  } catch (e) { console.error('[API] Stats error', e); }
}

function renderLeaderboard(players) {
  const list = $('leaderboard-list');
  if (!list) return;
  if (!players || !players.length) {
    list.innerHTML = '<div class="leader-row"><span style="color:var(--text-tertiary);font-size:13px;">No players yet</span></div>';
    return;
  }
  list.innerHTML = players.map((p, i) => {
    const isYou = p.name === (state.user?.name);
    const colors = ['#3b82f6', '#dc2626', '#22c55e', '#a855f7'];
    const bg = i === 0 ? 'var(--text-primary)' : colors[i % colors.length];
    return `
      <div class="leader-row">
        <div class="rank ${i < 3 ? 'top' : ''}">${i + 1}</div>
        <div class="avatar" style="background:${bg};color:#fff;font-size:11px;">${p.name.slice(0,2).toUpperCase()}</div>
        <div class="leader-info">
          <div class="leader-name">${p.name} ${isYou ? '<span style="color:var(--text-tertiary);font-size:11px;">(you)</span>' : ''}</div>
          <div class="leader-status">${p.online ? 'Online' : 'Offline'}</div>
        </div>
        <div class="leader-elo">${fmtElo(p.elo)}</div>
      </div>
    `;
  }).join('');
  const lc = $('leaderboard-count');
  if (lc) lc.textContent = 'Top ' + players.length;
}

// ===================== EVENT LISTENERS =====================
function bindEvents() {
  const bm = $('btn-match');
  if (bm) bm.addEventListener('click', () => {
    if (!state.isRegistered) {
      if (state.socket && state.socket.disconnected) {
        connect();
        return;
      }
      showError('Not connected yet. Please wait...');
      return;
    }
    console.log('[Match] Joining queue');
    if (state.socket) state.socket.emit('queue:join');
  });

  const bb = $('btn-bot');
  if (bb) bb.addEventListener('click', () => {
    if (!state.isRegistered) {
      showError('Not connected yet. Please wait...');
      return;
    }
    startBotMatch();
  });

  const bc = $('btn-cancel');
  if (bc) bc.addEventListener('click', () => {
    console.log('[Queue] Cancelling');
    if (state.socket) state.socket.emit('queue:leave');
  });

  const brc = $('btn-ready-confirm');
  if (brc) brc.addEventListener('click', () => {
    if (state.botMode) return;
    if (state.match && state.socket) {
      console.log('[Ready] Manual confirm');
      state.socket.emit('match:ready', { matchId: state.match.id, ready: true });
      brc.textContent = 'Ready!';
      brc.disabled = true;
    }
  });

  const brk = $('btn-ready-cancel');
  if (brk) brk.addEventListener('click', () => {
    if (state.botMode) {
      state.phase = 'idle';
      showScreen('home');
      return;
    }
    if (state.match && state.socket) {
      state.socket.emit('match:forfeit', { matchId: state.match.id });
    }
    closePeerConnection();
    state.phase = 'idle';
    showScreen('home');
  });

  const bf = $('btn-forfeit');
  if (bf) bf.addEventListener('click', () => {
    if (state.botMode) {
      if (state.botTimer) clearInterval(state.botTimer);
      state.botTimeouts.forEach(t => clearTimeout(t));
      state.botTimeouts = [];
      state.phase = 'idle';
      showScreen('home');
      return;
    }
    if (state.match && state.socket) {
      state.socket.emit('match:forfeit', { matchId: state.match.id });
    }
    closePeerConnection();
  });

  const ba = $('btn-again');
  if (ba) ba.addEventListener('click', () => {
    closePeerConnection();
    state.phase = 'idle';
    showScreen('home');
    fetchLeaderboard();
    fetchStats();
  });

  const ec = $('error-close');
  if (ec) ec.addEventListener('click', () => {
    const toast = $('error-toast');
    if (toast) toast.classList.remove('show');
  });
}

// Manual rep fallback — keyboard
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && state.phase === 'active') {
    e.preventDefault();
    addManualRep();
  }
});

// Manual rep fallback — touch
document.addEventListener('touchstart', (e) => {
  if (state.phase === 'active') {
    addManualRep();
  }
}, { passive: true });

function addManualRep() {
  state.manualReps = (state.manualReps || 0) + 1;
  const reps = state.manualReps;

  const src = $('sidebar-rep-count');
  const cyr = $('compete-your-reps');
  if (src) src.textContent = reps;
  if (cyr) cyr.textContent = reps;

  sendRepsToServer(reps);

  const cor = $('compete-opp-reps');
  const oppReps = cor ? parseInt(cor.textContent) || 0 : 0;
  updateCompeteProgress(reps, oppReps);
}

// ===================== INIT =====================
(async function init() {
  console.log('[Init] Push-up Arena starting...');
  bindEvents();

  const name = localStorage.getItem('arena_name');
  if (!name) {
    window.location.href = '/';
    return;
  }

  const camOk = await startCamera();
  if (camOk) {
    initPoseDetector().then((aiOk) => {
      if (aiOk) startPoseDetection();
    });
  }

  connect();

  setInterval(fetchStats, 5000);
  fetchStats();
})();
