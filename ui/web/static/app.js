// DMS V5 SENTINEL - Real-Time Dashboard App

const COLORS = {
    safe: '#10b981',
    mild: '#f59e0b',
    warning: '#f97316',
    critical: '#ef4444'
};

// Initialize Chart.js styling
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', sans-serif";

// Prediction Chart
const predCtx = document.getElementById('predictionChart').getContext('2d');
const predictionChart = new Chart(predCtx, {
    type: 'line',
    data: {
        labels: Array(60).fill(''),
        datasets: [{
            label: 'Fatigue Score',
            data: Array(60).fill(0),
            borderColor: COLORS.safe,
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 0
        }]
    },
    options: {
        responsive: true, maintainAspectRatio: false,
        scales: { y: { min: 0, max: 100, grid: { color: 'rgba(255,255,255,0.05)' } }, x: { grid: { display: false } } },
        plugins: { legend: { display: false } },
        animation: { duration: 0 } // No animation for real-time streaming
    }
});

// Radar Chart
const radarCtx = document.getElementById('radarChart').getContext('2d');
const radarChart = new Chart(radarCtx, {
    type: 'radar',
    data: {
        labels: ['EAR', 'PERCLOS', 'Blinks', 'Sway', 'Yawn', 'Gaze', 'HR'],
        datasets: [{
            label: 'Signal Strength',
            data: [0, 0, 0, 0, 0, 0, 0],
            backgroundColor: 'rgba(59, 130, 246, 0.2)',
            borderColor: '#3b82f6',
            pointBackgroundColor: '#3b82f6',
            borderWidth: 1
        }]
    },
    options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
            r: {
                min: 0, max: 100,
                angleLines: { color: 'rgba(255,255,255,0.1)' },
                grid: { color: 'rgba(255,255,255,0.1)' },
                pointLabels: { color: '#f8fafc', font: { size: 10 } },
                ticks: { display: false }
            }
        },
        plugins: { legend: { display: false } }
    }
});

// WebSocket Connection
const ws = new WebSocket(`ws://${window.location.host}/ws/telemetry`);

ws.onopen = () => console.log('Connected to V5 Telemetry Stream');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    const { topic, payload } = data;

    if (topic === 'FATIGUE_SCORE') {
        updateFatigue(payload);
    } else if (topic === 'SIGNAL_HEAD_POSE') {
        document.getElementById('pitch-val').innerText = payload.pitch.toFixed(1);
        document.getElementById('yaw-val').innerText = payload.yaw.toFixed(1);
    } else if (topic === 'UI_OVERLAY_UPDATE') {
        showCriticalAlert(payload.message);
    }
};

function updateFatigue(payload) {
    // Score & Level
    const score = payload.score.toFixed(1);
    document.getElementById('score-val').innerText = score;
    const badge = document.getElementById('level-badge');
    badge.innerText = payload.level.toUpperCase();
    
    // Theme colors
    let color = COLORS.safe;
    let bg = 'rgba(16, 185, 129, 0.2)';
    if (payload.level === 'mild') { color = COLORS.mild; bg = 'rgba(245, 158, 11, 0.2)'; }
    else if (payload.level === 'warning') { color = COLORS.warning; bg = 'rgba(249, 115, 22, 0.2)'; }
    else if (payload.level === 'critical') { color = COLORS.critical; bg = 'rgba(239, 68, 68, 0.2)'; }
    
    badge.style.color = color;
    badge.style.backgroundColor = bg;
    document.querySelector('.dial').style.borderTopColor = color;

    // Line Chart
    const d = predictionChart.data.datasets[0].data;
    d.shift();
    d.push(payload.score);
    predictionChart.data.datasets[0].borderColor = color;
    predictionChart.data.datasets[0].backgroundColor = bg;
    predictionChart.update();

    document.getElementById('pred-val').innerText = payload.prediction_3min.toFixed(1);

    // Radar Chart
    if (payload.components) {
        const c = payload.components;
        radarChart.data.datasets[0].data = [
            c.ear || 0, c.perclos || 0, c.blink_dynamics || 0, 
            c.head_sway || 0, c.mar || 0, c.gaze_quality || 0, c.rppg || 0
        ];
        radarChart.update();
    }
    
    if (payload.dominant_signal) {
        document.getElementById('dom-signal').innerText = payload.dominant_signal.toUpperCase().replace('_', ' ');
    }
}

let alertTimeout;
function showCriticalAlert(msg) {
    const overlay = document.getElementById('overlay-alert');
    document.getElementById('overlay-msg').innerText = msg;
    overlay.classList.remove('hidden');
    
    clearTimeout(alertTimeout);
    alertTimeout = setTimeout(() => {
        overlay.classList.add('hidden');
    }, 4000);
}
