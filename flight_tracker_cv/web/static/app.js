/**
 * FlightTracker CV — Frontend JavaScript
 * Handles: socket.io connection, Leaflet map, detection log, stats, Chart.js
 */

'use strict';

// ---------------------------------------------------------------------------
// Clock
// ---------------------------------------------------------------------------
function updateClock() {
  const now = new Date();
  document.getElementById('clock').textContent =
    now.toTimeString().slice(0, 8);
}
setInterval(updateClock, 1000);
updateClock();

// ---------------------------------------------------------------------------
// Socket.io
// ---------------------------------------------------------------------------
const socket = io({ transports: ['websocket', 'polling'] });
const connStatus = document.getElementById('conn-status');
const connLabel  = document.getElementById('conn-label');

socket.on('connect', () => {
  connStatus.classList.add('connected');
  connLabel.textContent = 'Connected';
});
socket.on('disconnect', () => {
  connStatus.classList.remove('connected');
  connLabel.textContent = 'Disconnected';
});

// Live frame
const liveFrame = document.getElementById('live-frame');
let lastFrameTs = 0;
let frameTimes = [];
socket.on('frame', ({ data }) => {
  liveFrame.src = 'data:image/jpeg;base64,' + data;
  const now = performance.now();
  frameTimes.push(now);
  frameTimes = frameTimes.filter(t => now - t < 2000);
  const fps = frameTimes.length / 2;
  document.getElementById('fps-label').textContent = fps.toFixed(1) + ' FPS';
});

// Detection event
let logCount = 0;
socket.on('detection', (det) => {
  addLogEntry(det);
  logCount++;
  document.getElementById('log-count').textContent = logCount + ' events';
  document.getElementById('ac-badge').textContent =
    'AC: ' + (det.active_count || '?');
  pushHourlyDataPoint();
});

// Flight matched event
socket.on('flight_matched', (data) => {
  addLogEntry(data, true);
  updateMapFlight(data);
  // Flash the video panel border
  const panel = document.getElementById('video-panel');
  panel.classList.add('matched-flash');
  setTimeout(() => panel.classList.remove('matched-flash'), 2500);
});

// ---------------------------------------------------------------------------
// Detection log
// ---------------------------------------------------------------------------
function addLogEntry(data, matched = false) {
  const log = document.getElementById('detection-log');
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  if (matched) entry.style.borderLeft = '3px solid var(--green)';

  const ts = new Date().toTimeString().slice(0, 8);
  const callsign = data.callsign || data.aircraft_type || 'UNKNOWN';
  const airline = data.airline || data.airline_name || '';
  const route = (data.origin_iata && data.dest_iata)
    ? `${data.origin_iata} → ${data.dest_iata}` : '';
  const typeStr = data.aircraft_type || data.variant || '';

  entry.innerHTML = `
    <span class="callsign">${escHtml(callsign)}</span>
    ${airline ? `<span style="color:var(--text)"> ${escHtml(airline)}</span>` : ''}
    ${typeStr ? `<span class="type"> [${escHtml(typeStr)}]</span>` : ''}
    ${route ? `<span class="route"> ${escHtml(route)}</span>` : ''}
    <span class="time">${ts}</span>
  `;

  log.insertBefore(entry, log.firstChild);

  // Keep log at most 200 entries
  while (log.children.length > 200) {
    log.removeChild(log.lastChild);
  }
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ---------------------------------------------------------------------------
// Stats auto-refresh
// ---------------------------------------------------------------------------
function refreshStats() {
  fetch('/api/stats')
    .then(r => r.json())
    .then(data => {
      document.getElementById('stat-today').textContent = data.today_count ?? 0;
      document.getElementById('stat-total').textContent = data.total_detections ?? 0;
      document.getElementById('stat-airlines').textContent = data.airlines_seen ?? 0;
      document.getElementById('stat-types').textContent = data.types_seen ?? 0;
      document.getElementById('stat-flights').textContent = data.unique_flights ?? 0;
    })
    .catch(() => {});
}
refreshStats();
setInterval(refreshStats, 5000);

// ---------------------------------------------------------------------------
// Leaflet map
// ---------------------------------------------------------------------------
const defaultLat = 1.3521;
const defaultLon = 103.8198;

const map = L.map('map', { zoomControl: true, attributionControl: false })
  .setView([defaultLat, defaultLon], 9);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© OpenStreetMap contributors',
  maxZoom: 18,
}).addTo(map);

// User location marker
const userIcon = L.divIcon({
  html: '<div style="width:12px;height:12px;background:#00e5ff;border-radius:50%;border:2px solid white;"></div>',
  iconSize: [12, 12],
  iconAnchor: [6, 6],
  className: '',
});
L.marker([defaultLat, defaultLon], { icon: userIcon })
  .addTo(map)
  .bindPopup('Camera location');

// Plane icon factory (rotated SVG)
function planeIcon(heading) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"
    style="transform:rotate(${heading}deg)">
    <path d="M21 16l-9-4.06V4a2 2 0 0 0-4 0v7.94L-1 16l3 1 6-2.12V20l-2 1.5V23l4-1 4 1v-1.5L12 20v-5.12l6 2.12 3-1z"
      fill="#00e5ff" opacity="0.9"/>
  </svg>`;
  return L.divIcon({
    html: svg,
    iconSize: [24, 24],
    iconAnchor: [12, 12],
    className: '',
  });
}

const flightMarkers = {};

function updateMapFlight(flight) {
  if (!flight.lat || !flight.lon) return;
  const id = flight.flight_id || flight.callsign;
  if (!id) return;

  const label = `${flight.callsign || ''} | ${flight.aircraft_type || ''}<br/>
    ${flight.origin_iata || '?'} → ${flight.dest_iata || '?'}<br/>
    Alt: ${Math.round(flight.altitude_ft || 0).toLocaleString()} ft`;

  if (flightMarkers[id]) {
    flightMarkers[id].setLatLng([flight.lat, flight.lon]);
    flightMarkers[id].setIcon(planeIcon(flight.heading_deg || 0));
    flightMarkers[id].getPopup().setContent(label);
  } else {
    flightMarkers[id] = L.marker([flight.lat, flight.lon], {
      icon: planeIcon(flight.heading_deg || 0),
    }).addTo(map).bindPopup(label);
  }
}

// Refresh map flights every 10 s
function refreshMapFlights() {
  fetch('/api/flights')
    .then(r => r.json())
    .then(flights => {
      flights.forEach(f => updateMapFlight(f));
    })
    .catch(() => {});
}
refreshMapFlights();
setInterval(refreshMapFlights, 10000);

// ---------------------------------------------------------------------------
// Chart.js — detections per hour
// ---------------------------------------------------------------------------
const hourLabels = Array.from({ length: 24 }, (_, i) => `${i}:00`);
const hourData = new Array(24).fill(0);

const chartCtx = document.getElementById('hourly-chart').getContext('2d');
const hourlyChart = new Chart(chartCtx, {
  type: 'line',
  data: {
    labels: hourLabels,
    datasets: [{
      label: 'Detections',
      data: hourData,
      borderColor: '#00e5ff',
      backgroundColor: 'rgba(0,229,255,0.12)',
      borderWidth: 1.5,
      pointRadius: 2,
      fill: true,
      tension: 0.3,
    }],
  },
  options: {
    responsive: true,
    animation: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: '#5e7a94', font: { size: 9 } }, grid: { color: '#0a1929' } },
      y: { ticks: { color: '#5e7a94', font: { size: 9 } }, grid: { color: '#1a3a5c' }, beginAtZero: true },
    },
  },
});

function pushHourlyDataPoint() {
  const hour = new Date().getHours();
  hourData[hour] = (hourData[hour] || 0) + 1;
  hourlyChart.update('none');
}

// ---------------------------------------------------------------------------
// Config form handlers
// ---------------------------------------------------------------------------
function sendConfig(key, value) {
  fetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ [key]: value }),
  }).catch(() => {});
}

document.getElementById('cfg-sound').addEventListener('change', e => {
  sendConfig('sound_enabled', e.target.checked);
});
document.getElementById('cfg-tts').addEventListener('change', e => {
  sendConfig('tts_enabled', e.target.checked);
});
document.getElementById('cfg-screenshot').addEventListener('change', e => {
  sendConfig('screenshot_enabled', e.target.checked);
});
document.getElementById('cfg-trails').addEventListener('change', e => {
  sendConfig('show_trails', e.target.checked);
});

const confSlider = document.getElementById('cfg-confidence');
const confVal = document.getElementById('conf-val');
confSlider.addEventListener('input', () => {
  const v = confSlider.value / 100;
  confVal.textContent = v.toFixed(2);
});
confSlider.addEventListener('change', () => {
  sendConfig('confidence_threshold', confSlider.value / 100);
});
