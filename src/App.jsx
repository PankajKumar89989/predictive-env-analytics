import { useCallback, useEffect, useState } from "react";
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Tooltip, Legend, Filler,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

const AQI = ["", "Good", "Fair", "Moderate", "Poor", "Very poor"];
const DEFAULT = { lat: 28.6139, lon: 77.209 }; // New Delhi, used if location is denied
const hh = (t) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

async function get(path, { lat, lon }) {
  const r = await fetch(`${path}?lat=${lat}&lon=${lon}`);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "Request failed");
  return j;
}

export default function App() {
  const [pos, setPos] = useState(null);
  const [note, setNote] = useState("Detecting your location…");
  const [wx, setWx] = useState(null);
  const [ml, setMl] = useState(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    if (!navigator.geolocation) { setPos(DEFAULT); setNote("Geolocation unavailable, showing New Delhi"); return; }
    navigator.geolocation.getCurrentPosition(
      (p) => { setPos({ lat: p.coords.latitude, lon: p.coords.longitude }); setNote(""); },
      () => { setPos(DEFAULT); setNote("Location blocked, showing New Delhi"); }
    );
  }, []);

  const load = useCallback(async () => {
    if (!pos) return;
    try {
      setErr("");
      const [w, m] = await Promise.all([get("/api/weather", pos), get("/api/predict", pos)]);
      setWx(w); setMl(m);
    } catch (e) { setErr(e.message); }
  }, [pos]);

  useEffect(() => { load(); const id = setInterval(load, 5 * 60 * 1000); return () => clearInterval(id); }, [load]);

  const opts = (unit) => ({
    responsive: true, maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    scales: { y: { title: { display: true, text: unit } }, x: { ticks: { maxTicksLimit: 8 } } },
  });

  let pmData = null;
  if (ml) {
    const h = ml.history, n = h.ts.length, last = h.pm2_5[n - 1];
    const pad = (arr, lead) => [...Array(lead).fill(null), ...arr];
    pmData = {
      labels: [...h.ts, ...ml.future_ts].map(hh),
      datasets: [
        { label: "Measured PM2.5", data: [...h.pm2_5, ...Array(ml.future_ts.length).fill(null)], borderColor: "#1c2a2b", pointRadius: 0, tension: 0.3 },
        { label: "Random Forest", data: pad([last, ...ml.predictions.random_forest], n - 1), borderColor: "#c2410c", borderDash: [6, 4], pointRadius: 0, tension: 0.3 },
        { label: "SVM (SVR)", data: pad([last, ...ml.predictions.svm], n - 1), borderColor: "#5b4bd6", borderDash: [6, 4], pointRadius: 0, tension: 0.3 },
      ],
    };
  }

  const tempData = wx && {
    labels: wx.forecast.map((f) => hh(f.dt)),
    datasets: [
      { label: "Temperature °C", data: wx.forecast.map((f) => f.temp), borderColor: "#0f6b6b", backgroundColor: "rgba(15,107,107,.12)", fill: true, tension: 0.35 },
    ],
  };

  return (
    <main>
      <header>
        <h1>{wx ? `${wx.city}, ${wx.country}` : "Environmental analytics"}</h1>
        <p>{note || (wx ? `${wx.description} · refreshes every 5 minutes` : "Loading live data…")}</p>
      </header>
      {err && <div className="error">{err}</div>}

      {wx && (
        <section className="stats">
          <div><b>{Math.round(wx.temp)}°C</b>Feels like {Math.round(wx.feels_like)}°C</div>
          <div><b>{wx.humidity}%</b>Humidity</div>
          <div><b>{wx.wind} m/s</b>Wind</div>
          <div><b>{wx.pressure} hPa</b>Pressure</div>
          <div><b>{wx.pm2_5.toFixed(1)}</b>PM2.5 µg/m³ · {AQI[wx.aqi]}</div>
        </section>
      )}

      {pmData && (
        <section className="panel">
          <h2>PM2.5 forecast, next 24 hours</h2>
          <div className="chart"><Line data={pmData} options={opts("µg/m³")} /></div>
          <p className="metrics">
            Holdout error (MAE): Random Forest {ml.metrics.random_forest.mae} · SVM {ml.metrics.svm.mae} µg/m³.
            Both models trained on the last 5 days of hourly readings for your location.
          </p>
        </section>
      )}

      {tempData && (
        <section className="panel">
          <h2>Temperature, next 48 hours</h2>
          <div className="chart"><Line data={tempData} options={opts("°C")} /></div>
        </section>
      )}
    </main>
  );
}
