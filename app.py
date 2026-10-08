import os, time
import numpy as np
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

load_dotenv()
KEY = os.getenv("OPENWEATHER_API_KEY")
BASE = "https://api.openweathermap.org"
LAGS, HORIZON = 3, 24

app = Flask(__name__)
CORS(app)


def ow(path, **params):
    """Call the OpenWeather API and return JSON."""
    params["appid"] = KEY
    r = requests.get(BASE + path, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def coords():
    try:
        return float(request.args["lat"]), float(request.args["lon"])
    except (KeyError, ValueError):
        raise ValueError("lat and lon query parameters are required")


@app.errorhandler(Exception)
def handle(e):
    code = e.response.status_code if isinstance(e, requests.HTTPError) else 400
    return jsonify(error=str(e)), code


@app.get("/api/weather")
def weather():
    lat, lon = coords()
    cur = ow("/data/2.5/weather", lat=lat, lon=lon, units="metric")
    fc = ow("/data/2.5/forecast", lat=lat, lon=lon, units="metric", cnt=16)
    air = ow("/data/2.5/air_pollution", lat=lat, lon=lon)["list"][0]
    return jsonify(
        city=cur["name"],
        country=cur["sys"]["country"],
        temp=cur["main"]["temp"],
        feels_like=cur["main"]["feels_like"],
        humidity=cur["main"]["humidity"],
        pressure=cur["main"]["pressure"],
        wind=cur["wind"]["speed"],
        description=cur["weather"][0]["description"],
        aqi=air["main"]["aqi"],
        pm2_5=air["components"]["pm2_5"],
        forecast=[
            {"dt": f["dt"], "temp": f["main"]["temp"], "humidity": f["main"]["humidity"]}
            for f in fc["list"]
        ],
    )


def features(window, ts):
    hour = time.gmtime(ts).tm_hour
    ang = 2 * np.pi * hour / 24
    return list(window) + [np.sin(ang), np.cos(ang)]


def make_models():
    return {
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "svm": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100, epsilon=1.0)),
    }


@app.get("/api/predict")
def predict():
    """Train RF + SVR on 5 days of hourly PM2.5, forecast the next 24 hours."""
    lat, lon = coords()
    end = int(time.time())
    hist = ow("/data/2.5/air_pollution/history", lat=lat, lon=lon,
              start=end - 5 * 86400, end=end)["list"]
    ts = [h["dt"] for h in hist]
    vals = np.array([h["components"]["pm2_5"] for h in hist], dtype=float)
    if len(vals) < 48:
        raise ValueError("Not enough history for this location to train models")

    X = np.array([features(vals[i - LAGS:i][::-1], ts[i]) for i in range(LAGS, len(vals))])
    y = vals[LAGS:]
    split = int(len(X) * 0.8)  # chronological split, no shuffling

    models, metrics = make_models(), {}
    for name, m in models.items():
        m.fit(X[:split], y[:split])
        metrics[name] = {
            "mae": round(float(mean_absolute_error(y[split:], m.predict(X[split:]))), 2),
            "train_rows": split,
            "test_rows": len(X) - split,
        }
        m.fit(X, y)  # refit on everything before forecasting

    preds, future_ts = {}, [ts[-1] + 3600 * (i + 1) for i in range(HORIZON)]
    for name, m in models.items():
        window, out = list(vals[-LAGS:]), []
        for t in future_ts:  # recursive multi-step forecast
            p = max(0.0, float(m.predict([features(window[::-1][:LAGS], t)])[0]))
            out.append(round(p, 2))
            window = window[1:] + [p]
        preds[name] = out

    return jsonify(
        history={"ts": ts[-48:], "pm2_5": [round(v, 2) for v in vals[-48:]]},
        future_ts=future_ts,
        predictions=preds,
        metrics=metrics,
    )


if __name__ == "__main__":
    app.run(port=5000, debug=True)
