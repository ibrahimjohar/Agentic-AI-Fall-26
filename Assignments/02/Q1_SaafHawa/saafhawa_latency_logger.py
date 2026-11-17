"""
SaafHawa tool-latency logger  (Assignment Beta, Q1 Section 2.6)
Ibrahim Johar Farooqi - 23K-0074

Times the four agent tools and appends every measurement to tool_latency_log.csv:
  1. air_quality  - OpenAQ v3, latest PM2.5 / PM10 for a Karachi monitoring location
  2. weather      - Open-Meteo forecast API, current temperature / humidity / wind for Karachi
  3. heat_index   - NWS Rothfusz heat-index formula, computed locally
  4. kb_lookup    - TF-IDF search over knowledge_base.json, computed locally

Only uses the Python standard library.

Usage (run once on day 1 and again on day 2, each run gives 12 calls per API):
    set OPENAQ_API_KEY=your_key        (Windows cmd)   |   $env:OPENAQ_API_KEY="your_key"  (PowerShell)
    python saafhawa_latency_logger.py --rounds 12 --gap 20
"""
import argparse, csv, json, math, os, re, statistics, sys, time, urllib.parse, urllib.request
from collections import Counter
from datetime import datetime, timezone, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "tool_latency_log.csv")
CACHE = os.path.join(HERE, "openaq_location.json")
KB_PATH = os.path.join(HERE, "knowledge_base.json")
PKT = timezone(timedelta(hours=5))
KHI_LAT, KHI_LON = 24.8607, 67.0011

QUERIES = [
    "is it safe for my child to walk home from school at noon",
    "evening jog by the sea today humidity",
    "asthma patient pm2.5 outdoor exercise",
    "labourer working outside heatwave water breaks",
    "elderly father heatstroke signs what to do",
    "pm10 high keep windows closed children",
]


def http_get(url, headers=None, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "SaafHawa-latency-logger/1.0", **(headers or {})})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        status = r.status
    return (time.perf_counter() - t0) * 1000, status, json.loads(body)


def find_openaq_location(key):
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    url = (f"https://api.openaq.org/v3/locations?coordinates={KHI_LAT},{KHI_LON}"
           f"&radius=25000&limit=50")
    _, _, data = http_get(url, {"X-API-Key": key})
    best = None
    for loc in data.get("results", []):
        params = {s["parameter"]["name"] for s in loc.get("sensors", [])}
        score = ("pm25" in params) + ("pm10" in params)
        last = (loc.get("datetimeLast") or {}).get("utc") or ""
        if score and (best is None or (score, last) > (best[0], best[1])):
            best = (score, last, loc)
    if best is None:
        sys.exit("No OpenAQ location with PM2.5/PM10 found within 25 km of Karachi.")
    loc = best[2]
    info = {"id": loc["id"], "name": loc.get("name"),
            "sensors": {s["id"]: s["parameter"]["name"] for s in loc.get("sensors", [])}}
    json.dump(info, open(CACHE, "w"), indent=2)
    print(f"Using OpenAQ location {info['id']} ({info['name']}), sensors: {set(info['sensors'].values())}")
    return info


def tool_air_quality(key, loc):
    ms, status, data = http_get(f"https://api.openaq.org/v3/locations/{loc['id']}/latest", {"X-API-Key": key})
    vals = {}
    for r in data.get("results", []):
        name = loc["sensors"].get(str(r.get("sensorsId"))) or loc["sensors"].get(r.get("sensorsId"))
        if name in ("pm25", "pm10"):
            vals[name] = r.get("value")
    return ms, status, f"pm25={vals.get('pm25')} pm10={vals.get('pm10')}"


def tool_weather():
    url = ("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
        "latitude": KHI_LAT, "longitude": KHI_LON, "timezone": "Asia/Karachi",
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m"}))
    ms, status, data = http_get(url)
    c = data["current"]
    return ms, status, (c["temperature_2m"], c["relative_humidity_2m"], c["wind_speed_10m"])


def heat_index_c(t_c, rh):
    # NWS: simple Steadman form first, full Rothfusz regression with adjustments if HI >= 80 F
    t = t_c * 9 / 5 + 32
    hi = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    if (hi + t) / 2 >= 80:
        hi = (-42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh - 6.83783e-3 * t * t
              - 5.481717e-2 * rh * rh + 1.22874e-3 * t * t * rh + 8.5282e-4 * t * rh * rh - 1.99e-6 * t * t * rh * rh)
        if rh < 13 and 80 <= t <= 112:
            hi -= ((13 - rh) / 4) * math.sqrt((17 - abs(t - 95)) / 17)
        elif rh > 85 and 80 <= t <= 87:
            hi += ((rh - 85) / 10) * ((87 - t) / 5)
    return (hi - 32) * 5 / 9


def tokenize(s):
    return re.findall(r"[a-z0-9.]+", s.lower())


class KB:
    def __init__(self, path):
        self.docs = json.load(open(path, encoding="utf-8"))
        self.tfs = [Counter(tokenize(d["text"])) for d in self.docs]
        df = Counter(w for tf in self.tfs for w in tf)
        n = len(self.docs)
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}

    def search(self, q, k=3):
        qt = Counter(tokenize(q))
        scores = []
        for d, tf in zip(self.docs, self.tfs):
            s = sum(qt[w] * tf[w] * self.idf.get(w, 0) ** 2 for w in qt)
            norm = math.sqrt(sum((c * self.idf[w]) ** 2 for w, c in tf.items()))
            scores.append((s / norm if norm else 0, d["id"]))
        return sorted(scores, reverse=True)[:k]


def log_row(writer, tool, ms, status, note, run_id):
    now = datetime.now(PKT)
    writer.writerow([now.isoformat(timespec="seconds"), now.date().isoformat(), run_id, tool, f"{ms:.6f}", status, note])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--gap", type=float, default=20, help="seconds between rounds")
    args = ap.parse_args()

    key = os.environ.get("OPENAQ_API_KEY")
    if not key:
        sys.exit("Set OPENAQ_API_KEY first (free key from https://explore.openaq.org/register).")
    loc = find_openaq_location(key)
    kb = KB(KB_PATH)

    new = not os.path.exists(LOG)
    run_id = datetime.now(PKT).strftime("run_%Y%m%d_%H%M")
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp_pkt", "date", "run_id", "tool", "latency_ms", "status", "note"])
        for i in range(args.rounds):
            try:
                ms, st, note = tool_air_quality(key, loc)
            except Exception as e:
                ms, st, note = float("nan"), "error", repr(e)[:120]
            log_row(w, "air_quality_openaq", ms, st, note, run_id)

            try:
                ms, st, (temp, rh, wind) = tool_weather()
                note = f"T={temp}C RH={rh}% wind={wind}km/h"
            except Exception as e:
                ms, st, note, temp, rh = float("nan"), "error", repr(e)[:120], 35.0, 60.0
            log_row(w, "weather_openmeteo", ms, st, note, run_id)

            # local tools are sub-millisecond, so time 1000 repeats and log the per-call mean
            reps = 1000
            t0 = time.perf_counter()
            for _ in range(reps):
                hi = heat_index_c(temp, rh)
            log_row(w, "heat_index_local", (time.perf_counter() - t0) * 1000 / reps, 200, f"HI={hi:.1f}C (mean of {reps})", run_id)

            q = QUERIES[i % len(QUERIES)]
            reps = 200
            t0 = time.perf_counter()
            for _ in range(reps):
                hits = kb.search(q)
            log_row(w, "kb_lookup_local", (time.perf_counter() - t0) * 1000 / reps, 200, f"top={hits[0][1]} (mean of {reps})", run_id)
            f.flush()
            print(f"round {i + 1}/{args.rounds} done")
            if i < args.rounds - 1:
                time.sleep(args.gap)

    rows = list(csv.DictReader(open(LOG, encoding="utf-8")))
    print(f"\nLog now has {len(rows)} rows in {LOG}")
    for tool in sorted({r['tool'] for r in rows}):
        vals = [float(r["latency_ms"]) for r in rows if r["tool"] == tool and r["status"] != "error"]
        days = len({r["date"] for r in rows if r["tool"] == tool})
        if vals:
            print(f"  {tool:22s} n={len(vals):3d} days={days} median={statistics.median(vals):9.3f} ms")


if __name__ == "__main__":
    main()
