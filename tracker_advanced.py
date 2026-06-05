"""
============================================================
  Advanced RSSI Indoor Location Tracker  |  Raspberry Pi 5
  ---------------------------------------------------------
  Dataset : cmti_with_photos.xlsx
  ---------------------------------------------------------
  Excel structure (verified):
    Row 0  : header  ->  sl no | place name | Latitude | Longitude | Photos
    Row 1+ : data    ->  int   | str        | float    | float     | (image)
    40 data rows, 40 embedded images (one per location row)
    Image anchor row N  =>  df_index = N - 1  (0-based)
  ---------------------------------------------------------
  Features
  ---------------------------------------------------------
  * Multi-AP fingerprinting  – kNN in full AP-vector space
  * Kalman filter            – smooths lat/lon estimates
  * WebSocket broadcast      – live JSON push to browser UI
  * Blockchain audit chain   – tamper-evident scan log
  * ML nearest-centroid      – softmax-weighted prediction
  * Direction / distance     – parallelogram law bearing
  * Structured logging       – clean terminal output
  ---------------------------------------------------------
  Dependencies
  ---------------------------------------------------------
      pip install pandas openpyxl pillow matplotlib
                  numpy websockets

  Hardware Wi-Fi scan uses:
      iw dev wlan0 station dump
  Falls back to interactive keyboard input when hw absent.
============================================================
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import pickle
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from PIL import Image
import matplotlib.pyplot as plt

# ── optional websockets ───────────────────────────────────────
try:
    import websockets
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────

EXCEL_FILE        = "cmti_with_photos.xlsx"
K_NEIGHBOURS      = 3
MAX_RSSI_DIFF     = 30       # dBm threshold for kNN candidacy
TRACKING_INTERVAL = 3        # seconds between scans
WS_PORT           = 8765
MODEL_FILE        = Path("tracker_location_model.pkl")
AUDIT_LEDGER_FILE = Path("tracker_audit_chain.jsonl")
MODEL_TEMPERATURE = 12.0

# Access-point MAC addresses from your site survey.
AP_MACS = [
    "aa:bb:cc:dd:ee:01",
    "aa:bb:cc:dd:ee:02",
    "aa:bb:cc:dd:ee:03",
    "aa:bb:cc:dd:ee:04",
]

PROCESS_NOISE     = 0.05
MEASUREMENT_NOISE = 1.5

# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt = "%H:%M:%S",
    stream  = sys.stdout,
)
log = logging.getLogger("tracker")


# ─────────────────────────────────────────────────────────────
#  DATA TYPES
# ─────────────────────────────────────────────────────────────

@dataclass
class LocationRecord:
    """One location row from the Excel sheet."""
    index    : int           # 0-based row index (matches image_map key)
    sl_no    : int           # original 'sl no' value from sheet
    place    : str
    latitude : float
    longitude: float
    rssi_vec : np.ndarray    # shape (len(AP_MACS),)


@dataclass
class ScanResult:
    """Output of one position fix."""
    place       : str
    latitude    : float
    longitude   : float
    k_lat       : float
    k_lon       : float
    confidence  : float
    direction   : float
    distance_m  : float
    df_index    : int
    scan_number : int
    timestamp   : float = field(default_factory=time.time)
    alternatives: list  = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("rssi_vec", None)
        return d


# ─────────────────────────────────────────────────────────────
#  KALMAN FILTER  (1-D, independent for lat and lon)
# ─────────────────────────────────────────────────────────────

class KalmanFilter1D:
    def __init__(self, q: float = PROCESS_NOISE, r: float = MEASUREMENT_NOISE):
        self.q = q
        self.r = r
        self.x = None
        self.p = 1.0

    def update(self, z: float) -> float:
        if self.x is None:
            self.x = z
            return z
        self.p += self.q
        k      = self.p / (self.p + self.r)
        self.x = self.x + k * (z - self.x)
        self.p = (1 - k) * self.p
        return self.x


# ─────────────────────────────────────────────────────────────
#  DATABASE LOADING
#  Exact column layout (verified from file inspection):
#    Col 0 : sl no       (int, 0-based serial number)
#    Col 1 : place name  (str)
#    Col 2 : Latitude    (float)
#    Col 3 : Longitude   (float)
#    Col 4 : Photos      (embedded image — read separately)
#  Row 0   : header row  → skip with iloc[1:]
#  40 data rows, 40 embedded images
#  Image anchor_row N   → df_index = N - 1
# ─────────────────────────────────────────────────────────────

def load_location_database(
    file_path: str,
) -> tuple[list[LocationRecord], dict[int, Image.Image]]:
    """
    Parse cmti_with_photos.xlsx into LocationRecords + image map.

    Column mapping (0-indexed):
        0 -> sl no  |  1 -> place name  |  2 -> Latitude  |  3 -> Longitude

    Image anchor formula:
        df_index = anchor._from.row - 1
        (anchor rows are 1-based; header is row 0 in Excel, data starts row 1)
    """
    # ── Read tabular data ─────────────────────────────────────
    raw = pd.read_excel(file_path, header=0)
    # Normalise column names defensively
    raw.columns = ["sl_no", "place", "latitude", "longitude", "photo"]

    # Drop the trailing empty row (sl_no=40, all None) if present
    raw = raw.dropna(subset=["place"]).reset_index(drop=True)

    # Clean coordinate columns (strip any stray degree symbols)
    for col in ("latitude", "longitude"):
        raw[col] = (
            raw[col]
            .astype(str)
            .str.replace("°", "", regex=False)
            .str.replace("deg", "", regex=False)
            .str.strip()
            .astype(float)
        )
    raw["place"]  = raw["place"].astype(str).str.strip()
    raw["sl_no"]  = raw["sl_no"].astype(int)

    # ── Simulate multi-AP RSSI fingerprints ──────────────────
    # Stable, deterministic per-row vectors (seed=42).
    # Replace with real site-survey data in production.
    rng = np.random.default_rng(42)
    records: list[LocationRecord] = []
    for idx, row in raw.iterrows():
        base = -45 - (idx % 45)
        vec  = np.array([
            float(np.clip(base - i * 7 + rng.integers(-3, 4), -95, -30))
            for i in range(len(AP_MACS))
        ])
        records.append(LocationRecord(
            index    = int(idx),
            sl_no    = int(row["sl_no"]),
            place    = str(row["place"]),
            latitude = float(row["latitude"]),
            longitude= float(row["longitude"]),
            rssi_vec = vec,
        ))

    # ── Extract embedded images ───────────────────────────────
    # Image anchor._from.row is 1-based Excel row.
    # Header is row 0 in Python (Excel row 0), data starts at Excel row 1.
    # Therefore: df_index = anchor._from.row - 1
    wb = load_workbook(file_path)
    ws = wb.active
    image_map: dict[int, Image.Image] = {}
    for sheet_img in ws._images:
        try:
            df_index = sheet_img.anchor._from.row - 1   # verified mapping
            if df_index < 0 or df_index >= len(records):
                continue
            pil = Image.open(io.BytesIO(sheet_img._data()))
            pil.load()
            image_map[df_index] = pil
        except Exception:
            pass
    wb.close()

    log.info(
        "Database loaded: %d locations, %d images matched.",
        len(records), len(image_map),
    )
    return records, image_map


# ─────────────────────────────────────────────────────────────
#  MODEL TRAINING
# ─────────────────────────────────────────────────────────────

def _canonical_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_model_bundle(bundle: dict) -> str:
    serialisable = {
        "version"    : bundle["version"],
        "trained_at" : round(float(bundle["trained_at"]), 6),
        "ap_count"   : int(bundle["ap_count"]),
        "temperature": round(float(bundle["temperature"]), 6),
        "prototypes" : [
            {
                "place"       : p["place"],
                "latitude"    : round(float(p["latitude"]), 6),
                "longitude"   : round(float(p["longitude"]), 6),
                "df_index"    : int(p["df_index"]),
                "sample_count": int(p["sample_count"]),
                "centroid"    : np.round(
                    np.asarray(p["centroid"], dtype=float), 3
                ).tolist(),
            }
            for p in bundle.get("prototypes", [])
        ],
    }
    return hashlib.sha256(
        _canonical_json(serialisable).encode("utf-8")
    ).hexdigest()


def train_location_model(
    database  : list[LocationRecord],
    model_path: Path = MODEL_FILE,
) -> dict:
    """Build and persist a nearest-centroid prototype model."""
    if not database:
        raise ValueError("Cannot train: empty database.")

    grouped: dict[str, list[LocationRecord]] = {}
    for rec in database:
        grouped.setdefault(rec.place, []).append(rec)

    prototypes: list[dict] = []
    for place in sorted(grouped.keys(), key=str.lower):
        recs     = grouped[place]
        stack    = np.vstack([r.rssi_vec for r in recs])
        centroid = stack.mean(axis=0)
        avg_lat  = float(np.mean([r.latitude  for r in recs]))
        avg_lon  = float(np.mean([r.longitude for r in recs]))
        exemplar = min(recs, key=lambda r: float(
            np.linalg.norm(r.rssi_vec - centroid)
        ))
        prototypes.append({
            "place"       : place,
            "centroid"    : centroid,
            "latitude"    : avg_lat,
            "longitude"   : avg_lon,
            "df_index"    : exemplar.index,
            "sample_count": len(recs),
        })

    bundle = {
        "version"    : 1,
        "trained_at" : time.time(),
        "ap_count"   : len(AP_MACS),
        "temperature": MODEL_TEMPERATURE,
        "prototypes" : prototypes,
    }
    bundle["model_hash"] = _hash_model_bundle(bundle)

    with model_path.open("wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)

    log.info("ML model: %d prototypes saved to %s", len(prototypes), model_path)
    return bundle


# ─────────────────────────────────────────────────────────────
#  BLOCKCHAIN AUDIT CHAIN
# ─────────────────────────────────────────────────────────────

class BlockchainAuditChain:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.previous_hash = "0" * 64
        if self.path.exists() and self.path.stat().st_size > 0:
            self.previous_hash = self._verify_existing_chain()

    def _hash_payload(self, payload: dict) -> str:
        return hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()

    def _verify_existing_chain(self) -> str:
        previous = "0" * 64
        with self.path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                block      = json.loads(line)
                block_hash = block.get("hash")
                payload    = dict(block)
                payload.pop("hash", None)
                expected   = self._hash_payload(payload)
                if block_hash != expected or payload.get("previous_hash") != previous:
                    raise ValueError(f"Audit chain corrupt at line {line_no}")
                previous = block_hash
        return previous

    def append(
        self,
        result     : dict,
        rssi_vec   : np.ndarray,
        scan_number: int,
        model_hash : str,
    ) -> dict:
        block = {
            "scan"             : int(scan_number),
            "timestamp"        : float(time.time()),
            "place"            : result["place"],
            "latitude"         : float(result["latitude"]),
            "longitude"        : float(result["longitude"]),
            "k_lat"            : float(result.get("k_lat", result["latitude"])),
            "k_lon"            : float(result.get("k_lon", result["longitude"])),
            "confidence"       : float(result["confidence"]),
            "direction_bearing": float(result.get("direction_bearing", 0.0)),
            "distance_m"       : float(result.get("distance_m", 0.0)),
            "df_index"         : int(result["df_index"]),
            "model_hash"       : model_hash,
            "rssi_vec"         : [float(v) for v in np.round(rssi_vec, 1)],
            "alternatives"     : result.get("alternatives", []),
            "previous_hash"    : self.previous_hash,
        }
        block["hash"] = self._hash_payload(block)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(block, ensure_ascii=False, sort_keys=True) + "\n")
        self.previous_hash = block["hash"]
        return block


# ─────────────────────────────────────────────────────────────
#  RSSI MEASUREMENT
# ─────────────────────────────────────────────────────────────

def _parse_iw_output(raw: str) -> dict[str, float]:
    result: dict[str, float] = {}
    bssid = None
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("Station") or line.startswith("BSS"):
            parts = line.split()
            bssid = parts[1] if len(parts) > 1 else None
        if bssid and "signal:" in line:
            try:
                result[bssid] = float(line.split("signal:")[1].strip().split()[0])
            except ValueError:
                pass
    return result


def measure_rssi_vector_hardware() -> Optional[np.ndarray]:
    try:
        out = subprocess.run(
            ["iw", "dev", "wlan0", "station", "dump"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        readings = _parse_iw_output(out)
        return np.array([readings.get(mac, -95.0) for mac in AP_MACS])
    except Exception:
        return None


def measure_rssi_vector_interactive() -> np.ndarray:
    print(f"\n[RSSI] Enter RSSI for each of {len(AP_MACS)} access point(s).")
    print("       Press Enter to use a random demo value for any AP.")
    vec = []
    rng = np.random.default_rng()
    for mac in AP_MACS:
        while True:
            try:
                raw = input(f"  {mac} (dBm, e.g. -65): ").strip()
            except EOFError:
                raw = ""
            if raw == "":
                val = float(rng.integers(-80, -40))
                print(f"  -> demo: {val:.0f} dBm")
                vec.append(val)
                break
            try:
                val = float(raw)
                if -100 <= val <= 0:
                    vec.append(val)
                    break
                print("  [!] Must be between -100 and 0.")
            except ValueError:
                print("  [!] Enter a number.")
    return np.array(vec)


def measure_rssi_vector() -> np.ndarray:
    hw = measure_rssi_vector_hardware()
    if hw is not None:
        log.info("Hardware RSSI: %s", hw)
        return hw
    return measure_rssi_vector_interactive()


# ─────────────────────────────────────────────────────────────
#  GPS DIRECTION & DISTANCE GEOMETRY
# ─────────────────────────────────────────────────────────────

def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def rssi_distance_to_meters(rssi_dist: float, num_aps: int = 4) -> float:
    return float(np.clip(rssi_dist / (np.sqrt(num_aps) * 3.5), 0.5, 100.0))


def gps_to_cartesian(lat, lon, ref_lat=0.0, ref_lon=0.0) -> tuple[float, float]:
    mpl = 111_320.0
    mpp = mpl * np.cos(np.radians(ref_lat))
    return (lon - ref_lon) * mpp, (lat - ref_lat) * mpl


def vector_resultant_direction(vectors, weights) -> tuple[float, float]:
    if not vectors:
        return 0.0, 0.0
    we = sum(w * v[0] for w, v in zip(weights, vectors))
    wn = sum(w * v[1] for w, v in zip(weights, vectors))
    return (
        float(np.sqrt(we ** 2 + wn ** 2)),
        float((np.degrees(np.arctan2(we, wn)) + 360) % 360),
    )


def estimate_direction_from_knn(
    knn_results: dict,
    database   : list[LocationRecord],
    ref_lat    : float = 0.0,
    ref_lon    : float = 0.0,
) -> tuple[float, float]:
    main_lat, main_lon = knn_results["latitude"], knn_results["longitude"]
    distance = haversine_distance(ref_lat, ref_lon, main_lat, main_lon)
    alts     = knn_results.get("alternatives", [])

    if not alts:
        e, n = gps_to_cartesian(main_lat, main_lon, ref_lat, ref_lon)
        return distance, float((np.degrees(np.arctan2(e, n)) + 360) % 360)

    record_lookup = {r.index: r for r in database}
    conf          = knn_results["confidence"]
    vectors       = [gps_to_cartesian(main_lat, main_lon, ref_lat, ref_lon)]
    weights_raw   = [conf]
    total_alt_w   = max(0.001, 1.0 - conf)
    n_alts        = max(1, len(alts))

    for alt in alts:
        alt_d   = float(alt.get("distance", alt.get("distance_m",
                         alt.get("rssi_distance", 0.0))))
        alt_rec = record_lookup.get(int(alt["df_index"])) if alt.get("df_index") is not None else None
        alt_lat = alt_rec.latitude  if alt_rec else main_lat
        alt_lon = alt_rec.longitude if alt_rec else main_lon
        vectors.append(gps_to_cartesian(alt_lat, alt_lon, ref_lat, ref_lon))
        weights_raw.append((1.0 / (alt_d + 1.0)) * total_alt_w / n_alts)

    w_arr = np.array(weights_raw)
    w_arr /= w_arr.sum()
    _, bearing = vector_resultant_direction(vectors, w_arr)
    return distance, bearing


# ─────────────────────────────────────────────────────────────
#  kNN LOCATION MATCHING
# ─────────────────────────────────────────────────────────────

def find_location_knn(
    query_vec: np.ndarray,
    database : list[LocationRecord],
    k        : int = K_NEIGHBOURS,
) -> Optional[dict]:
    distances = sorted(
        [(float(np.linalg.norm(query_vec - r.rssi_vec)), r) for r in database],
        key=lambda t: t[0],
    )
    threshold  = MAX_RSSI_DIFF * (len(AP_MACS) ** 0.5)
    candidates = [(d, r) for d, r in distances if d <= threshold]
    if not candidates:
        return None

    top_k   = candidates[:k]
    weights = np.array([1.0 / (d ** 2 + 1e-9) for d, _ in top_k])
    weights /= weights.sum()

    w_lat = float(sum(w * r.latitude  for w, (_, r) in zip(weights, top_k)))
    w_lon = float(sum(w * r.longitude for w, (_, r) in zip(weights, top_k)))

    best_d, best_rec = top_k[0]
    confidence       = float(max(0.0, 1.0 - best_d / threshold))

    result = {
        "place"       : best_rec.place,
        "latitude"    : w_lat,
        "longitude"   : w_lon,
        "df_index"    : best_rec.index,
        "confidence"  : confidence,
        "alternatives": [
            {
                "place"        : r.place,
                "rssi_distance": round(d, 2),
                "distance_m"   : round(rssi_distance_to_meters(d, len(AP_MACS)), 1),
                "df_index"     : r.index,
            }
            for d, r in top_k[1:]
        ],
    }
    dist_m, bearing = estimate_direction_from_knn(result, database)
    result["direction_bearing"] = round(bearing, 1)
    result["distance_m"]        = round(dist_m, 2)
    return result


# ─────────────────────────────────────────────────────────────
#  ML LOCATION MATCHING  (nearest-centroid + softmax)
# ─────────────────────────────────────────────────────────────

def find_location_ml(
    query_vec   : np.ndarray,
    database    : list[LocationRecord],
    model_bundle: dict,
    k           : int = K_NEIGHBOURS,
) -> Optional[dict]:
    prototypes = model_bundle.get("prototypes", [])
    if not prototypes:
        return find_location_knn(query_vec, database, k)

    centroid_matrix = np.vstack([
        np.asarray(p["centroid"], dtype=float) for p in prototypes
    ])
    distances  = np.linalg.norm(centroid_matrix - query_vec, axis=1)
    ranked     = np.argsort(distances)
    threshold  = MAX_RSSI_DIFF * (len(AP_MACS) ** 0.5)
    candidates = [i for i in ranked if float(distances[i]) <= threshold]
    if not candidates:
        return None

    top_k       = candidates[:k]
    temperature = max(1e-6, float(model_bundle.get("temperature", MODEL_TEMPERATURE)))
    logits      = -distances / temperature
    logits     -= float(np.max(logits))
    probs       = np.exp(logits)
    probs      /= probs.sum()

    best_i      = int(ranked[0])
    best_proto  = prototypes[best_i]
    nearest_rec = next(
        (r for r in database if r.index == int(best_proto["df_index"])),
        min(database, key=lambda r: float(np.linalg.norm(query_vec - r.rssi_vec))),
    )

    conf       = float(probs[best_i])
    dist_conf  = float(max(0.0, 1.0 - distances[best_i] / max(1.0, threshold)))
    confidence = float(np.clip(0.7 * conf + 0.3 * dist_conf, 0.0, 1.0))

    result = {
        "place"               : best_proto["place"],
        "latitude"            : float(best_proto["latitude"]),
        "longitude"           : float(best_proto["longitude"]),
        "df_index"            : int(nearest_rec.index),
        "confidence"          : confidence,
        "model_probability"   : float(probs[best_i]),
        "nearest_match"       : best_proto["place"],
        "nearest_rssi_distance": round(float(distances[best_i]), 2),
        "alternatives"        : [
            {
                "place"     : prototypes[i]["place"],
                "distance"  : round(float(distances[i]), 2),
                "distance_m": round(rssi_distance_to_meters(
                                  float(distances[i]), len(AP_MACS)), 1),
                "df_index"  : int(prototypes[i]["df_index"]),
            }
            for i in top_k[1:]
        ],
    }
    dist_m, bearing = estimate_direction_from_knn(result, database)
    result["direction_bearing"] = round(bearing, 1)
    result["distance_m"]        = round(dist_m, 2)
    return result


# ─────────────────────────────────────────────────────────────
#  IMAGE DISPLAY  (matplotlib)
# ─────────────────────────────────────────────────────────────

def display_location_image(
    place_name: str,
    df_index  : int,
    image_map : dict[int, Image.Image],
) -> None:
    fig = plt.figure(figsize=(7, 5))
    fig.patch.set_facecolor("#1e1e2e")
    ax  = fig.add_subplot(111)
    ax.set_facecolor("#1e1e2e")
    ax.axis("off")

    if df_index in image_map:
        pil = image_map[df_index].copy()
        pil.thumbnail((800, 800), Image.LANCZOS)
        ax.imshow(pil)
        title_color = "#cdd6f4"
    else:
        ax.text(
            0.5, 0.5, "No image available",
            ha="center", va="center", fontsize=14,
            color="#6c7086", transform=ax.transAxes,
        )
        title_color = "#6c7086"

    fig.suptitle(
        f"Location: {place_name}",
        color=title_color, fontsize=12, fontweight="bold", y=0.97,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show(block=False)
    plt.pause(0.5)


# ─────────────────────────────────────────────────────────────
#  WEBSOCKET BROADCAST
# ─────────────────────────────────────────────────────────────

class WebSocketBroadcaster:
    def __init__(self, port: int = WS_PORT):
        self.port     = port
        self._clients : set = set()
        self._loop    : Optional[asyncio.AbstractEventLoop] = None
        self._thread  : Optional[threading.Thread] = None

    async def _handler(self, ws):
        self._clients.add(ws)
        log.info("Map UI connected (%d client(s))", len(self._clients))
        try:
            await ws.wait_closed()
        finally:
            self._clients.discard(ws)

    async def _serve(self):
        async with websockets.serve(self._handler, "localhost", self.port):
            log.info("WebSocket listening on ws://localhost:%d", self.port)
            await asyncio.Future()

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    def start(self):
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def broadcast(self, payload: dict):
        if not self._loop or not self._clients:
            return
        msg = json.dumps(payload)

        async def _send():
            if self._clients:
                await asyncio.gather(
                    *[c.send(msg) for c in list(self._clients)],
                    return_exceptions=True,
                )

        asyncio.run_coroutine_threadsafe(_send(), self._loop)


# ─────────────────────────────────────────────────────────────
#  TERMINAL OUTPUT
# ─────────────────────────────────────────────────────────────

def _bearing_to_direction(bearing: float) -> str:
    directions = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]
    return directions[round(bearing / 22.5) % 16]


def display_result(
    result  : Optional[dict],
    rssi_vec: np.ndarray,
    scan_n  : int,
) -> None:
    bar = "-" * 62
    print("")
    print(bar)
    print(f"  Scan #{scan_n}  |  RSSI: {np.round(rssi_vec, 1).tolist()}")
    print(bar)

    if result is None:
        print("  Location     : UNKNOWN")
        print("  Reason       : No match within RSSI distance threshold.")
        print("  Tip          : Widen MAX_RSSI_DIFF or update fingerprints.")
    else:
        print(f"  Best Match   : {result['place']}")
        print(f"  Latitude     : {result['latitude']:.6f} deg")
        print(f"  Longitude    : {result['longitude']:.6f} deg")
        print(f"  Kalman Lat   : {result.get('k_lat', result['latitude']):.6f} deg")
        print(f"  Kalman Lon   : {result.get('k_lon', result['longitude']):.6f} deg")
        print(f"  Confidence   : {result['confidence'] * 100:.1f}%")
        if "model_probability" in result:
            print(f"  Model Prob.  : {result['model_probability'] * 100:.1f}%")
        if result.get("audit_hash"):
            print(f"  Block Hash   : {result['audit_hash'][:20]}...")
        bearing  = result.get("direction_bearing", 0.0)
        distance = result.get("distance_m", 0.0)
        print(f"  Direction    : {_bearing_to_direction(bearing)}  ({bearing:.1f} deg)")
        print(f"  Distance     : {distance:.1f} m")
        alts = result.get("alternatives", [])
        if alts:
            print("  Alternatives :")
            for a in alts:
                print(f"    * {a['place']:<55s}  ({a.get('distance_m', 0.0):.1f} m)")
    print(bar)
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Advanced RSSI Campus Location Tracker ===")
    log.info("Dataset       : %s", EXCEL_FILE)
    log.info("APs monitored : %d", len(AP_MACS))
    log.info("kNN k         : %d", K_NEIGHBOURS)
    log.info("Scan interval : %d s", TRACKING_INTERVAL)

    # ── Load Excel database ──────────────────────────────────
    if not Path(EXCEL_FILE).exists():
        log.error("File not found: %s", EXCEL_FILE)
        log.error("Place %s in the same directory as this script.", EXCEL_FILE)
        sys.exit(1)

    try:
        database, image_map = load_location_database(EXCEL_FILE)
    except Exception as exc:
        log.exception("Failed to load database: %s", exc)
        sys.exit(1)

    # ── Train ML model ───────────────────────────────────────
    try:
        model_bundle = train_location_model(database, MODEL_FILE)
    except Exception as exc:
        log.exception("Failed to train ML model: %s", exc)
        sys.exit(1)

    # ── Audit chain ──────────────────────────────────────────
    try:
        audit_chain = BlockchainAuditChain(AUDIT_LEDGER_FILE)
        log.info("Audit ledger  : %s", AUDIT_LEDGER_FILE)
    except Exception as exc:
        log.exception("Failed to initialise audit chain: %s", exc)
        sys.exit(1)

    # ── Kalman filters ───────────────────────────────────────
    kf_lat = KalmanFilter1D()
    kf_lon = KalmanFilter1D()

    # ── WebSocket broadcaster ────────────────────────────────
    broadcaster: Optional[WebSocketBroadcaster] = None
    if WS_AVAILABLE:
        broadcaster = WebSocketBroadcaster(WS_PORT)
        broadcaster.start()
        log.info("WebSocket ready at ws://localhost:%d", WS_PORT)
    else:
        log.warning("websockets not installed -- run: pip install websockets")

    log.info("Press Ctrl+C to stop.\n")

    scan_count = 0
    try:
        while True:
            scan_count += 1
            log.info("-- Scan #%d --", scan_count)

            rssi_vec = measure_rssi_vector()
            result   = find_location_ml(rssi_vec, database, model_bundle)

            if result is not None:
                result["k_lat"]      = round(kf_lat.update(result["latitude"]),  6)
                result["k_lon"]      = round(kf_lon.update(result["longitude"]), 6)
                result["model_hash"] = model_bundle["model_hash"]
                audit_block          = audit_chain.append(
                    result, rssi_vec, scan_count, model_bundle["model_hash"]
                )
                result["audit_hash"]    = audit_block["hash"]
                result["previous_hash"] = audit_block["previous_hash"]

            display_result(result, rssi_vec, scan_count)

            if result is not None:
                # ── WebSocket push ───────────────────────────
                if broadcaster:
                    broadcaster.broadcast({
                        "scan"             : scan_count,
                        "place"            : result["place"],
                        "latitude"         : result["latitude"],
                        "longitude"        : result["longitude"],
                        "k_lat"            : result["k_lat"],
                        "k_lon"            : result["k_lon"],
                        "confidence"       : round(result["confidence"], 3),
                        "model_probability": round(result.get("model_probability", 0.0), 3),
                        "direction_bearing": result.get("direction_bearing", 0.0),
                        "distance_m"       : result.get("distance_m", 0.0),
                        "model_hash"       : result.get("model_hash"),
                        "audit_hash"       : result.get("audit_hash"),
                        "rssi_vec"         : list(np.round(rssi_vec, 1)),
                        "alternatives"     : result.get("alternatives", []),
                        "timestamp"        : time.time(),
                    })

                # ── Show location photo ──────────────────────
                display_location_image(
                    place_name = result["place"],
                    df_index   = result["df_index"],
                    image_map  = image_map,
                )

            log.info("Next scan in %d s...", TRACKING_INTERVAL)
            time.sleep(TRACKING_INTERVAL)

    except KeyboardInterrupt:
        log.info("Stopped after %d scan(s). Goodbye!", scan_count)
        plt.close("all")
        sys.exit(0)


if __name__ == "__main__":
    main()