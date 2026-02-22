"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   SWING BULL TRADER API SERVER v4.0 — Flask Backend                         ║
║   Architected by Freddy • For Cloudflare/Appwrite Deployment                ║
╚══════════════════════════════════════════════════════════════════════════════╝

ENDPOINTS:
  GET  /                          → health check + dashboard meta
  GET  /api/market-regime         → Market Pulse (renamed) + breadth + sectors
  POST /api/scan                  → stock scanner with portfolio blocks
  POST /api/analyse               → deep single-stock analysis
  GET  /api/sectors               → sector RS rankings
  POST /api/portfolio/validate    → portfolio-level risk check
  POST /api/portfolio/positions   → update active positions for constraint checks
  POST /api/trade/record          → record a completed trade result
  GET  /api/performance           → expectancy & edge stability stats
  POST /api/active-trade          → evaluate an open position health
  POST /api/login                 → get session token
  POST /api/logout                → end session
  GET  /api/meta                  → dashboard branding & color palette

V4 CHANGES:
  ✅ Renamed "Freddy Gauge" → "Market Pulse"
  ✅ Added Market Pulse zones (0-45° Red, 45-135° Yellow, 135-180° Green)
  ✅ Portfolio hard blocks integrated
  ✅ Trap detection in scan results
  ✅ Intraday chase guard alerts
  ✅ Dashboard meta with branding and color palette
  ✅ UI-ready color hints in all responses
"""

import os
import logging
import json
import numpy as np
import threading
from datetime import datetime
from flask import Flask, jsonify, request
# flask_cors replaced by custom wildcard CORS handler below


# ═══════════════════════════════════════════════════════════════════════════
#  NUMPY-SAFE JSON ENCODER
#  Fixes: "Object of type bool/int64/float64 is not JSON serializable"
#  numpy returns its own bool/int/float types — this converts them to plain
#  Python types so Flask's jsonify() can handle them without crashing.
# ═══════════════════════════════════════════════════════════════════════════

def _safe_convert(obj):
    """Recursively convert all non-JSON-safe types in dicts/lists."""
    if isinstance(obj, dict):
        return {k: _safe_convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_convert(i) for i in obj]
    if isinstance(obj, bool) or isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    # Handle plain Python float NaN/Inf (not caught by numpy check above)
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    if isinstance(obj, np.ndarray):
        return [_safe_convert(i) for i in obj.tolist()]
    try:
        import pandas as pd
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return _safe_convert(obj.tolist())
    except Exception:
        pass
    return obj


def safe_jsonify(data):
    """Drop-in replacement for jsonify() that handles all numpy/bool/NaN types."""
    converted = _safe_convert(data)
    try:
        response_str = json.dumps(converted, allow_nan=False)
    except (ValueError, TypeError):
        # Brute-force fallback — replace any NaN/Inf that slipped through
        response_str = json.dumps(converted, allow_nan=True)
        response_str = (response_str
            .replace(": NaN",        ": null")
            .replace(":NaN",         ":null")
            .replace(": Infinity",   ": null")
            .replace(":-Infinity",   ":null")
            .replace(": -Infinity",  ": null"))
    return app.response_class(
        response=response_str,
        status=200,
        mimetype="application/json"
    )


class NumpyEncoder(json.JSONEncoder):
    """Kept for backward compat — safe_jsonify is the real fix."""
    def default(self, obj):
        if isinstance(obj, bool) or isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ── Import the trading engine ──────────────────────────────────────────────
from trading_engine_v4 import (
    SwingBullEngine,
    FreddyEngine,  # Backward compat alias
    SetupClassifier,
    SetupType,
    PortfolioRiskEngine,
    ActiveTradeEvaluator,
    MarketRegimeEngine,
    MarketBreadthEngine,
    VolatilityRegimeEngine,
    SectorLeadershipEngine,
    RegimeType,
    DASHBOARD_META,
)

# ═══════════════════════════════════════════════════════════════════════════
#  APP SETUP
# ═══════════════════════════════════════════════════════════════════════════

# ── Live NSE index constituent fetcher ───────────────────────────────────────
_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://niftyindices.com",
}

_NSE_URLS = {
    "nifty50":     "https://niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "midcap150":   "https://niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
}

def _fetch_nse_tickers(index: str) -> list:
    """Fetch live official ticker list from NSE. Falls back to hardcoded list on failure."""
    import requests, csv, io
    url = _NSE_URLS.get(index)
    if not url:
        return []
    try:
        resp = requests.get(url, headers=_NSE_HEADERS, timeout=15)
        resp.raise_for_status()
        reader = csv.DictReader(io.StringIO(resp.text))
        tickers = [f"{row['Symbol'].strip()}.NS" for row in reader if row.get('Symbol')]
        if len(tickers) > 10:
            logger.info(f"[NSE] Fetched {len(tickers)} live tickers for {index}")
            return tickers
    except Exception as e:
        logger.warning(f"[NSE] Could not fetch live {index} tickers: {e} — using hardcoded list")
    return []  # caller falls back to hardcoded

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("swingbull.api")

app = Flask(__name__)
# json_encoder removed — deprecated in Flask 3.x. safe_jsonify() handles all serialization.

# ── CORS: allow your frontend origins ─────────────────────────────────────
# Allow all Cloudflare Pages preview URLs + Cloudflare Worker + localhost
import re as _re

def _is_allowed_origin(origin: str) -> bool:
    if not origin:
        return False
    # Exact matches
    exact = [
        "https://swing-trader-dashboard.pages.dev",
        "https://swing-trader-worker.swingbulltrader.workers.dev",
        "http://localhost:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ]
    if origin in exact:
        return True
    # Wildcard: any *.swing-trader-dashboard.pages.dev preview URL
    if _re.match(r"^https://[a-z0-9]+\.swing-trader-dashboard\.pages\.dev$", origin):
        return True
    return False

class _WildcardCORS:
    """Flask-CORS replacement that supports wildcard origin matching."""
    pass

# Use after_request hook for CORS — gives us full control per-request
@app.after_request
def _add_cors(response):
    origin = request.headers.get("Origin", "")
    if _is_allowed_origin(origin):
        response.headers["Access-Control-Allow-Origin"]  = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Vary"] = "Origin"
    return response

@app.before_request
def _handle_options():
    if request.method == "OPTIONS":
        origin = request.headers.get("Origin", "")
        resp = app.make_default_options_response()
        if _is_allowed_origin(origin):
            resp.headers["Access-Control-Allow-Origin"]  = origin
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            resp.headers["Vary"] = "Origin"
        return resp

# ── Engine: initialised once at startup ────────────────────────────────────
TOTAL_CAPITAL    = float(os.environ.get("TOTAL_CAPITAL",    "1000000"))
RISK_PER_TRADE   = float(os.environ.get("RISK_PER_TRADE",   "2.0"))

engine = SwingBullEngine(
    total_capital=TOTAL_CAPITAL,
    risk_per_trade=RISK_PER_TRADE,
)
logger.info(f"SwingBullEngine v4 initialised — capital ₹{TOTAL_CAPITAL:,.0f}, risk {RISK_PER_TRADE}%/trade")


# ═══════════════════════════════════════════════════════════════════════════
#  STOCK UNIVERSE — Live from NSE with hardcoded fallback
# ═══════════════════════════════════════════════════════════════════════════

_NIFTY50_FALLBACK = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "NESTLEIND.NS", "HCLTECH.NS",
    "WIPRO.NS", "TECHM.NS", "M&M.NS", "POWERGRID.NS", "NTPC.NS",
    "COALINDIA.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "BAJAJFINSV.NS", "DRREDDY.NS",
    "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS", "EICHERMOT.NS", "GRASIM.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "INDUSINDBK.NS", "HINDALCO.NS", "BPCL.NS",
    "ONGC.NS", "TATACONSUM.NS", "BRITANNIA.NS", "HEROMOTOCO.NS", "SHRIRAMFIN.NS",
    "BAJAJ-AUTO.NS", "TRENT.NS", "BEL.NS", "VEDL.NS", "SBILIFE.NS",
]

NIFTY50_TICKERS = _fetch_nse_tickers("nifty50") or _NIFTY50_FALLBACK
logger.info(f"Nifty50 universe: {len(NIFTY50_TICKERS)} tickers")

# Map ticker → sector (for RS and sector-cap checks)
TICKER_SECTOR = {
    "HDFCBANK.NS": "banking",   "ICICIBANK.NS": "banking",
    "SBIN.NS": "banking",       "KOTAKBANK.NS": "banking",
    "AXISBANK.NS": "banking",   "INDUSINDBK.NS": "banking",
    "SHRIRAMFIN.NS": "banking", "BAJFINANCE.NS": "banking",
    "BAJAJFINSV.NS": "banking", "SBILIFE.NS": "banking",

    "TCS.NS": "it",             "INFY.NS": "it",
    "HCLTECH.NS": "it",         "WIPRO.NS": "it",
    "TECHM.NS": "it",

    "SUNPHARMA.NS": "pharma",   "DRREDDY.NS": "pharma",
    "CIPLA.NS": "pharma",       "DIVISLAB.NS": "pharma",
    "APOLLOHOSP.NS": "pharma",

    "MARUTI.NS": "auto",        "M&M.NS": "auto",
    "EICHERMOT.NS": "auto",     "BAJAJ-AUTO.NS": "auto",
    "HEROMOTOCO.NS": "auto",

    "JSWSTEEL.NS": "metal",     "TATASTEEL.NS": "metal",
    "HINDALCO.NS": "metal",     "VEDL.NS": "metal",

    "RELIANCE.NS": "energy",    "ONGC.NS": "energy",
    "BPCL.NS": "energy",        "NTPC.NS": "energy",
    "POWERGRID.NS": "energy",   "COALINDIA.NS": "energy",

    "LT.NS": "infra",           "ADANIPORTS.NS": "infra",
    "ADANIENT.NS": "infra",     "BEL.NS": "infra",

    "HINDUNILVR.NS": "fmcg",    "ITC.NS": "fmcg",
    "NESTLEIND.NS": "fmcg",     "BRITANNIA.NS": "fmcg",
    "TATACONSUM.NS": "fmcg",    "TRENT.NS": "fmcg",

    "ASIANPAINT.NS": "fmcg",    "TITAN.NS": "fmcg",
    "BHARTIARTL.NS": "infra",   "GRASIM.NS": "infra",
    "ULTRACEMCO.NS": "infra",
}

# Map ticker → market-cap category
TICKER_MARKET_CAP = {t: "largecap" for t in NIFTY50_TICKERS}

# ── Nifty Midcap 150 universe ─────────────────────────────────────────────
_MIDCAP150_FALLBACK = [
    "ABCAPITAL.NS","ABFRL.NS","AIAENG.NS","ALKEM.NS","AMBUJACEM.NS",
    "APLAPOLLO.NS","ASTRAL.NS","ATUL.NS","AUBANK.NS","BALKRISIND.NS",
    "BANKINDIA.NS","BATAINDIA.NS","BERGEPAINT.NS","BHARATFORG.NS","BHEL.NS",
    "BIOCON.NS","BOSCHLTD.NS","CANBK.NS","CANFINHOME.NS","CASTROLIND.NS",
    "CEATLTD.NS","CENTRALBK.NS","CESC.NS","CHOLAFIN.NS","COFORGE.NS",
    "COLPAL.NS","CONCOR.NS","COROMANDEL.NS","CROMPTON.NS","CUMMINSIND.NS",
    "DABUR.NS","DALBHARAT.NS","DEEPAKNTR.NS","DELTAMAGNT.NS","DIXON.NS",
    "ESCORTS.NS","EXIDEIND.NS","FEDERALBNK.NS","GAIL.NS","GMRAIRPORT.NS",
    "GODREJCP.NS","GODREJPROP.NS","GRANULES.NS","GSPL.NS","HAVELLS.NS",
    "HFCL.NS","HONAUT.NS","IDFCFIRSTB.NS","IGL.NS","INDUSTOWER.NS",
    "INOXWIND.NS","IOC.NS","IPCALAB.NS","IRCTC.NS","JKCEMENT.NS",
    "JSL.NS","JUBLFOOD.NS","KAJARIACER.NS","KEC.NS","KPITTECH.NS",
    "LTF.NS","LALPATHLAB.NS","LICHSGFIN.NS","LUPIN.NS","MARICO.NS",
    "UNITDSPR.NS","MCX.NS","MFSL.NS","MOTHERSON.NS","MPHASIS.NS",
    "MRF.NS","MUTHOOTFIN.NS","NAUKRI.NS","NBCC.NS","NCC.NS",
    "NMDC.NS","OBEROIRLTY.NS","OFSS.NS","OIL.NS","PAGEIND.NS",
    "PERSISTENT.NS","PETRONET.NS","PFIZER.NS","PIDILITIND.NS","PIIND.NS",
    "PNB.NS","POLYCAB.NS","PRINCEPIPE.NS","RAMCOCEM.NS","RBLBANK.NS",
    "RECLTD.NS","SAIL.NS","SCHAEFFLER.NS","SFL.NS","SIEMENS.NS",
    "SKFINDIA.NS","SRF.NS","STARHEALTH.NS","SUMICHEM.NS","SUNTV.NS",
    "SUPREMEIND.NS","SYMPHONY.NS","TATACOMM.NS","TATACHEM.NS","TATAELXSI.NS",
    "TATAPOWER.NS","TIINDIA.NS","TORNTPHARM.NS","TORNTPOWER.NS","TVSMOTOR.NS",
    "UBL.NS","UJJIVANSFB.NS","UNIONBANK.NS","UPL.NS","VOLTAS.NS",
    "WHIRLPOOL.NS","YESBANK.NS","ZEEL.NS","ZYDUSLIFE.NS",
    "AAVAS.NS","AFFLE.NS","ANGELONE.NS","APTUS.NS","ARVINDFASN.NS",
    "ASAHIINDIA.NS","BSOFT.NS","CARBORUNIV.NS","CDSL.NS","CLEAN.NS",
    "DATAMATICS.NS","DELHIVERY.NS","DOMS.NS","EPIGRAL.NS","FINEORG.NS",
    "FINPIPE.NS","GLAND.NS","GRINDWELL.NS","HOMEFIRST.NS","IONEXCHANG.NS",
    "ICICIGI.NS","ITI.NS","JYOTHYLAB.NS","KALYANKJIL.NS","KFINTECH.NS",
    "KIMS.NS","LATENTVIEW.NS","MAPMYINDIA.NS","METROPOLIS.NS","NAZARA.NS",
    "NETWORK18.NS","NLCINDIA.NS","NYKAA.NS","PAYTM.NS",
]

NIFTY_MIDCAP150_TICKERS = _fetch_nse_tickers("midcap150") or _MIDCAP150_FALLBACK
logger.info(f"Midcap150 universe: {len(NIFTY_MIDCAP150_TICKERS)} tickers")

TICKER_SECTOR_MIDCAP = {
    "BANKINDIA.NS":"banking","CANBK.NS":"banking","CANFINHOME.NS":"banking",
    "CENTRALBK.NS":"banking","CHOLAFIN.NS":"banking","FEDERALBNK.NS":"banking",
    "IDFCFIRSTB.NS":"banking","LTF.NS":"banking","LICHSGFIN.NS":"banking",
    "MUTHOOTFIN.NS":"banking","PNB.NS":"banking","RBLBANK.NS":"banking",
    "RECLTD.NS":"banking","UJJIVANSFB.NS":"banking","UNIONBANK.NS":"banking",
    "ABCAPITAL.NS":"banking","ANGELONE.NS":"banking","ICICIGI.NS":"banking",
    "KFINTECH.NS":"banking","STARHEALTH.NS":"banking","MFSL.NS":"banking",

    "COFORGE.NS":"it","KPITTECH.NS":"it","MPHASIS.NS":"it",
    "OFSS.NS":"it","PERSISTENT.NS":"it","TATAELXSI.NS":"it",
    "BSOFT.NS":"it","LATENTVIEW.NS":"it","MAPMYINDIA.NS":"it",

    "ALKEM.NS":"pharma","BIOCON.NS":"pharma","GRANULES.NS":"pharma",
    "IPCALAB.NS":"pharma","LALPATHLAB.NS":"pharma","LUPIN.NS":"pharma",
    "PFIZER.NS":"pharma","TORNTPHARM.NS":"pharma","ZYDUSLIFE.NS":"pharma",
    "GLAND.NS":"pharma","METROPOLIS.NS":"pharma",

    "BALKRISIND.NS":"auto","BHARATFORG.NS":"auto","CEATLTD.NS":"auto",
    "ESCORTS.NS":"auto","MOTHERSON.NS":"auto","MRF.NS":"auto",
    "TVSMOTOR.NS":"auto","TIINDIA.NS":"auto","SCHAEFFLER.NS":"auto",

    "AIAENG.NS":"metal","JSPL.NS":"metal","NMDC.NS":"metal",
    "SAIL.NS":"metal","JSL.NS":"metal",

    "GAIL.NS":"energy","IOC.NS":"energy","OIL.NS":"energy",
    "TATAPOWER.NS":"energy","CESC.NS":"energy","GSPL.NS":"energy",
    "IGL.NS":"energy","PETRONET.NS":"energy","INOXWIND.NS":"energy",

    "GMRAIRPORT.NS":"infra","KEC.NS":"infra","NBCC.NS":"infra",
    "NCC.NS":"infra","CONCOR.NS":"infra","INDUSTOWER.NS":"infra",
    "HFCL.NS":"infra","BHEL.NS":"infra","ITI.NS":"infra","NLCINDIA.NS":"infra",

    "ABFRL.NS":"fmcg","BATAINDIA.NS":"fmcg","COLPAL.NS":"fmcg",
    "CROMPTON.NS":"fmcg","DABUR.NS":"fmcg","GODREJCP.NS":"fmcg",
    "HAVELLS.NS":"fmcg","JUBLFOOD.NS":"fmcg","MARICO.NS":"fmcg",
    "UNITDSPR.NS":"fmcg","PAGEIND.NS":"fmcg","SYMPHONY.NS":"fmcg",
    "UBL.NS":"fmcg","WHIRLPOOL.NS":"fmcg","JYOTHYLAB.NS":"fmcg",
    "KALYANKJIL.NS":"fmcg","NYKAA.NS":"fmcg",

    "AMBUJACEM.NS":"infra","APLAPOLLO.NS":"metal","ASTRAL.NS":"infra",
    "BERGEPAINT.NS":"fmcg","BOSCHLTD.NS":"auto","CASTROLIND.NS":"energy",
    "COROMANDEL.NS":"fmcg","CUMMINSIND.NS":"infra","DALBHARAT.NS":"infra",
    "DEEPAKNTR.NS":"pharma","DIXON.NS":"it","EXIDEIND.NS":"auto",
    "GODREJPROP.NS":"infra","KAJARIACER.NS":"infra","MCX.NS":"banking",
    "OBEROIRLTY.NS":"infra","PIDILITIND.NS":"fmcg","PIIND.NS":"pharma",
    "POLYCAB.NS":"infra","RAMCOCEM.NS":"infra","SIEMENS.NS":"infra",
    "SKFINDIA.NS":"auto","SRF.NS":"pharma","SUNTV.NS":"fmcg",
    "SUPREMEIND.NS":"infra","TATACOMM.NS":"it","TATACHEM.NS":"pharma",
    "TORNTPOWER.NS":"energy","ATUL.NS":"pharma","AUBANK.NS":"banking",
    "DELTAMAGNT.NS":"fmcg","SUMICHEM.NS":"pharma","VOLTAS.NS":"fmcg",
    "ZEEL.NS":"fmcg","DELHIVERY.NS":"infra","NAZARA.NS":"it",
    "NETWORK18.NS":"fmcg","PAYTM.NS":"it","CDSL.NS":"banking",
    "CLEAN.NS":"energy","FINEORG.NS":"pharma","GRINDWELL.NS":"metal",
    "CARBORUNIV.NS":"metal","DOMS.NS":"fmcg","EPIGRAL.NS":"pharma",
    "FINPIPE.NS":"infra","HOMEFIRST.NS":"banking","IONEXCHANG.NS":"pharma",
    "KIMS.NS":"pharma","AAVAS.NS":"banking","AFFLE.NS":"it",
    "APTUS.NS":"banking","ARVINDFASN.NS":"fmcg","ASAHIINDIA.NS":"auto",
    "DATAMATICS.NS":"it","GSPL.NS":"energy","PRINCEPIPE.NS":"infra",
    "SCHAEFFLER.NS":"auto","SFL.NS":"metal","YESBANK.NS":"banking",
    "ICICIGI.NS":"banking","NAUKRI.NS":"it","IRCTC.NS":"infra",
    "UNITDSPR.NS":"fmcg","HONAUT.NS":"auto","PBFINTECH.NS":"it",
}

TICKER_MARKET_CAP_MIDCAP = {t: "midcap" for t in NIFTY_MIDCAP150_TICKERS}



# ═══════════════════════════════════════════════════════════════════════════
#  CACHE LAYER — Disk-based, process-safe
#
#  ROOT CAUSE of all previous failures:
#    gunicorn sync worker FORKS a child process to serve HTTP.
#    In-memory dicts updated by background threads are INVISIBLE to the fork.
#    Trigger files had a race condition: "if not exists" prevents re-queuing
#    when the file already exists from a previous request that stalled.
#
#  PERMANENT ARCHITECTURE:
#    1. Background thread (parent) runs scans, writes JSON results to DISK.
#    2. HTTP routes (child) read results from DISK — works across fork.
#    3. Midcap requests: HTTP child writes trigger file to disk.
#       Scanner worker (parent) polls trigger file every 10s → enqueues job.
#    4. Trigger file is cleared ONLY after scan completes successfully.
#    5. Queue used inside parent process — single worker, no OOM.
#    6. Stale trigger files cleaned on startup.
# ═══════════════════════════════════════════════════════════════════════════

import time as _time
import queue as _queue

SCAN_INTERVAL_SECONDS = int(os.environ.get("SCAN_INTERVAL_SECONDS", "300"))

_BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
_CACHE_FILE          = os.path.join(_BASE_DIR, "swingbull_scan_cache.json")
_MIDCAP_CACHE_FILE   = os.path.join(_BASE_DIR, "swingbull_midcap_cache.json")
_MIDCAP_TRIGGER_FILE = os.path.join(_BASE_DIR, "scan_midcap.trigger")
_SECTORS_CACHE_FILE  = os.path.join(_BASE_DIR, "swingbull_sectors_cache.json")
_SCAN_LOCK           = threading.Lock()
_SCAN_QUEUE          = _queue.Queue()
_ACTIVE_JOB          = None   # name of currently running scan job

def _read_cache(filepath: str) -> dict:
    """Read scan results from disk. Returns warming dict if missing/stale."""
    try:
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
            if _time.time() - data.get("_saved_at", 0) < 1200:
                return data
    except Exception as e:
        logger.warning(f"Cache read error {filepath}: {e}")
    return {"results": [], "status": "warming", "last_updated": None, "count": 0, "scanned": 0}

def _write_cache(filepath: str, data: dict):
    """Write scan results atomically — no partial reads possible."""
    try:
        to_save = _safe_convert(dict(data))
        to_save["_saved_at"] = _time.time()
        tmp = filepath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(to_save, f, allow_nan=False)
        os.replace(tmp, filepath)
        logger.info(f"Cache written: {os.path.basename(filepath)} ({data.get('count', 0)} results)")
    except Exception as e:
        logger.warning(f"Cache write error {filepath}: {e}")

_MARKET_CACHE: dict = {"data": None, "status": "warming", "last_updated": None}

def _scanner_worker():
    """Single worker — processes scan jobs one at a time. No OOM."""
    global _ACTIVE_JOB
    logger.info("Scanner worker started")
    while True:
        try:
            job = _SCAN_QUEUE.get(timeout=10)
            _ACTIVE_JOB = job
            try:
                if job == "nifty50":
                    logger.info("Scanner: running Nifty50...")
                    results = engine.scan_stocks_public(
                        tickers=NIFTY50_TICKERS,
                        sector_map=TICKER_SECTOR,
                        market_cap_map=TICKER_MARKET_CAP,
                    )
                    _write_cache(_CACHE_FILE, {
                        "results": results, "status": "ready",
                        "last_updated": datetime.utcnow().isoformat() + "Z",
                        "count": len(results), "scanned": len(NIFTY50_TICKERS),
                    })
                    logger.info(f"Scanner: Nifty50 done — {len(results)} setups")

                elif job == "midcap":
                    logger.info("Scanner: running Midcap150...")
                    results = engine.scan_stocks_public(
                        tickers=NIFTY_MIDCAP150_TICKERS,
                        sector_map=TICKER_SECTOR_MIDCAP,
                        market_cap_map=TICKER_MARKET_CAP_MIDCAP,
                    )
                    _write_cache(_MIDCAP_CACHE_FILE, {
                        "results": results, "status": "ready",
                        "last_updated": datetime.utcnow().isoformat() + "Z",
                        "count": len(results), "scanned": len(NIFTY_MIDCAP150_TICKERS),
                    })
                    logger.info(f"Scanner: Midcap150 done — {len(results)} setups")
                    # Clear trigger AFTER successful write
                    try:
                        os.remove(_MIDCAP_TRIGGER_FILE)
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"Scanner error (job={job}): {e}")
            finally:
                _ACTIVE_JOB = None
                _SCAN_QUEUE.task_done()

        except _queue.Empty:
            # Poll trigger file — written by HTTP child process
            if os.path.exists(_MIDCAP_TRIGGER_FILE):
                if "midcap" not in list(_SCAN_QUEUE.queue) and _ACTIVE_JOB != "midcap":
                    _SCAN_QUEUE.put("midcap")
                    logger.info("Midcap trigger detected — scan enqueued")


def _nifty50_scheduler():
    """Enqueues nifty50 every 5 min. Never runs if midcap is pending."""
    while True:
        _time.sleep(SCAN_INTERVAL_SECONDS)
        midcap_active  = _ACTIVE_JOB == "midcap"
        midcap_queued  = "midcap" in list(_SCAN_QUEUE.queue)
        midcap_trigger = os.path.exists(_MIDCAP_TRIGGER_FILE)
        nifty_queued   = "nifty50" in list(_SCAN_QUEUE.queue)
        if not midcap_active and not midcap_queued and not midcap_trigger and not nifty_queued:
            _SCAN_QUEUE.put("nifty50")
            logger.info("Scheduler: nifty50 enqueued")
        elif midcap_active or midcap_queued or midcap_trigger:
            logger.info("Scheduler: skipping nifty50 — midcap pending")


def _market_regime_worker():
    """Fetches market regime every 5 min. In-memory — parent process only."""
    global _MARKET_CACHE
    _time.sleep(90)
    while True:
        try:
            logger.info("Market regime: fetching...")
            data = engine.get_market_regime_public()
            with _SCAN_LOCK:
                _MARKET_CACHE = {
                    "data": data, "status": "ready",
                    "last_updated": datetime.utcnow().isoformat() + "Z",
                }
            logger.info("Market regime: done")
        except Exception as e:
            logger.error(f"Market regime error: {e}")
        _time.sleep(SCAN_INTERVAL_SECONDS)


def _sectors_worker():
    """Fetches sector rankings in background every 30 min. Writes to disk."""
    _time.sleep(60)  # wait for startup to settle
    while True:
        try:
            logger.info("Sectors: fetching...")
            data = SectorLeadershipEngine.analyze_sectors()
            if data:
                _write_cache(_SECTORS_CACHE_FILE, {
                    "sectors": data,
                    "status": "ready",
                    "last_updated": datetime.utcnow().isoformat() + "Z",
                })
                logger.info(f"Sectors: done — {len(data)} sectors")
        except Exception as e:
            logger.error(f"Sectors error: {e}")
        _time.sleep(1800)  # refresh every 30 min


def _start_thread(target, name):
    def wrapper():
        while True:
            try:
                target()
            except Exception as e:
                logger.error(f"Thread '{name}' crashed: {e} — restarting in 5s")
                _time.sleep(5)
    threading.Thread(target=wrapper, daemon=True, name=name).start()

# Clean up any stale trigger files from previous deploys
try:
    if os.path.exists(_MIDCAP_TRIGGER_FILE):
        os.remove(_MIDCAP_TRIGGER_FILE)
        logger.info("Cleared stale midcap trigger from previous session")
except Exception:
    pass

_start_thread(_scanner_worker,       "scanner-worker")
_start_thread(_nifty50_scheduler,    "nifty50-scheduler")
_start_thread(_market_regime_worker, "market-regime")
_start_thread(_sectors_worker,       "sectors-worker")

_SCAN_QUEUE.put("nifty50")
logger.info("Background threads started. Nifty50 scan enqueued.")

# ═══════════════════════════════════════════════════════════════════════════
#  UTILITY
# ═══════════════════════════════════════════════════════════════════════════

def _err(msg: str, code: int = 500):
    logger.error(msg)
    return safe_jsonify({"error": msg, "dashboard_meta": DASHBOARD_META}), code


def _resolve_tickers(body: dict) -> list:
    """Return requested tickers or default Nifty50 universe."""
    tickers = body.get("tickers", [])
    if not tickers or not isinstance(tickers, list):
        return NIFTY50_TICKERS
    return [t if "." in t else f"{t}.NS" for t in tickers]


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════

# ── Health check ────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "HEAD"])
def root():
    return safe_jsonify({
        "status": "live",
        "engine": "SwingBullEngine v4",
        "name": DASHBOARD_META["branding"]["name"],
        "tagline": DASHBOARD_META["branding"]["tagline"],
        "version": DASHBOARD_META["version"],
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dashboard_meta": DASHBOARD_META,
    })


# ── Dashboard Meta (branding, colors) ───────────────────────────────────────
@app.route("/api/meta", methods=["GET"])
def dashboard_meta():
    """Returns branding, color palette, and UI configuration."""
    return jsonify(DASHBOARD_META)


# ── Market Regime + Market Pulse ────────────────────────────────────────────
@app.route("/api/market-regime", methods=["GET"])
def market_regime():
    """
    Returns pre-computed market regime from background thread cache.
    Responds instantly — no blocking yfinance calls on this request.
    """
    try:
        with _SCAN_LOCK:
            cache = dict(_MARKET_CACHE)

        if cache["status"] == "warming" or cache["data"] is None:
            return safe_jsonify({
                "status":  "warming",
                "message": "Market data loading — refresh in 60 seconds.",
                "dashboard_meta": DASHBOARD_META,
            })

        data = dict(cache["data"])
        data["last_updated"] = cache["last_updated"]
        return safe_jsonify(data)

    except Exception as e:
        logger.exception("market-regime error")
        return _err(str(e))


# ── Stock Scanner ────────────────────────────────────────────────────────────
@app.route("/api/scan", methods=["GET", "POST", "OPTIONS"])
def scan():
    """
    Returns pre-computed scan results from background thread cache.
    Responds instantly (<1ms) — no yfinance calls on this request.
    Background thread refreshes every SCAN_INTERVAL_SECONDS (default 5 min).
    """
    if request.method == "OPTIONS":
        return safe_jsonify({}), 200

    try:
        # Read from disk — the ONLY shared state between gunicorn processes.
        # Background thread writes here; HTTP worker reads here. No memory sharing needed.
        cache = _read_cache(_CACHE_FILE)

        return safe_jsonify({
            "status":       cache["status"],
            "message":      "Scanner warming up — auto-refreshing..." if cache["status"] == "warming" else None,
            "results":      cache.get("results", []),
            "count":        cache.get("count", 0),
            "scanned":      cache.get("scanned", len(NIFTY50_TICKERS)),
            "last_updated": cache.get("last_updated"),
            "timestamp":    datetime.utcnow().isoformat() + "Z",
            "dashboard_meta": DASHBOARD_META,
        })

    except Exception as e:
        logger.exception("scan error")
        return _err(str(e))


# ── Midcap Scanner ───────────────────────────────────────────────────────────
@app.route("/api/scan/midcap", methods=["GET", "POST", "OPTIONS"])
def scan_midcap():
    """
    Returns Midcap150 scan results from cache.
    If cache is empty, enqueues a scan job — processed by the single scanner worker.
    Never runs a scan inline — no OOM risk.
    """
    if request.method == "OPTIONS":
        return safe_jsonify({}), 200
    try:
        # Read from disk — same reason as /api/scan (gunicorn fork isolation)
        cache = _read_cache(_MIDCAP_CACHE_FILE)

        if cache["status"] == "ready":
            return safe_jsonify({
                "status":       "ready",
                "results":      cache.get("results", []),
                "count":        cache.get("count", 0),
                "scanned":      cache.get("scanned", len(NIFTY_MIDCAP150_TICKERS)),
                "last_updated": cache.get("last_updated"),
                "timestamp":    datetime.utcnow().isoformat() + "Z",
                "universe":     "midcap150",
                "dashboard_meta": DASHBOARD_META,
            })

        # Write trigger file — scanner worker polls this every 10s
        try:
            with open(_MIDCAP_TRIGGER_FILE, "w") as f:
                f.write(str(_time.time()))
            logger.info("Midcap trigger written — scan will start within 10s")
        except Exception as e:
            logger.warning(f"Trigger write error: {e}")

        wait_msg = "Midcap scan running — results ready in ~2 minutes."

        return safe_jsonify({
            "status":   "warming",
            "message":  wait_msg + " Auto-refreshing...",
            "results":  [],
            "count":    0,
            "scanned":  len(NIFTY_MIDCAP150_TICKERS),
            "universe": "midcap150",
            "dashboard_meta": DASHBOARD_META,
        })

    except Exception as e:
        logger.exception("scan/midcap error")
        return _err(str(e))


# ── Deep Single-Stock Analysis ───────────────────────────────────────────────
@app.route("/api/analyse", methods=["POST", "OPTIONS"])
def analyse():
    """
    Body:
      {
        "ticker":          "RELIANCE.NS",
        "sector":          "energy",          (optional)
        "market_cap_cat":  "largecap",        (optional)
        "days_to_earnings":30,                (optional)
        "beta":            1.1,               (optional)
        "auth_token":      "xxx"              (required for full auth path)
      }

    Returns full 7-layer analysis with portfolio constraints.
    """
    if request.method == "OPTIONS":
        return safe_jsonify({}), 200

    try:
        body       = request.get_json(silent=True) or {}
        ticker     = body.get("ticker", "").strip()
        if not ticker:
            return _err("Missing 'ticker'", 400)

        ticker = ticker if "." in ticker else f"{ticker}.NS"
        sector = body.get("sector", TICKER_SECTOR.get(ticker, ""))
        market_cap_cat = body.get("market_cap_cat", TICKER_MARKET_CAP.get(ticker, "midcap"))
        days_to_earnings = int(body.get("days_to_earnings", 999))
        beta = float(body.get("beta", 1.0))
        auth_token = body.get("auth_token", "")

        result = engine.full_analysis(
            ticker          = ticker,
            sector          = sector,
            market_cap_cat  = market_cap_cat,
            days_to_earnings= days_to_earnings,
            beta            = beta,
            auth_token      = auth_token,
        )

        return safe_jsonify(result)

    except PermissionError as e:
        return _err(str(e), 401)
    except Exception as e:
        logger.exception("analyse error")
        return _err(str(e))


# ── Sectors ─────────────────────────────────────────────────────────────────
@app.route("/api/sectors", methods=["GET"])
def sectors():
    """Returns sector RS rankings from disk cache — never blocks."""
    try:
        cache = _read_cache(_SECTORS_CACHE_FILE)
        if cache.get("status") == "ready" and cache.get("sectors"):
            return safe_jsonify({
                "sectors":      cache["sectors"],
                "last_updated": cache.get("last_updated"),
                "timestamp":    datetime.utcnow().isoformat() + "Z",
                "dashboard_meta": DASHBOARD_META,
            })
        return safe_jsonify({
            "sectors":  {},
            "status":   "warming",
            "message":  "Sector data loading — ready in ~2 minutes.",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "dashboard_meta": DASHBOARD_META,
        })
    except Exception as e:
        logger.exception("sectors error")
        return _err(str(e))


# ── Portfolio Validation ────────────────────────────────────────────────────
@app.route("/api/portfolio/validate", methods=["POST", "OPTIONS"])
def portfolio_validate():
    """
    Body:
      {
        "proposed_ticker":     "RELIANCE.NS",
        "proposed_sector":     "energy",
        "proposed_beta":       1.1,
        "proposed_capital_pct":15.0,
        "active_positions":    [{ticker, sector, beta, capital_pct}, ...],
        "portfolio_drawdown_pct": 0
      }

    Returns: {allowed, block_reason, violations, warnings, operating_mode}
    """
    if request.method == "OPTIONS":
        return safe_jsonify({}), 200

    try:
        body = request.get_json(silent=True) or {}

        result = PortfolioRiskEngine.validate_new_trade(
            proposed_ticker       = body.get("proposed_ticker", ""),
            proposed_sector       = body.get("proposed_sector", ""),
            proposed_beta         = float(body.get("proposed_beta", 1.0)),
            proposed_capital_pct  = float(body.get("proposed_capital_pct", 0)),
            active_positions      = body.get("active_positions", []),
            portfolio_current_drawdown_pct = float(body.get("portfolio_drawdown_pct", 0)),
        )

        # Also get operating mode
        breadth  = MarketBreadthEngine.calculate_breadth()
        vol_data = VolatilityRegimeEngine.detect_state()
        regime   = MarketRegimeEngine.detect_regime(breadth_data=breadth, volatility_data=vol_data)
        op_mode  = PortfolioRiskEngine.get_operating_mode(
            regime_type  = regime["type"],
            breadth_slope= breadth.get("breadth_slope", 0),
            vol_regime   = vol_data.get("vol_regime", "Neutral_Transitioning"),
        )

        result["operating_mode"] = op_mode
        result["dashboard_meta"] = DASHBOARD_META
        return safe_jsonify(result)

    except Exception as e:
        logger.exception("portfolio/validate error")
        return _err(str(e))


# ── Update Active Positions ─────────────────────────────────────────────────
@app.route("/api/portfolio/positions", methods=["POST", "OPTIONS"])
def update_positions():
    """
    Body:
      {
        "positions": [
          {"ticker": "RELIANCE.NS", "sector": "energy", "beta": 1.1, "capital_pct": 15.0},
          ...
        ]
      }

    Updates the engine's active positions for portfolio constraint checks.
    """
    if request.method == "OPTIONS":
        return safe_jsonify({}), 200

    try:
        body = request.get_json(silent=True) or {}
        positions = body.get("positions", [])

        engine.set_active_positions(positions)

        return safe_jsonify({
            "message": "Positions updated",
            "count": len(positions),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

    except Exception as e:
        logger.exception("portfolio/positions error")
        return _err(str(e))


# ── Record Trade Result ──────────────────────────────────────────────────────
@app.route("/api/trade/record", methods=["POST", "OPTIONS"])
def record_trade():
    """
    Body:
      {
        "setup_type": "Breakout",
        "r_result":   1.5,
        "won":        true,
        "auth_token": "xxx"
      }
    Records trade to expectancy tracker (requires auth).
    """
    if request.method == "OPTIONS":
        return safe_jsonify({}), 200

    try:
        body       = request.get_json(silent=True) or {}
        setup_type = body.get("setup_type", "")
        r_result   = float(body.get("r_result", 0))
        won        = bool(body.get("won", False))
        auth_token = body.get("auth_token", "")

        if not setup_type:
            return _err("Missing 'setup_type'", 400)

        engine.record_trade_result(
            setup_type = setup_type,
            r_result   = r_result,
            won        = won,
            auth_token = auth_token,
        )
        return safe_jsonify({
            "recorded": True,
            "setup_type": setup_type,
            "r": r_result,
            "won": won,
            "dashboard_meta": DASHBOARD_META,
        })

    except PermissionError as e:
        return _err(str(e), 401)
    except Exception as e:
        logger.exception("trade/record error")
        return _err(str(e))


# ── Performance / Expectancy Stats ───────────────────────────────────────────
@app.route("/api/performance", methods=["GET"])
def performance():
    """
    Query params:
      ?auth_token=xxx   (required)
    Returns per-setup expectancy, edge stability, confidence meter.
    """
    try:
        auth_token = request.args.get("auth_token", "")
        stats = engine.get_performance_stats(auth_token=auth_token)
        return safe_jsonify({
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "dashboard_meta": DASHBOARD_META,
        })
    except PermissionError as e:
        return _err(str(e), 401)
    except Exception as e:
        logger.exception("performance error")
        return _err(str(e))


# ── Active Trade Health Check ────────────────────────────────────────────────
@app.route("/api/active-trade", methods=["POST", "OPTIONS"])
def active_trade():
    """
    Body:
      {
        "ticker":       "RELIANCE.NS",
        "entry_price":  2800,
        "stop_loss":    2720,
        "target":       3000,
        "setup_type":   "Breakout",
        "entry_date":   "2026-02-01"
      }
    """
    if request.method == "OPTIONS":
        return safe_jsonify({}), 200

    try:
        body       = request.get_json(silent=True) or {}
        ticker     = body.get("ticker", "").strip()
        if not ticker:
            return _err("Missing 'ticker'", 400)

        ticker = ticker if "." in ticker else f"{ticker}.NS"
        entry_price = float(body.get("entry_price", 0))
        stop_loss   = float(body.get("stop_loss", 0))
        target      = float(body.get("target", 0))
        setup_type_str = body.get("setup_type", "Momentum")
        entry_date_str = body.get("entry_date", "")

        # Parse setup type
        setup_map = {s.value: s for s in SetupType}
        setup_type = setup_map.get(setup_type_str, SetupType.MOMENTUM)

        # Parse entry date
        try:
            entry_date = datetime.strptime(entry_date_str, "%Y-%m-%d")
        except ValueError:
            entry_date = datetime.utcnow()

        result = ActiveTradeEvaluator.evaluate_trade(
            ticker      = ticker,
            entry_price = entry_price,
            stop_loss   = stop_loss,
            target      = target,
            setup_type  = setup_type,
            entry_date  = entry_date,
        )
        result["dashboard_meta"] = DASHBOARD_META
        return safe_jsonify(result)

    except Exception as e:
        logger.exception("active-trade error")
        return _err(str(e))


# ── Login ────────────────────────────────────────────────────────────────────
@app.route("/api/login", methods=["POST", "OPTIONS"])
def login():
    """
    Body: { "username": "freddy", "password": "your_password" }
    Returns: { "token": "xxx" }  — store in frontend sessionStorage.
    """
    if request.method == "OPTIONS":
        return safe_jsonify({}), 200

    try:
        body     = request.get_json(silent=True) or {}
        username = body.get("username", "")
        password = body.get("password", "")
        ip       = request.remote_addr or "127.0.0.1"

        token = engine.login(username, password, ip)
        if token:
            return safe_jsonify({
                "token": token,
                "message": "Login successful",
                "dashboard_meta": DASHBOARD_META,
            })
        return safe_jsonify({"error": "Invalid credentials or IP blocked"}), 401

    except Exception as e:
        logger.exception("login error")
        return _err(str(e))


# ── Logout ───────────────────────────────────────────────────────────────────
@app.route("/api/logout", methods=["POST", "OPTIONS"])
def logout():
    """Body: { "token": "xxx" }"""
    if request.method == "OPTIONS":
        return safe_jsonify({}), 200

    try:
        body  = request.get_json(silent=True) or {}
        token = body.get("token", "")
        engine.logout(token)
        return safe_jsonify({"message": "Logged out"})
    except Exception as e:
        return _err(str(e))


# ═══════════════════════════════════════════════════════════════════════════
#  ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    return safe_jsonify({
        "error": "Endpoint not found",
        "available": [
            "GET  /",
            "GET  /api/meta",
            "GET  /api/market-regime",
            "POST /api/scan",
            "POST /api/analyse",
            "GET  /api/sectors",
            "POST /api/portfolio/validate",
            "POST /api/portfolio/positions",
            "POST /api/trade/record",
            "GET  /api/performance?auth_token=xxx",
            "POST /api/active-trade",
            "POST /api/login",
            "POST /api/logout",
        ],
        "dashboard_meta": DASHBOARD_META,
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return safe_jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(e):
    logger.exception("Unhandled 500")
    return safe_jsonify({
        "error": "Internal server error",
        "dashboard_meta": DASHBOARD_META,
    }), 500


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting Swing Bull Trader API on port {port} | debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)
