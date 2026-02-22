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
from flask_cors import CORS


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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("swingbull.api")

app = Flask(__name__)
app.json_encoder = NumpyEncoder          # ← use numpy-safe encoder globally

# ── CORS: allow your frontend origins ─────────────────────────────────────
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGINS",
        "https://swingtraderindia.netlify.app,http://localhost:3000,http://localhost:5500"
    ).split(",")
    if o.strip()
]

CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})

# ── Engine: initialised once at startup ────────────────────────────────────
TOTAL_CAPITAL    = float(os.environ.get("TOTAL_CAPITAL",    "1000000"))
RISK_PER_TRADE   = float(os.environ.get("RISK_PER_TRADE",   "2.0"))

engine = SwingBullEngine(
    total_capital=TOTAL_CAPITAL,
    risk_per_trade=RISK_PER_TRADE,
)
logger.info(f"SwingBullEngine v4 initialised — capital ₹{TOTAL_CAPITAL:,.0f}, risk {RISK_PER_TRADE}%/trade")


# ═══════════════════════════════════════════════════════════════════════════
#  STOCK UNIVERSE (Nifty 50 default + sector mapping)
# ═══════════════════════════════════════════════════════════════════════════

NIFTY50_TICKERS = [
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
NIFTY_MIDCAP150_TICKERS = [
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
    "ICICISecurities.NS","ITI.NS","JYOTHYLAB.NS","KALYANKJIL.NS","KFINTECH.NS",
    "KIMS.NS","LATENTVIEW.NS","MAPMYINDIA.NS","METROPOLIS.NS","NAZARA.NS",
    "NETWORK18.NS","NLCINDIA.NS","NYKAA.NS","PAYTM.NS",
]

TICKER_SECTOR_MIDCAP = {
    "BANKINDIA.NS":"banking","CANBK.NS":"banking","CANFINHOME.NS":"banking",
    "CENTRALBK.NS":"banking","CHOLAFIN.NS":"banking","FEDERALBNK.NS":"banking",
    "IDFCFIRSTB.NS":"banking","LTF.NS":"banking","LICHSGFIN.NS":"banking",
    "MUTHOOTFIN.NS":"banking","PNB.NS":"banking","RBLBANK.NS":"banking",
    "RECLTD.NS":"banking","UJJIVANSFB.NS":"banking","UNIONBANK.NS":"banking",
    "ABCAPITAL.NS":"banking","ANGELONE.NS":"banking","ICICISecurities.NS":"banking",
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
    "ICICISecurities.NS":"banking","NAUKRI.NS":"it","IRCTC.NS":"infra",
    "UNITDSPR.NS":"fmcg","HONAUT.NS":"auto","PBFINTECH.NS":"it",
}

TICKER_MARKET_CAP_MIDCAP = {t: "midcap" for t in NIFTY_MIDCAP150_TICKERS}



# ═══════════════════════════════════════════════════════════════════════════
#  BACKGROUND SCANNER
#  ROOT CAUSE FIX: /api/scan was running ALL yfinance downloads synchronously
#  inside the HTTP request. With 50 tickers × ~1s each = 50s+ response time,
#  which hits Render's 30s proxy timeout and returns nothing to the frontend.
#
#  Fix: scanner runs in a background thread every 5 minutes.
#  /api/scan endpoint now just returns the pre-computed cache instantly (<1ms).
# ═══════════════════════════════════════════════════════════════════════════

import time as _time

_SCAN_LOCK = threading.Lock()
SCAN_INTERVAL_SECONDS = int(os.environ.get("SCAN_INTERVAL_SECONDS", "300"))  # default 5 min

# ── Disk-backed cache — survives Render worker restarts ──────────────────────
_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swingbull_scan_cache.json")

def _load_disk_cache() -> dict:
    """Load scan results from disk if available and fresh (< 10 min old)."""
    try:
        if os.path.exists(_CACHE_FILE):
            with open(_CACHE_FILE, "r") as f:
                data = json.load(f)
            # Use disk cache only if it's less than 10 minutes old
            import time as _t
            age = _t.time() - data.get("_saved_at", 0)
            if age < 3600:  # use disk cache if less than 60 min old
                logger.info(f"Loaded scan cache from disk ({age:.0f}s old, {data.get('count',0)} results)")
                return data
    except Exception as e:
        logger.warning(f"Could not load disk cache: {e}")
    return {"results": [], "status": "warming", "last_updated": None, "count": 0, "scanned": 0}

def _save_disk_cache(cache: dict):
    """Persist scan results to disk — sanitise NaN/Inf before saving."""
    try:
        import time as _t
        to_save = _safe_convert(dict(cache))
        to_save["_saved_at"] = _t.time()
        # Write via safe_json string to guarantee valid JSON on disk
        raw = json.dumps(to_save, allow_nan=False)
        with open(_CACHE_FILE, "w") as f:
            f.write(raw)
        logger.info(f"Disk cache saved ({cache.get('count', 0)} results)")
    except Exception as e:
        logger.warning(f"Could not save disk cache: {e}")

# Initialise from disk on startup — instant results if recently deployed
_SCAN_CACHE: dict = _load_disk_cache()
_MARKET_CACHE: dict = {"data": None, "status": "warming", "last_updated": None}

# ── Midcap cache ──────────────────────────────────────────────────────────────
_MIDCAP_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swingbull_midcap_cache.json")

def _load_midcap_cache() -> dict:
    try:
        if os.path.exists(_MIDCAP_CACHE_FILE):
            with open(_MIDCAP_CACHE_FILE, "r") as f:
                data = json.load(f)
            age = _time.time() - data.get("_saved_at", 0)
            if age < 3600:
                logger.info(f"Loaded midcap cache from disk ({age:.0f}s old, {data.get('count',0)} results)")
                return data
    except Exception as e:
        logger.warning(f"Could not load midcap disk cache: {e}")
    return {"results": [], "status": "warming", "last_updated": None, "count": 0, "scanned": 0}

def _save_midcap_cache(cache: dict):
    try:
        to_save = _safe_convert(dict(cache))
        to_save["_saved_at"] = _time.time()
        raw = json.dumps(to_save, allow_nan=False)
        with open(_MIDCAP_CACHE_FILE, "w") as f:
            f.write(raw)
        logger.info(f"Midcap disk cache saved ({cache.get('count', 0)} results)")
    except Exception as e:
        logger.warning(f"Could not save midcap disk cache: {e}")

_MIDCAP_CACHE: dict = _load_midcap_cache()


def _run_background_scanner():
    """Runs the full stock scan in background. Updates _SCAN_CACHE when done."""
    global _SCAN_CACHE
    logger.info("Background scanner: starting first scan...")
    _first_run = True

    while True:
        try:
            logger.info("Background scanner: running scan...")
            results = engine.scan_stocks_public(
                tickers        = NIFTY50_TICKERS,
                sector_map     = TICKER_SECTOR,
                market_cap_map = TICKER_MARKET_CAP,
            )
            ts = datetime.utcnow().isoformat() + "Z"
            with _SCAN_LOCK:
                _SCAN_CACHE = {
                    "results":      results,
                    "status":       "ready",
                    "last_updated": ts,
                    "count":        len(results),
                    "scanned":      len(NIFTY50_TICKERS),
                }
            logger.info(f"Background scanner: done — {len(results)} setups found.")
            _save_disk_cache(_SCAN_CACHE)  # persist so restarts load instantly

        except Exception as e:
            logger.error(f"Background scanner error: {e}")
            with _SCAN_LOCK:
                _SCAN_CACHE["status"] = "error"

        _time.sleep(SCAN_INTERVAL_SECONDS)


def _run_background_midcap_scanner():
    """Runs midcap scan in background — staggered 3min after nifty50 to avoid OOM."""
    global _MIDCAP_CACHE
    logger.info("Background midcap scanner: starting (delayed 3min)...")
    _time.sleep(180)  # wait for nifty50 scan to finish first
    while True:
        try:
            logger.info("Background midcap scanner: running...")
            results = engine.scan_stocks_public(
                tickers        = NIFTY_MIDCAP150_TICKERS,
                sector_map     = TICKER_SECTOR_MIDCAP,
                market_cap_map = TICKER_MARKET_CAP_MIDCAP,
            )
            ts = datetime.utcnow().isoformat() + "Z"
            with _SCAN_LOCK:
                _MIDCAP_CACHE = {
                    "results":      results,
                    "status":       "ready",
                    "last_updated": ts,
                    "count":        len(results),
                    "scanned":      len(NIFTY_MIDCAP150_TICKERS),
                }
            logger.info(f"Background midcap scanner: done — {len(results)} setups found.")
            _save_midcap_cache(_MIDCAP_CACHE)
        except Exception as e:
            logger.error(f"Background midcap scanner error: {e}")
            with _SCAN_LOCK:
                _MIDCAP_CACHE["status"] = "error"
        _time.sleep(SCAN_INTERVAL_SECONDS)


def _run_background_market():
    """Runs market regime fetch in background — staggered 1.5min after startup."""
    global _MARKET_CACHE
    logger.info("Background market: starting (delayed 90s)...")
    _time.sleep(90)  # let nifty50 scanner start first
    while True:
        try:
            logger.info("Background market: fetching regime...")
            data = engine.get_market_regime_public()
            ts = datetime.utcnow().isoformat() + "Z"
            with _SCAN_LOCK:
                _MARKET_CACHE = {
                    "data":         data,
                    "status":       "ready",
                    "last_updated": ts,
                }
            logger.info("Background market: done.")
        except Exception as e:
            logger.error(f"Background market error: {e}")
            with _SCAN_LOCK:
                _MARKET_CACHE["status"] = "error"

        _time.sleep(SCAN_INTERVAL_SECONDS)


# NOTE: threads are launched AFTER ticker lists are defined — see below


# ── Keep-alive: prevents Render free tier from sleeping ──────────────────────
# Keep-alive: Cloudflare Worker cron handles warming now.
# Internal self-ping removed — it caused timeout warnings on Render.
logger.info("Keep-alive: handled by Cloudflare Worker cron")


# ── Launch background threads NOW — all ticker lists are defined above ────────
threading.Thread(target=_run_background_scanner,        daemon=True).start()
threading.Thread(target=_run_background_midcap_scanner, daemon=True).start()
threading.Thread(target=_run_background_market,         daemon=True).start()
logger.info("Background scanner + midcap + market threads launched")


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
        with _SCAN_LOCK:
            cache = dict(_SCAN_CACHE)

        # Always return whatever is in cache — even if warming.
        # Frontend gets "status: warming" with empty results on first boot,
        # then "status: ready" with full results once scanner completes (~40s).
        # Never block or return nothing — Render proxy kills requests > 30s.
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
    """Returns pre-computed Nifty Midcap 150 scan results from background cache."""
    if request.method == "OPTIONS":
        return safe_jsonify({}), 200
    try:
        with _SCAN_LOCK:
            cache = dict(_MIDCAP_CACHE)
        return safe_jsonify({
            "status":       cache["status"],
            "message":      "Midcap scanner warming up..." if cache["status"] == "warming" else None,
            "results":      cache.get("results", []),
            "count":        cache.get("count", 0),
            "scanned":      cache.get("scanned", len(NIFTY_MIDCAP150_TICKERS)),
            "last_updated": cache.get("last_updated"),
            "timestamp":    datetime.utcnow().isoformat() + "Z",
            "universe":     "midcap150",
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
    """Returns sector RS rankings and concentration index."""
    try:
        data = SectorLeadershipEngine.analyze_sectors()
        return safe_jsonify({
            "sectors": data,
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
