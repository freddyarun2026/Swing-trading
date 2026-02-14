"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   FREDDY API SERVER v3.0 — Flask Backend for Render                        ║
║   Connects trading_engine_v3.py to the Netlify frontend                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

ENDPOINTS:
  GET  /                          → health check
  GET  /api/market-regime         → Freddy Gauge + breadth + sectors
  POST /api/scan                  → stock scanner (body: {tickers: [...]} optional)
  POST /api/analyse               → deep single-stock analysis (body: {ticker, sector, ...})
  GET  /api/sectors               → sector RS rankings
  POST /api/portfolio/validate    → portfolio-level risk check
  POST /api/trade/record          → record a completed trade result
  GET  /api/performance           → expectancy & edge stability stats
  POST /api/active-trade          → evaluate an open position health
  POST /api/login                 → get session token
  POST /api/logout                → end session
"""

import os
import logging
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── Import the trading engine ──────────────────────────────────────────────
from trading_engine_v3 import (
    FreddyEngine,
    SetupClassifier,
    SetupType,
    PortfolioRiskEngine,
    ActiveTradeEvaluator,
    MarketRegimeEngine,
    MarketBreadthEngine,
    VolatilityRegimeEngine,
    SectorLeadershipEngine,
    RegimeType,
)

# ═══════════════════════════════════════════════════════════════════════════
#  APP SETUP
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("freddy.api")

app = Flask(__name__)

# ── CORS: allow only your Netlify frontend ─────────────────────────────────
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
RISK_PER_TRADE   = float(os.environ.get("RISK_PER_TRADE",   "2.0"))   # % per trade

engine = FreddyEngine(
    total_capital=TOTAL_CAPITAL,
    risk_per_trade=RISK_PER_TRADE,
)
logger.info(f"FreddyEngine v3 initialised — capital ₹{TOTAL_CAPITAL:,.0f}, risk {RISK_PER_TRADE}%/trade")


# ═══════════════════════════════════════════════════════════════════════════
#  STOCK UNIVERSE  (Nifty 50 default + sector mapping)
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
TICKER_MARKET_CAP = {t: "largecap" for t in NIFTY50_TICKERS}   # All Nifty50 are largecap


# ═══════════════════════════════════════════════════════════════════════════
#  UTILITY
# ═══════════════════════════════════════════════════════════════════════════

def _err(msg: str, code: int = 500):
    logger.error(msg)
    return jsonify({"error": msg}), code


def _resolve_tickers(body: dict) -> list:
    """Return requested tickers or default Nifty50 universe."""
    tickers = body.get("tickers", [])
    if not tickers or not isinstance(tickers, list):
        return NIFTY50_TICKERS
    # Ensure .NS suffix for NSE tickers
    return [t if "." in t else f"{t}.NS" for t in tickers]


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════

# ── Health check ────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "HEAD"])
def root():
    return jsonify({
        "status": "live",
        "engine": "FreddyEngine v3",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


# ── Market Regime + Freddy Gauge ────────────────────────────────────────────
@app.route("/api/market-regime", methods=["GET"])
def market_regime():
    """
    Returns: Freddy Gauge (MPI, degrees, regime label),
             breadth data, volatility state, sector RS rankings.
    No authentication required — read-only market data.
    """
    try:
        data = engine.get_market_regime_public()
        return jsonify(data)
    except Exception as e:
        logger.exception("market-regime error")
        return _err(str(e))


# ── Stock Scanner ────────────────────────────────────────────────────────────
@app.route("/api/scan", methods=["POST", "OPTIONS"])
def scan():
    """
    Body (optional):
      { "tickers": ["RELIANCE.NS", "TCS.NS", ...] }

    Returns list of setups sorted by confidence (best first).
    No authentication required.
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        body    = request.get_json(silent=True) or {}
        tickers = _resolve_tickers(body)

        results = engine.scan_stocks_public(
            tickers       = tickers,
            sector_map    = TICKER_SECTOR,
            market_cap_map= TICKER_MARKET_CAP,
        )

        return jsonify({
            "results": results,
            "count":   len(results),
            "scanned": len(tickers),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

    except Exception as e:
        logger.exception("scan error")
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

    Returns: full 6-layer analysis + playbook.
    Falls back to classify_setup (no auth) if no token provided.
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        body    = request.get_json(silent=True) or {}
        ticker  = body.get("ticker", "").strip()
        if not ticker:
            return _err("Missing 'ticker' in request body", 400)

        ticker       = ticker if "." in ticker else f"{ticker}.NS"
        sector       = body.get("sector", TICKER_SECTOR.get(ticker, ""))
        cap_cat      = body.get("market_cap_cat", TICKER_MARKET_CAP.get(ticker, "midcap"))
        days_earn    = int(body.get("days_to_earnings", 999))
        beta         = float(body.get("beta", 1.0))
        auth_token   = body.get("auth_token", "")

        # Attempt authenticated full analysis; fall back to public scan
        if auth_token:
            result = engine.full_analysis(
                ticker           = ticker,
                sector           = sector,
                market_cap_cat   = cap_cat,
                days_to_earnings = days_earn,
                beta             = beta,
                auth_token       = auth_token,
            )
        else:
            # Public-mode: return classification without auth gate
            import yfinance as yf
            import pandas_ta as ta
            df = yf.download(ticker, period="6mo", progress=False)
            if df.empty:
                return _err(f"No market data for {ticker}", 404)

            sector_data = engine._get_sector_data()
            classification = SetupClassifier.classify_setup(
                df, ticker, sector, cap_cat, days_earn, beta, sector_data
            )
            result = {
                "ticker":     ticker,
                "timestamp":  datetime.utcnow().isoformat() + "Z",
                "mode":       "public (no auth)",
                "setup_type": classification["setup_type"].value,
                "tier":       classification["probability_tier"].value,
                "status":     classification["status"].value,
                "confidence": classification["confidence"],
                "trend_context": classification.get("trend_context", "Mid_Trend"),
                "hard_kill_conditions": classification.get("hard_kill_conditions", []),
                "rsi":        classification.get("rsi", {}),
                "rs":         classification.get("rs", {}),
                "gap_risk":   classification.get("gap_risk", {}),
                "liquidity":  classification.get("liquidity", {}),
                "breakout_confirmation": classification.get("breakout_confirmation", {}),
            }

        return jsonify(result)

    except PermissionError as e:
        return _err(str(e), 401)
    except Exception as e:
        logger.exception(f"analyse error for {body.get('ticker')}")
        return _err(str(e))


# ── Sector Rankings ──────────────────────────────────────────────────────────
@app.route("/api/sectors", methods=["GET"])
def sectors():
    """Returns sector RS rankings with concentration index."""
    try:
        data = SectorLeadershipEngine.analyze_sectors()
        return jsonify({"sectors": data, "timestamp": datetime.utcnow().isoformat() + "Z"})
    except Exception as e:
        logger.exception("sectors error")
        return _err(str(e))


# ── Portfolio Risk Validator ─────────────────────────────────────────────────
@app.route("/api/portfolio/validate", methods=["POST", "OPTIONS"])
def portfolio_validate():
    """
    Body:
      {
        "proposed_ticker":    "TITAN.NS",
        "proposed_sector":    "fmcg",
        "proposed_beta":      0.9,
        "proposed_capital_pct": 8.5,
        "active_positions":   [
          {"ticker":"RELIANCE.NS","sector":"energy","beta":1.1,"capital_pct":10},
          {"ticker":"TCS.NS","sector":"it","beta":0.85,"capital_pct":9}
        ],
        "portfolio_drawdown_pct": 2.5
      }
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

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
        return jsonify(result)

    except Exception as e:
        logger.exception("portfolio/validate error")
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
        return jsonify({}), 200

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
        return jsonify({"recorded": True, "setup_type": setup_type, "r": r_result, "won": won})

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
        return jsonify({"stats": stats, "timestamp": datetime.utcnow().isoformat() + "Z"})
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
        return jsonify({}), 200

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
        return jsonify(result)

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
        return jsonify({}), 200

    try:
        body     = request.get_json(silent=True) or {}
        username = body.get("username", "")
        password = body.get("password", "")
        ip       = request.remote_addr or "127.0.0.1"

        token = engine.login(username, password, ip)
        if token:
            return jsonify({"token": token, "message": "Login successful"})
        return jsonify({"error": "Invalid credentials or IP blocked"}), 401

    except Exception as e:
        logger.exception("login error")
        return _err(str(e))


# ── Logout ───────────────────────────────────────────────────────────────────
@app.route("/api/logout", methods=["POST", "OPTIONS"])
def logout():
    """Body: { "token": "xxx" }"""
    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        body  = request.get_json(silent=True) or {}
        token = body.get("token", "")
        engine.logout(token)
        return jsonify({"message": "Logged out"})
    except Exception as e:
        return _err(str(e))


# ═══════════════════════════════════════════════════════════════════════════
#  ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "available": [
        "GET  /",
        "GET  /api/market-regime",
        "POST /api/scan",
        "POST /api/analyse",
        "GET  /api/sectors",
        "POST /api/portfolio/validate",
        "POST /api/trade/record",
        "GET  /api/performance?auth_token=xxx",
        "POST /api/active-trade",
        "POST /api/login",
        "POST /api/logout",
    ]}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(e):
    logger.exception("Unhandled 500")
    return jsonify({"error": "Internal server error"}), 500


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting Freddy API on port {port} | debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)
