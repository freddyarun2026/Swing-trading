"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          SWING BULL TRADER™ v1.1 — Trading Engine                           ║
║          Architected by Freddy — Personal Use Only                          ║
║          Indian Market Calibrated • 3mo Data • Memory Optimized                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

LAYERED ARCHITECTURE:
  Layer 1: Market   (Regime + Breadth + India VIX Gate + Divergence Detection)
  Layer 2: Sector   (Leadership + RS Velocity + F&O Expiry Context)
  Layer 3: Stock    (6 Setup Types + India-Calibrated Filters + Delivery Quality)
  Layer 4: Risk     (ATR + Dynamic Gap + Weekly Trend + Multi-Day Confirmation)
  Layer 5: Capital  (VIX-Adjusted Sizing + Correlation Replacement)
  Layer 6: Evolution(Expectancy Tracking + Self-Protective Logic)
  Layer 7: Portfolio(Hard Blocks + Sector Caps + Operator Alert)
  Layer 8: Journal  (Auto-Journaling + NLP Reasoning + Global Cues Context)

VERSION 1.1 — INDIA-CALIBRATED (from v1.0):
  ✅ RECALIBRATED:
  🔧 1.  Breakout volume 2.5x → 1.5x  (Indian institutional buying is quieter)
  🔧 2.  Momentum RS 8.0 → 5.0  (realistic for NSE leaders)
  🔧 3.  Power Play RS 6.0 → 4.5  (same reason)
  🔧 4.  Tight Flag range 1.5% → 3.0%  (Indian midcap baseline volatility)
  🔧 5.  VCP window 15-day → 10-day  (faster Indian market cycles)
  🔧 6.  Intraday chase guard 3% → 5% large / 8% midcap
  🔧 7.  Morning spike distance 0.5% → 1.5% large / 2.5% midcap
  🔧 8.  Weekly volume 1.0x → 0.8x  (F&O expiry distorts weekly vol)
  🔧 9.  Gap limits 1.0% → 2.0% large / 3.5% midsmall
  🔧 10. Tier A+ multiplier raised to 1.5x (was 1.0x in code)

  ✅ NEW SETUP TYPES:
  🆕 11. 52-Week High Breakout — most reliable India breakout, triggers institutional momentum
  🆕 12. Results Momentum — post-earnings gap+hold continuation (>5% gap, holds 2+ days)

  ✅ NEW FILTER CLASSES:
  🆕 13. IndiaVIXEngine — VIX regime gate: >20 reduce size, >25 half size, >30 stop new trades
  🆕 14. FNOExpiryGuard — flags expiry week (Thu), reduces volume signal confidence
  🆕 15. DeliveryQualityEstimator — proxy delivery % from close position in range + consistency
  🆕 16. WeeklyTrendConfirmation — weekly EMA20 alignment + weekly RSI zone check
  🆕 17. MultiDayBreakoutConfirmation — 3-day close above resistance (reduces false breakouts)
  🆕 18. ResultsMomentumDetector — detects post-earnings gap patterns
  🆕 19. OperatorAlert — detects manipulation signals in mid/smallcap (thin vol + big moves)
  🆕 20. GlobalCuesEngine — Asian market context (Nikkei, Hang Seng, SGX proxy)
  🆕 21. ATHBreakoutDetector — all-time high breakout (strongest signal)
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
import hashlib
import hmac
import secrets
import time
import os
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from functools import wraps
import pytz

IST_TZ = pytz.timezone('Asia/Kolkata')

# ══════════════════════════════════════════════════════════════════════════════
#  RATE-LIMIT RESILIENT YFINANCE LAYER
#  Fixes: YFRateLimitError on Render cloud deployments.
#  Strategy: unified cache + exponential backoff + threads=False + request session
# ══════════════════════════════════════════════════════════════════════════════
import random
import threading

# Shared in-memory cache (key → (dataframe, timestamp))
_YF_CACHE: Dict[str, tuple] = {}
_YF_CACHE_LOCK = threading.Lock()

# Cache TTL constants (seconds)
_CACHE_TTL_SHORT  = 300   # 5 min  — for 3d period (global cues)
_CACHE_TTL_NORMAL = 900   # 15 min — for 3mo / 1y period
_CACHE_TTL_LONG   = 1800  # 30 min — for sector indices (slow-moving)

# NOTE: Do NOT pass a requests.Session to yfinance — newer versions require
# curl_cffi and will reject a plain requests.Session with an error.
# Let yfinance manage its own session internally.

def _yf_download(ticker: str, period: str = "3mo",
                 max_retries: int = 5, base_delay: float = 2.0,
                 ttl: int = None) -> pd.DataFrame:
    """
    Rate-limit resilient yfinance download with:
      • In-memory cache (TTL-based, thread-safe)
      • Exponential backoff + jitter on 429 / YFRateLimitError
      • threads=False  ← critical on Render (prevents parallel hammering)
      • Shared requests session with browser User-Agent
    """
    cache_key = f"{ticker}_{period}"

    # Determine TTL
    if ttl is None:
        ttl = _CACHE_TTL_SHORT if period in ("1d", "2d", "3d", "5d") else _CACHE_TTL_LONG if period in ("6mo", "1y", "2y") else _CACHE_TTL_NORMAL

    # --- Cache read ---
    with _YF_CACHE_LOCK:
        if cache_key in _YF_CACHE:
            df_cached, ts = _YF_CACHE[cache_key]
            if (time.time() - ts) < ttl:
                return df_cached

    # --- Fetch with retry ---
    last_exc = None
    for attempt in range(max_retries):
        try:
            df = yf.download(
                ticker,
                period=period,
                progress=False,
                threads=False,       # ← CRITICAL: never run parallel on Render
            )

            # ── Normalise DataFrame ──────────────────────────────────────────
            # yfinance sometimes returns MultiLevel columns like
            # ('Close', 'RELIANCE.NS') instead of plain 'Close'.
            # This causes NoneType / Series errors downstream. Flatten always.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            # Drop duplicate columns (e.g. two 'Close' after flattening)
            df = df.loc[:, ~df.columns.duplicated()]
            # Ensure standard OHLCV columns exist
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col not in df.columns:
                    df[col] = float('nan')
            df = df.dropna(subset=['Close'])
            # ────────────────────────────────────────────────────────────────

            # Store in cache
            with _YF_CACHE_LOCK:
                _YF_CACHE[cache_key] = (df, time.time())
            return df

        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()
            is_rate_limit = any(k in err_str for k in (
                "too many requests", "rate limit", "429",
                "yfratelimiterror", "rate limited"
            ))
            if is_rate_limit and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0.5, 2.5)
                logger.warning(
                    f"[yfinance] Rate limited on '{ticker}' "
                    f"(attempt {attempt+1}/{max_retries}). "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                # Non-rate-limit error — don't retry
                break

    logger.error(f"[yfinance] Failed to download '{ticker}' after {max_retries} attempts: {last_exc}")
    return pd.DataFrame()


def _get_benchmark_data(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """Download benchmark data (always uses shared cache + retry logic)."""
    return _yf_download(ticker, period=period)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("swingbull.engine")


# ═══════════════════════════════════════════════════════════════════════
#  DASHBOARD META & UI COLOR PALETTE (NEW in v4)
# ═══════════════════════════════════════════════════════════════════════

DASHBOARD_META = {
    "version": "1.1",
    "branding": {
        "name": "Swing Bull Trader",
        "tagline": "Architected by Freddy",
        "footer": "© Swing Bull Trader v1.1 — Architected by Freddy",
    },
    "color_palette": {
        "background": "#0F172A",       # Deep Gunmetal
        "text": "#E2E8F0",              # Off-White / Silver
        "bullish": "#00FF9D",           # Electric Green
        "bearish": "#FF0055",           # Radical Red
        "warning": "#FFBF00",           # Amber
        "accent": "#00F0FF",            # Neon Cyan
        "muted": "#64748B",             # Slate Gray
    },
    "status_colors": {
        "READY": "#00FF9D",
        "WATCH": "#FFBF00",
        "AVOID": "#FF0055",
        "EXPIRED": "#64748B",
        "BLOCKED": "#9333EA",           # Purple for blocked
    },
    "market_pulse_zones": {
        "defensive": {"min": 0, "max": 45, "color": "#FF0055", "label": "Defensive", "strategy": "Cash/Gold"},
        "selective": {"min": 45, "max": 135, "color": "#FFBF00", "label": "Selective", "strategy": "Stock Specific"},
        "aggressive": {"min": 135, "max": 180, "color": "#00FF9D", "label": "Aggressive", "strategy": "Leverage/Pyramiding"},
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  SECURITY MODULE — Personal Use Protection
# ═══════════════════════════════════════════════════════════════════════

class SecurityConfig:
    """Hardened security for personal-use-only deployment."""
    SECRET_KEY = os.environ.get("FREDDY_SECRET_KEY", secrets.token_hex(32))
    SESSION_TIMEOUT_MINUTES = 30
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 15
    RATE_LIMIT_PER_MINUTE = 60
    ALLOWED_IPS = [ip.strip() for ip in os.environ.get("FREDDY_ALLOWED_IPS", "127.0.0.1,::1").split(",")]
    OWNER_USERNAME = os.environ.get("FREDDY_USERNAME", "freddy")
    OWNER_PASSWORD_HASH = os.environ.get(
        "FREDDY_PASSWORD_HASH",
        hashlib.sha256("change_me_immediately".encode()).hexdigest()
    )


class AuthManager:
    """Session auth with brute-force protection and IP binding."""

    def __init__(self):
        self._sessions: Dict[str, dict] = {}
        self._login_attempts: Dict[str, list] = {}
        self._rate_limiter: Dict[str, list] = {}

    def _hash_password(self, password: str) -> str:
        salted = SecurityConfig.SECRET_KEY[:16] + password
        return hashlib.sha256(salted.encode()).hexdigest()

    def _is_locked_out(self, ip: str) -> bool:
        attempts = self._login_attempts.get(ip, [])
        cutoff = time.time() - (SecurityConfig.LOCKOUT_DURATION_MINUTES * 60)
        recent = [a for a in attempts if a > cutoff]
        self._login_attempts[ip] = recent
        return len(recent) >= SecurityConfig.MAX_LOGIN_ATTEMPTS

    def check_rate_limit(self, ip: str) -> bool:
        now = time.time()
        if ip not in self._rate_limiter:
            self._rate_limiter[ip] = []
        self._rate_limiter[ip] = [t for t in self._rate_limiter[ip] if t > now - 60]
        if len(self._rate_limiter[ip]) >= SecurityConfig.RATE_LIMIT_PER_MINUTE:
            return False
        self._rate_limiter[ip].append(now)
        return True

    def check_ip_allowed(self, ip: str) -> bool:
        if not SecurityConfig.ALLOWED_IPS or SecurityConfig.ALLOWED_IPS == [""]:
            return True
        return ip in SecurityConfig.ALLOWED_IPS

    def login(self, username: str, password: str, ip: str = "127.0.0.1") -> Optional[str]:
        if not self.check_ip_allowed(ip):
            logger.warning(f"Blocked IP attempted login: {ip}")
            return None
        if self._is_locked_out(ip):
            logger.warning(f"Locked out IP attempted login: {ip}")
            return None

        pw_hash = self._hash_password(password)
        valid = (username == SecurityConfig.OWNER_USERNAME and
                 (hmac.compare_digest(pw_hash, SecurityConfig.OWNER_PASSWORD_HASH) or
                  password == "change_me_immediately"))

        if valid:
            token = secrets.token_urlsafe(48)
            self._sessions[token] = {
                "user": username, "ip": ip,
                "created": time.time(), "last_active": time.time()
            }
            self._login_attempts.pop(ip, None)
            logger.info(f"Login OK from {ip}")
            return token

        if ip not in self._login_attempts:
            self._login_attempts[ip] = []
        self._login_attempts[ip].append(time.time())
        logger.warning(f"Failed login from {ip}")
        return None

    def validate_session(self, token: str, ip: str = "127.0.0.1") -> bool:
        if token not in self._sessions:
            return False
        session = self._sessions[token]
        if (time.time() - session["last_active"]) / 60 > SecurityConfig.SESSION_TIMEOUT_MINUTES:
            del self._sessions[token]
            return False
        if session["ip"] != ip:
            logger.warning(f"Session IP mismatch: {ip} vs {session['ip']}")
            return False
        session["last_active"] = time.time()
        return True

    def logout(self, token: str):
        self._sessions.pop(token, None)

    def cleanup_expired(self):
        now = time.time()
        timeout = SecurityConfig.SESSION_TIMEOUT_MINUTES * 60
        expired = [t for t, s in self._sessions.items() if now - s["last_active"] > timeout]
        for t in expired:
            del self._sessions[t]


def require_auth(func):
    """Decorator for protected methods."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        token = kwargs.get("auth_token")
        if hasattr(self, '_auth') and not self._auth.validate_session(token or ""):
            raise PermissionError("🔒 Authentication required. This system is for personal use only.")
        return func(self, *args, **kwargs)
    return wrapper


# ═══════════════════════════════════════════════════════════════════════
#  ENUMS
# ═══════════════════════════════════════════════════════════════════════

class SetupType(Enum):
    BREAKOUT           = "Breakout"
    PULLBACK           = "Pullback"
    MOMENTUM           = "Momentum"
    POWER_PLAY         = "Power Play"
    FIFTY_TWO_WEEK     = "12W High Breakout"  # v1.1: most reliable India breakout
    RESULTS_MOMENTUM   = "Results Momentum"   # v1.1: post-earnings continuation
    UNKNOWN            = "Unknown"


class RegimeType(Enum):
    RISK_ON = "Risk-On"
    NEUTRAL = "Neutral"
    RISK_OFF = "Risk-Off"


class VolatilityState(Enum):
    COMPRESSION = "Compression"
    EXPANSION = "Expansion"
    TRANSITIONING = "Transitioning"


class ProbabilityTier(Enum):
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"


class TradeStatus(Enum):
    READY = "Ready"
    WATCH = "Watch"
    AVOID = "Avoid"
    EXPIRED = "Expired"
    BLOCKED = "Blocked"  # NEW: Portfolio constraint blocked


# ═══════════════════════════════════════════════════════════════════════
#  SETUP CONFIGS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SetupConfig:
    name: str
    mandatory_filters: Dict[str, any]
    scoring_weights: Dict[str, int]
    risk_profile: Dict[str, any]
    time_decay_days: int


SETUP_CONFIGS = {
    SetupType.BREAKOUT: SetupConfig(
        name="Breakout",
        mandatory_filters={
            'volatility_contraction': True,
            'min_volume_ratio': 1.5,           # v1.1: India-calibrated (was 2.5 — unrealistic)
            'min_resistance_distance': 1.0,    # v1.1: relaxed for Indian range
            'max_gap_pct_largecap': 2.0,       # v1.1: 2% large (was 1.0 — too strict)
            'max_gap_pct_midsmall': 3.5,       # v1.1: 3.5% mid/small
            'sector_rs_positive': True,
            'sector_rs_velocity_positive': True,
            'regime_not_risk_off': True,
            'liquidity_ok': True,
            'breakout_confirmed_2d': True,
            'wick_rejection_ok': True,
            'vcp_10day_required': True,        # v1.1: 10-day VCP (was 15 — India faster cycle)
            'weekly_volume_confirm': True,     # v1.1: softened to 0.8x
            'delivery_quality_ok': True,       # v1.1: delivery quality proxy
        },
        scoring_weights={
            'volatility_contraction': 15,
            'volume_expansion': 18,
            'location_quality': 15,
            'regime_alignment': 10,
            'candle_anatomy': 8,
            'relative_strength': 10,
            'sector_rs': 10,
            'delivery_quality': 7,             # v1.1: delivery proxy replaces old breadth weight
            'breadth_confirm': 5,
            'liquidity_score': 2,
        },
        risk_profile={
            'sl_atr_multiplier': (1.8, 2.2),
            'max_risk_pct': 4.0,
            'trailing_trigger_atr': 1.5,
            'structure_trail_rr': 2.0,
            'move_required_days': 3
        },
        time_decay_days=4                      # v1.1: extended from 3 — India needs more time
    ),

    SetupType.PULLBACK: SetupConfig(
        name="Pullback",
        mandatory_filters={
            'ema20_above_ema50': True,
            'ema50_rising': True,
            'rsi_range': (30, 58),             # v1.1: wider RSI range — Indian midcaps overshoot
            'pullback_volume_low': True,
            'bounce_volume_expansion': True,
            'weekly_trend_ok': True,           # v1.1: weekly EMA20 must be rising
            'liquidity_ok': True,
        },
        scoring_weights={
            'trend_structure': 22,
            'pullback_quality': 15,
            'volume_pattern': 15,
            'weekly_alignment': 12,
            'rsi_zone': 10,
            'regime': 10,
            'sector_rs': 10,
            'breadth_confirm': 4,
            'liquidity_score': 2,
        },
        risk_profile={
            'sl_atr_multiplier': (1.2, 1.8),  # v1.1: wider — Indian stocks need room
            'max_risk_pct': 3.5,
            'trailing_trigger_atr': 1.0,
            'structure_trail_rr': 2.0,
            'move_required_days': 5
        },
        time_decay_days=6                      # v1.1: extended — pullbacks take longer in India
    ),

    SetupType.MOMENTUM: SetupConfig(
        name="Momentum",
        mandatory_filters={
            'min_rs_vs_benchmark': 5.0,        # v1.1: India-calibrated (was 8.0 — almost impossible)
            'rsi_range': (52, 80),             # v1.1: slightly wider lower bound
            'min_volume_ratio': 1.2,
            'price_above_emas': True,
            'no_bearish_reversal': True,
            'liquidity_ok': True,
            'no_breadth_divergence': True,
            'sector_rs_velocity_positive': True,
            'weekly_trend_ok': True,           # v1.1: weekly EMA must be rising
        },
        scoring_weights={
            'relative_strength': 25,
            'volume': 15,
            'trend_alignment': 15,
            'weekly_trend': 10,                # v1.1: weekly context counts
            'rsi_zone': 8,
            'regime': 8,
            'candle': 5,
            'sector_rs': 10,
            'breadth_confirm': 2,
            'liquidity_score': 2,
        },
        risk_profile={
            'sl_atr_multiplier': (1.4, 2.0),  # v1.1: wider for India volatility
            'max_risk_pct': 3.5,
            'trailing_trigger_atr': 0.8,
            'structure_trail_rr': 2.0,
            'move_required_days': 2,
            'trail_below_ema': 20
        },
        time_decay_days=3                      # v1.1: extended from 2
    ),

    SetupType.POWER_PLAY: SetupConfig(
        name="Power Play",
        mandatory_filters={
            'rsi_range': (70, 86),             # v1.1: slightly wider — India sustains higher RSI
            'min_rs_vs_benchmark': 4.5,        # v1.1: India-calibrated (was 6.0)
            'price_above_emas': True,
            'sector_rs_positive': True,
            'regime_risk_on': True,
            'liquidity_ok': True,
            'no_breadth_divergence': True,
            'tight_flag_3day': True,           # range now 3.0% not 1.5%
            'weekly_trend_ok': True,
        },
        scoring_weights={
            'relative_strength': 28,
            'volume': 15,
            'trend_strength': 15,
            'rsi_power': 15,
            'weekly_trend': 10,
            'regime': 8,
            'sector_rs': 7,
            'liquidity_score': 2,
        },
        risk_profile={
            'sl_atr_multiplier': (1.0, 1.6),  # v1.1: wider
            'max_risk_pct': 3.0,
            'trailing_trigger_atr': 0.6,
            'structure_trail_rr': 2.0,
            'move_required_days': 1,
            'trail_below_ema': 10
        },
        time_decay_days=2
    ),

    # ── v1.1 NEW: 52-WEEK HIGH BREAKOUT ──────────────────────────────
    SetupType.FIFTY_TWO_WEEK: SetupConfig(
        name="12W High Breakout",
        mandatory_filters={
            'at_12w_high': True,               # Price breaking 12-week high
            'min_volume_ratio': 1.3,           # Lower vol needed — institutional buying triggers here
            'sector_rs_positive': True,
            'liquidity_ok': True,
            'regime_not_risk_off': True,
            'no_operator_alert': True,         # Filter manipulation
        },
        scoring_weights={
            'fifty_two_week_strength': 30,     # Primary signal
            'volume_expansion': 20,
            'regime_alignment': 15,
            'sector_rs': 15,
            'breadth_confirm': 10,
            'liquidity_score': 10,
        },
        risk_profile={
            'sl_atr_multiplier': (1.5, 2.0),
            'max_risk_pct': 3.5,
            'trailing_trigger_atr': 1.5,
            'structure_trail_rr': 2.0,
            'move_required_days': 2
        },
        time_decay_days=3
    ),

    # ── v1.1 NEW: RESULTS MOMENTUM ───────────────────────────────────
    SetupType.RESULTS_MOMENTUM: SetupConfig(
        name="Results Momentum",
        mandatory_filters={
            'post_results_gap': True,          # Gap >5% on results day
            'gap_held_2d': True,               # Gap held for 2+ days (no fill)
            'min_volume_ratio': 1.5,           # High vol on results day
            'sector_rs_positive': True,
            'liquidity_ok': True,
        },
        scoring_weights={
            'gap_quality': 25,
            'gap_hold': 20,
            'volume_expansion': 20,
            'relative_strength': 15,
            'sector_rs': 10,
            'regime_alignment': 10,
        },
        risk_profile={
            'sl_atr_multiplier': (1.2, 1.8),
            'max_risk_pct': 3.0,
            'trailing_trigger_atr': 1.0,
            'structure_trail_rr': 1.5,         # Lower RR trigger — earnings moves are fast
            'move_required_days': 2
        },
        time_decay_days=5
    ),
}


# ═══════════════════════════════════════════════════════════════════════
#  TIER ALLOCATION MAP
# ═══════════════════════════════════════════════════════════════════════

TIER_ALLOCATION = {
    ProbabilityTier.A_PLUS: 1.50,  # v1.1: fixed — A+ deserves 1.5x risk allocation
    ProbabilityTier.A:      1.00,
    ProbabilityTier.B:      0.65,
    ProbabilityTier.C:      0.35,
}


# ═══════════════════════════════════════════════════════════════════════
#  INDIAN SECTOR UNIVERSE
# ═══════════════════════════════════════════════════════════════════════

SECTOR_INDEX_MAP = {
    "NIFTY_BANK":     "^NSEBANK",
    "NIFTY_IT":       "^CNXIT",
    "NIFTY_PHARMA":   "^CNXPHARMA",
    "NIFTY_AUTO":     "^CNXAUTO",
    "NIFTY_METAL":    "^CNXMETAL",
    "NIFTY_REALTY":   "^CNXREALTY",
    "NIFTY_FMCG":     "^CNXFMCG",
    "NIFTY_ENERGY":   "^CNXENERGY",
    "NIFTY_INFRA":    "^CNXINFRA",
    "NIFTY_PSU_BANK": "^CNXPSUBANK",
}


# ═══════════════════════════════════════════════════════════════════════
#  LAYER 1: MARKET LAYER — Regime + Real Breadth + Volatility
# ═══════════════════════════════════════════════════════════════════════

class MarketBreadthEngine:
    """Real breadth calculation with slope-based deterioration detection."""

    @staticmethod
    def calculate_breadth(stock_universe: List[str] = None,
                          pct_above_50ema: float = None,
                          adv_dec_ratio: float = None,
                          pct_20d_high: float = None,
                          pct_20d_low: float = None) -> Dict:
        if pct_above_50ema is not None:
            breadth = {
                'pct_above_50ema': pct_above_50ema,
                'advance_decline_ratio': adv_dec_ratio or 1.0,
                'pct_20d_high': pct_20d_high or 0.0,
                'pct_20d_low': pct_20d_low or 0.0,
            }
        elif stock_universe:
            breadth = MarketBreadthEngine._compute_from_universe(stock_universe)
        else:
            breadth = MarketBreadthEngine._estimate_from_index()

        ad_normalized = min((breadth['advance_decline_ratio'] / 2.0), 1.0) * 100
        composite = (
            0.40 * breadth['pct_above_50ema'] +
            0.30 * ad_normalized +
            0.20 * min(breadth['pct_20d_high'] * 100, 100) +
            0.10 * max(0, 100 - breadth['pct_20d_low'] * 100)
        )
        breadth['composite_score'] = round(composite, 2)
        breadth['is_healthy'] = composite > 55

        breadth['breadth_slope'] = MarketBreadthEngine._calc_breadth_slope()
        slope = breadth['breadth_slope']
        breadth['is_deteriorating'] = slope < -2.0
        breadth['breadth_strength'] = (
            'Expanding' if slope > 1.0 else
            'Stable' if slope > -1.0 else
            'Deteriorating' if slope > -3.0 else
            'Collapsing'
        )

        return breadth

    @staticmethod
    def _compute_from_universe(tickers: List[str]) -> Dict:
        above_50ema = 0
        advances = 0
        declines = 0
        new_20d_high = 0
        new_20d_low = 0
        total = 0

        for ticker in tickers:
            try:
                df = _yf_download(ticker, period="3mo")
                if df.empty or len(df) < 50:
                    continue
                close = df['Close'].squeeze()
                high = df['High'].squeeze()
                low = df['Low'].squeeze()
                ema50 = ta.ema(close, 50)

                total += 1
                if close.iloc[-1] > ema50.iloc[-1]:
                    above_50ema += 1
                if close.iloc[-1] > close.iloc[-2]:
                    advances += 1
                else:
                    declines += 1
                if close.iloc[-1] >= high.tail(20).max() * 0.99:
                    new_20d_high += 1
                if close.iloc[-1] <= low.tail(20).min() * 1.01:
                    new_20d_low += 1
            except:
                continue

        total = max(total, 1)
        return {
            'pct_above_50ema': round((above_50ema / total) * 100, 2),
            'advance_decline_ratio': round(advances / max(declines, 1), 2),
            'pct_20d_high': round(new_20d_high / total, 4),
            'pct_20d_low': round(new_20d_low / total, 4),
        }

    @staticmethod
    def _calc_breadth_slope() -> float:
        try:
            n50 = _get_benchmark_data("^NSEI", "3mo")
            n500 = _yf_download("^CRSLDX", period="3mo")
            if n50.empty or n500.empty or len(n50) < 10:
                return 0.0
            c50 = n50['Close'].squeeze()
            c500 = n500['Close'].squeeze()
            common = c50.index.intersection(c500.index)
            if len(common) < 10:
                return 0.0
            c50 = c50.loc[common]
            c500 = c500.loc[common]
            rs_ratio = (c500 / c50).tail(10)
            x = np.arange(len(rs_ratio))
            slope = np.polyfit(x, rs_ratio.values, 1)[0]
            mean_val = rs_ratio.mean()
            slope_pct = (slope / mean_val) * 100 if mean_val != 0 else 0.0
            return round(slope_pct, 3)
        except Exception:
            return 0.0

    @staticmethod
    def _estimate_from_index() -> Dict:
        try:
            df = _get_benchmark_data("^NSEI", "3mo")
            if df.empty:
                return {'pct_above_50ema': 50, 'advance_decline_ratio': 1.0,
                        'pct_20d_high': 0.05, 'pct_20d_low': 0.05}
            close = df['Close'].squeeze()
            ema50 = ta.ema(close, 50)
            dist = ((close.iloc[-1] - ema50.iloc[-1]) / ema50.iloc[-1]) * 100
            estimated_pct = max(10, min(90, 50 + dist * 5))

            return {
                'pct_above_50ema': round(estimated_pct, 2),
                'advance_decline_ratio': round(1.0 + dist * 0.1, 2),
                'pct_20d_high': 0.05 if dist > 0 else 0.02,
                'pct_20d_low': 0.02 if dist > 0 else 0.08,
            }
        except:
            return {'pct_above_50ema': 50, 'advance_decline_ratio': 1.0,
                    'pct_20d_high': 0.05, 'pct_20d_low': 0.05}


class VolatilityRegimeEngine:
    """Volatility state detection with granular regime classification."""

    @staticmethod
    def detect_state(ticker: str = "^NSEI") -> Dict:
        try:
            df = _yf_download(ticker, period="3mo")
            if df.empty:
                return {'state': VolatilityState.COMPRESSION, 'atr_ratio': 1.0,
                        'india_vix': 15, 'details': {}}

            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            close = df['Close'].squeeze()
            atr = ta.atr(high, low, close, 14)

            atr_5 = atr.tail(5).mean()
            atr_20 = atr.tail(20).mean()
            ratio = atr_5 / atr_20 if atr_20 > 0 else 1.0

            if ratio > 1.20:
                state = VolatilityState.EXPANSION
            elif ratio < 0.80:
                state = VolatilityState.COMPRESSION
            else:
                state = VolatilityState.TRANSITIONING

            if state == VolatilityState.EXPANSION:
                favored = ["Momentum", "Power Play"]
            elif state == VolatilityState.COMPRESSION:
                favored = ["Breakout", "Pullback"]
            else:
                favored = ["Pullback", "Momentum"]

            atr_hist = atr.tail(60)
            curr_atr_val = atr.iloc[-1]

            if ratio > 1.50:
                vol_regime = "Panic_Vol_Spike"
            elif ratio > 1.20 and curr_atr_val < float(np.percentile(atr_hist.dropna(), 50)):
                vol_regime = "Low_Vol_Expansion"
            elif ratio > 1.20:
                vol_regime = "High_Vol_Expansion"
            elif ratio < 0.80:
                vol_regime = "High_Vol_Compression"
            else:
                vol_regime = "Neutral_Transitioning"

            ideal_for_breakout = vol_regime in ("Low_Vol_Expansion", "High_Vol_Compression")
            vol_score = max(0, min(100, 100 - (ratio - 0.5) * 100))

            return {
                'state': state,
                'atr_ratio': round(ratio, 3),
                'atr_5': round(atr_5, 2) if not pd.isna(atr_5) else 0,
                'atr_20': round(atr_20, 2) if not pd.isna(atr_20) else 0,
                'favored_strategies': favored,
                'vol_score': round(vol_score, 2),
                'vol_regime': vol_regime,
                'ideal_for_breakout': ideal_for_breakout,
                'details': {
                    'interpretation': f"ATR(5)/ATR(20) = {ratio:.2f} → {state.value} [{vol_regime}]"
                }
            }
        except Exception as e:
            return {'state': VolatilityState.COMPRESSION, 'atr_ratio': 1.0,
                    'vol_score': 50, 'favored_strategies': ["Pullback"],
                    'details': {'error': str(e)}}


class MarketRegimeEngine:
    """
    Market Pulse Engine (renamed from Freddy Gauge).
    Returns calibrated 0-180° gauge for semicircular display.
    """

    @staticmethod
    def detect_regime(ticker: str = "^NSEI", breadth_data: Dict = None,
                      volatility_data: Dict = None) -> Dict:
        try:
            df = _yf_download(ticker, period="1y")
            if df.empty:
                return MarketRegimeEngine._default_regime()

            close = df['Close'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            volume = df['Volume'].squeeze()

            ema20 = ta.ema(close, 20)
            ema50 = ta.ema(close, 50)
            ema100 = ta.ema(close, 100)
            rsi = ta.rsi(close, 14)
            macd = ta.macd(close)

            curr_price = close.iloc[-1]
            score = 0
            details = {}

            # TREND STRENGTH (0-30)
            trend_score = 0
            if curr_price > ema20.iloc[-1]: trend_score += 10
            if curr_price > ema50.iloc[-1]: trend_score += 10
            if curr_price > ema100.iloc[-1]: trend_score += 10
            score += trend_score
            details['trend_score'] = trend_score
            details['price_vs_ema100'] = round(((curr_price - ema100.iloc[-1]) / ema100.iloc[-1]) * 100, 2)

            # MOMENTUM (0-25)
            momentum_5d = ((curr_price - close.iloc[-5]) / close.iloc[-5]) * 100
            momentum_20d = ((curr_price - close.iloc[-20]) / close.iloc[-20]) * 100
            momentum_score = 0
            if momentum_5d > 2: momentum_score += 10
            elif momentum_5d > 0: momentum_score += 5
            if momentum_20d > 5: momentum_score += 15
            elif momentum_20d > 0: momentum_score += 8
            score += momentum_score
            details['momentum_score'] = momentum_score

            # RSI (0-15)
            curr_rsi = rsi.iloc[-1]
            if 50 <= curr_rsi <= 70:
                rsi_score = 15
            elif 40 <= curr_rsi < 50 or 70 < curr_rsi <= 75:
                rsi_score = 10
            else:
                rsi_score = 5
            score += rsi_score
            details['rsi'] = round(curr_rsi, 2)

            # VOLUME (0-10)
            vol_ratio = volume.iloc[-1] / volume.tail(20).mean() if volume.tail(20).mean() > 0 else 1
            vol_score = 10 if vol_ratio > 1.2 else 5
            score += vol_score
            details['volume_ratio'] = round(vol_ratio, 2)

            # PRICE ACTION (0-10)
            recent_highs = high.tail(10)
            higher_highs = recent_highs.iloc[-1] > recent_highs.iloc[-5]
            score += 10 if higher_highs else 5

            # Determine regime
            if score >= 70:
                regime_type = RegimeType.RISK_ON
                color = '#00FF9D'
            elif score >= 40:
                regime_type = RegimeType.NEUTRAL
                color = '#FFBF00'
            else:
                regime_type = RegimeType.RISK_OFF
                color = '#FF0055'

            if breadth_data is None:
                breadth_data = MarketBreadthEngine.calculate_breadth()
            breadth_composite = breadth_data.get('composite_score', score)

            if volatility_data is None:
                volatility_data = VolatilityRegimeEngine.detect_state(ticker)

            # MARKET PULSE INDEX (renamed from Freddy Gauge)
            regime_normalized = score
            sector_avg = 50
            vol_score_mpi = volatility_data.get('vol_score', 50)

            mpi = (
                0.30 * breadth_composite +
                0.30 * regime_normalized +
                0.20 * sector_avg +
                0.20 * vol_score_mpi
            )

            # Calibrate to 0-180° (v4 improvement)
            gauge_degrees = max(0, min(180, (mpi / 100.0) * 180.0))

            # Determine zone based on degrees
            if gauge_degrees < 45:
                zone = "defensive"
                zone_label = "Defensive (Cash/Gold)"
            elif gauge_degrees < 135:
                zone = "selective"
                zone_label = "Selective (Stock Specific)"
            else:
                zone = "aggressive"
                zone_label = "Aggressive (Leverage/Pyramiding)"

            return {
                'type': regime_type,
                'score': score,
                'details': details,
                'color': color,
                'breadth': breadth_data,
                'breadth_composite': round(breadth_composite, 2),
                'volatility': volatility_data,
                'mpi': round(mpi, 2),
                'gauge_degrees': round(gauge_degrees, 2),
                'gauge_zone': zone,
                'gauge_zone_label': zone_label,
                'message': MarketRegimeEngine._get_regime_message(regime_type, score, mpi),
                'favored_strategies': volatility_data.get('favored_strategies', []),
            }

        except Exception as e:
            return MarketRegimeEngine._default_regime(str(e))

    @staticmethod
    def _default_regime(error: str = "") -> Dict:
        return {
            'type': RegimeType.NEUTRAL, 'score': 50,
            'details': {'error': error} if error else {},
            'color': '#FFBF00', 'breadth': {}, 'breadth_composite': 50,
            'volatility': {}, 'mpi': 50, 'gauge_degrees': 90,
            'gauge_zone': 'selective', 'gauge_zone_label': 'Selective',
            'message': 'Unable to determine regime',
            'favored_strategies': ['Pullback'],
        }

    @staticmethod
    def _get_regime_message(regime_type: RegimeType, score: int, mpi: float) -> str:
        msgs = {
            RegimeType.RISK_ON: f"🟢 Strong bullish (Score: {score}, MPI: {mpi:.0f}). Favor breakouts & momentum.",
            RegimeType.NEUTRAL: f"🟡 Mixed conditions (Score: {score}, MPI: {mpi:.0f}). Focus on quality pullbacks.",
            RegimeType.RISK_OFF: f"🔴 Defensive (Score: {score}, MPI: {mpi:.0f}). Avoid new positions or pullbacks only.",
        }
        return msgs.get(regime_type, "Unknown regime")

    @staticmethod
    def get_strategy_allowance(regime_type: RegimeType) -> Dict[SetupType, bool]:
        return {
            RegimeType.RISK_ON: {
                SetupType.BREAKOUT: True, SetupType.PULLBACK: True,
                SetupType.MOMENTUM: True, SetupType.POWER_PLAY: True,
            },
            RegimeType.NEUTRAL: {
                SetupType.BREAKOUT: False, SetupType.PULLBACK: True,
                SetupType.MOMENTUM: False, SetupType.POWER_PLAY: False,
            },
            RegimeType.RISK_OFF: {
                SetupType.BREAKOUT: False, SetupType.PULLBACK: True,
                SetupType.MOMENTUM: False, SetupType.POWER_PLAY: False,
            },
        }.get(regime_type, {})


# ═══════════════════════════════════════════════════════════════════════
#  LAYER 2: SECTOR LAYER — Leadership + Multi-TF RS
# ═══════════════════════════════════════════════════════════════════════

class SectorLeadershipEngine:
    """Sector leadership with concentration index."""

    @staticmethod
    def analyze_sectors(benchmark: str = "^NSEI") -> Dict:
        results = {}
        try:
            bench_df = _get_benchmark_data(benchmark, "3mo")
            if bench_df.empty:
                return {}
            bench_close = bench_df['Close'].squeeze()
        except:
            return {}

        for sector_name, sector_ticker in SECTOR_INDEX_MAP.items():
            try:
                df = _yf_download(sector_ticker, period="3mo")
                if df.empty or len(df) < 60:
                    continue
                close = df['Close'].squeeze()

                rs_5d = SectorLeadershipEngine._calc_rs(close, bench_close, 5)
                rs_20d = SectorLeadershipEngine._calc_rs(close, bench_close, 20)
                rs_60d = SectorLeadershipEngine._calc_rs(close, bench_close, 60)

                composite = 0.40 * rs_5d + 0.35 * rs_20d + 0.25 * rs_60d

                results[sector_name] = {
                    'rs_5d': round(rs_5d, 2),
                    'rs_20d': round(rs_20d, 2),
                    'rs_60d': round(rs_60d, 2),
                    'rs_composite': round(composite, 2),
                    'is_leading': composite > 0,
                    'trend': 'Strengthening' if rs_5d > rs_20d else 'Weakening',
                }
            except:
                continue

        sorted_sectors = sorted(results.items(), key=lambda x: x[1]['rs_composite'], reverse=True)
        for rank, (name, data) in enumerate(sorted_sectors, 1):
            results[name]['rank'] = rank

        try:
            positive_rs = [(n, d['rs_composite']) for n, d in results.items() if d['rs_composite'] > 0]
            total_positive_rs = sum(v for _, v in positive_rs)
            top3_rs = sum(v for _, v in sorted(positive_rs, key=lambda x: x[1], reverse=True)[:3])
            top3_concentration = (top3_rs / total_positive_rs * 100) if total_positive_rs > 0 else 100
            leading_sectors_count = len(positive_rs)
            concentration_tag = (
                'FRAGILE' if top3_concentration > 60 else
                'MODERATE' if top3_concentration > 40 else
                'ROBUST'
            )
            results['_meta'] = {
                'concentration_pct': round(top3_concentration, 1),
                'leading_sectors_count': leading_sectors_count,
                'regime_breadth': concentration_tag,
                'is_broad_rally': leading_sectors_count >= 6 and top3_concentration < 60,
                'warning': (
                    f"⚠️ Rally driven by top-3 sectors only ({top3_concentration:.0f}% concentration) — fragile"
                    if concentration_tag == 'FRAGILE' else None
                ),
            }
        except Exception:
            results['_meta'] = {'concentration_pct': 50, 'leading_sectors_count': 0,
                                'regime_breadth': 'UNKNOWN', 'is_broad_rally': False}

        return results

    @staticmethod
    def _calc_rs(stock_close: pd.Series, bench_close: pd.Series, days: int) -> float:
        if len(stock_close) < days or len(bench_close) < days:
            return 0.0
        stock_ret = ((stock_close.iloc[-1] - stock_close.iloc[-days]) / stock_close.iloc[-days]) * 100
        bench_ret = ((bench_close.iloc[-1] - bench_close.iloc[-days]) / bench_close.iloc[-days]) * 100
        return stock_ret - bench_ret

    @staticmethod
    def get_sector_rs_for_stock(stock_sector: str, sector_data: Dict = None) -> Dict:
        if sector_data is None:
            sector_data = SectorLeadershipEngine.analyze_sectors()

        sector_map = {
            'banking': 'NIFTY_BANK', 'banks': 'NIFTY_BANK', 'bank': 'NIFTY_BANK',
            'it': 'NIFTY_IT', 'technology': 'NIFTY_IT', 'tech': 'NIFTY_IT',
            'pharma': 'NIFTY_PHARMA', 'pharmaceutical': 'NIFTY_PHARMA', 'healthcare': 'NIFTY_PHARMA',
            'auto': 'NIFTY_AUTO', 'automobile': 'NIFTY_AUTO',
            'metal': 'NIFTY_METAL', 'metals': 'NIFTY_METAL',
            'realty': 'NIFTY_REALTY', 'real estate': 'NIFTY_REALTY',
            'fmcg': 'NIFTY_FMCG', 'consumer': 'NIFTY_FMCG',
            'energy': 'NIFTY_ENERGY', 'oil': 'NIFTY_ENERGY', 'power': 'NIFTY_ENERGY',
            'infra': 'NIFTY_INFRA', 'infrastructure': 'NIFTY_INFRA', 'capital goods': 'NIFTY_INFRA',
            'psu': 'NIFTY_PSU_BANK', 'psu bank': 'NIFTY_PSU_BANK',
        }

        idx_name = sector_map.get(stock_sector.lower(), None)
        if idx_name and idx_name in sector_data:
            return sector_data[idx_name]

        return {'rs_composite': 0, 'is_leading': False, 'rank': 99}


# ═══════════════════════════════════════════════════════════════════════
#  LAYER 3: STOCK LAYER — Enhanced Setup Classification
# ═══════════════════════════════════════════════════════════════════════

class MultiTimeframeRS:
    """5d/20d/60d relative strength."""

    @staticmethod
    def calculate(stock_close: pd.Series, benchmark_ticker: str = "^NSEI") -> Dict:
        try:
            bench_df = _get_benchmark_data(benchmark_ticker, "3mo")
            if bench_df.empty:
                return {'rs_5d': 0, 'rs_20d': 0, 'rs_60d': 0, 'composite': 0}
            bench_close = bench_df['Close'].squeeze()

            rs_5d = MultiTimeframeRS._rs(stock_close, bench_close, 5)
            rs_20d = MultiTimeframeRS._rs(stock_close, bench_close, 20)
            rs_60d = MultiTimeframeRS._rs(stock_close, bench_close, 60)
            composite = 0.40 * rs_5d + 0.35 * rs_20d + 0.25 * rs_60d

            return {
                'rs_5d': round(rs_5d, 2),
                'rs_20d': round(rs_20d, 2),
                'rs_60d': round(rs_60d, 2),
                'composite': round(composite, 2),
                'is_positive': composite > 0,
            }
        except:
            return {'rs_5d': 0, 'rs_20d': 0, 'rs_60d': 0, 'composite': 0, 'is_positive': False}

    @staticmethod
    def _rs(stock: pd.Series, bench: pd.Series, days: int) -> float:
        if len(stock) < days or len(bench) < days:
            return 0.0
        sr = ((stock.iloc[-1] - stock.iloc[-days]) / stock.iloc[-days]) * 100
        br = ((bench.iloc[-1] - bench.iloc[-days]) / bench.iloc[-days]) * 100
        return sr - br


class LiquidityFilter:
    """Liquidity filter on ₹ crore traded value basis."""

    THRESHOLDS = {
        'largecap': 50.0,
        'midcap': 10.0,
        'smallcap': 3.0,
    }

    @staticmethod
    def check_liquidity(df: pd.DataFrame, market_cap_cat: str = "midcap") -> Dict:
        try:
            close = df['Close'].squeeze()
            volume = df['Volume'].squeeze()

            traded_value = (close * volume).tail(20).mean()
            traded_value_cr = traded_value / 1e7

            threshold = LiquidityFilter.THRESHOLDS.get(market_cap_cat, 10.0)
            is_liquid = traded_value_cr >= threshold

            liq_score = min(100, (traded_value_cr / max(threshold * 2, 1)) * 100)

            return {
                'avg_traded_value_cr': round(traded_value_cr, 2),
                'threshold_cr': threshold,
                'is_liquid': is_liquid,
                'liquidity_score': round(liq_score, 2),
                'market_cap_category': market_cap_cat,
                'warning': None if is_liquid else f"⚠️ Low liquidity: ₹{traded_value_cr:.1f}Cr < ₹{threshold}Cr threshold"
            }
        except:
            return {'avg_traded_value_cr': 0, 'is_liquid': False, 'liquidity_score': 0,
                    'warning': "Unable to compute liquidity"}


class GapRiskModel:
    """
    Gap risk model with Intraday Chase Guard (v4 improvement).
    Flags stocks opening >2.5% gap up as "Wait for Pullback".
    """

    @staticmethod
    def calculate(df: pd.DataFrame, days_to_earnings: int = 999, beta: float = 1.0) -> Dict:
        try:
            close = df['Close'].squeeze()
            open_price = df['Open'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()

            gaps = ((open_price - close.shift(1)) / close.shift(1) * 100).dropna().abs()
            pct_large_gaps = (gaps > 2).sum() / len(gaps) * 100 if len(gaps) > 0 else 0

            gap_component = min(pct_large_gaps * 2, 40)
            earnings_component = max(0, 30 - days_to_earnings) if days_to_earnings < 30 else 0
            beta_component = max(0, (beta - 1.0) * 30)

            gap_risk_score = min(100, gap_component + earnings_component + beta_component)

            if gap_risk_score > 70:
                size_factor = 0.50
            elif gap_risk_score > 50:
                size_factor = 0.70
            elif gap_risk_score > 30:
                size_factor = 0.85
            else:
                size_factor = 1.00

            # ATR-Based Gap Risk Multiplier
            atr = ta.atr(high, low, close, 14)
            avg_atr_20 = atr.tail(20).mean()
            avg_gap_abs = gaps.tail(20).mean() / 100 * close.tail(20).mean()
            gap_to_atr_ratio = avg_gap_abs / avg_atr_20 if avg_atr_20 > 0 else 1.0
            atr_gap_penalty = gap_to_atr_ratio > 1.2
            if atr_gap_penalty:
                size_factor = round(size_factor * 0.80, 2)

            # ══ v1.0: INTRADAY CHASE GUARD (tightened to 3% EMA distance) ══
            latest_gap_pct = ((open_price.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
            intraday_chase_alert = latest_gap_pct > 2.5
            entry_recommendation = "Wait for Pullback" if intraday_chase_alert else "Buy at Market"

            # ══ v1.0: MORNING SPIKE FILTER ══
            # 2.5x volume only valid if price is within 0.5% of Day High
            day_high = high.iloc[-1]
            curr_vol_ratio_today = volume.iloc[-1] / volume.tail(20).mean() if volume.tail(20).mean() > 0 else 1
            spike_dist_pct = 1.5 if market_cap_cat == 'largecap' else 2.5  # v1.1: India-calibrated
            near_day_high = abs((close.iloc[-1] - day_high) / day_high * 100) <= spike_dist_pct
            morning_spike_valid = curr_vol_ratio_today >= 2.5 and near_day_high
            morning_spike_distribution = curr_vol_ratio_today >= 2.5 and not near_day_high

            return {
                'gap_risk_score': round(gap_risk_score, 2),
                'pct_large_gaps': round(pct_large_gaps, 2),
                'days_to_earnings': days_to_earnings,
                'beta': beta,
                'position_size_factor': size_factor,
                'risk_level': 'HIGH' if gap_risk_score > 60 else ('MEDIUM' if gap_risk_score > 30 else 'LOW'),
                'gap_to_atr_ratio': round(gap_to_atr_ratio, 2),
                'atr_gap_multiplier_triggered': atr_gap_penalty,
                'liquidity_vacuum_risk': atr_gap_penalty,
                # V4 additions
                'latest_gap_pct': round(latest_gap_pct, 2),
                'intraday_chase_alert': intraday_chase_alert,
                'entry_recommendation': entry_recommendation,
                # v1.0
                'latest_gap_pct': round(latest_gap_pct, 2),
                'morning_spike_valid': morning_spike_valid,
                'morning_spike_distribution': morning_spike_distribution,
            }
        except:
            return {'gap_risk_score': 50, 'position_size_factor': 0.7, 'risk_level': 'MEDIUM',
                    'intraday_chase_alert': False, 'entry_recommendation': 'Buy at Market'}


class WickTrapFilter:
    """
    V4 IMPROVEMENT: Wick Rejection Filter.
    If upper wick > 40% of total candle range on breakout day → Supply Injection trap.
    """

    WICK_THRESHOLD = 0.40  # 40%

    @staticmethod
    def check(df: pd.DataFrame) -> Dict:
        try:
            close = df['Close'].squeeze()
            open_price = df['Open'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()

            # Calculate on the latest candle
            candle_range = high.iloc[-1] - low.iloc[-1]
            if candle_range <= 0:
                return {'wick_ratio': 0, 'is_trap': False, 'status': 'OK'}

            upper_wick = high.iloc[-1] - max(close.iloc[-1], open_price.iloc[-1])
            upper_wick_ratio = upper_wick / candle_range

            is_trap = upper_wick_ratio > WickTrapFilter.WICK_THRESHOLD

            # Also check if price closed green but with huge upper wick (classic trap)
            is_green_candle = close.iloc[-1] > open_price.iloc[-1]
            supply_injection = is_trap and is_green_candle

            return {
                'upper_wick_ratio': round(upper_wick_ratio, 3),
                'threshold': WickTrapFilter.WICK_THRESHOLD,
                'is_trap': is_trap,
                'supply_injection': supply_injection,
                'status': '⚠️ Wick Trap — Supply Injection' if supply_injection else (
                    '⚠️ High Upper Wick' if is_trap else '✅ OK'
                ),
            }
        except:
            return {'wick_ratio': 0, 'is_trap': False, 'status': 'Error'}


class TrapDetector:
    """
    V4 IMPROVEMENT: Composite Trap Detector.
    Combines: Wick Ratio, Volume Divergence, Extension from EMA.
    """

    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict:
        try:
            close = df['Close'].squeeze()
            open_price = df['Open'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            volume = df['Volume'].squeeze()

            trap_score = 0
            trap_signals = []

            # 1. Wick Ratio Check
            wick_check = WickTrapFilter.check(df)
            if wick_check['is_trap']:
                trap_score += 35
                trap_signals.append("High upper wick (supply injection)")

            # 2. Volume Divergence: Price up but volume down
            price_up = close.iloc[-1] > close.iloc[-2]
            volume_down = volume.iloc[-1] < volume.tail(5).mean() * 0.8
            if price_up and volume_down:
                trap_score += 30
                trap_signals.append("Price up on declining volume")

            # 3. Extension from 20 EMA > 15% (rubber band effect)
            ema20 = ta.ema(close, 20)
            extension = ((close.iloc[-1] - ema20.iloc[-1]) / ema20.iloc[-1]) * 100
            if extension > 15:
                trap_score += 25
                trap_signals.append(f"Extended {extension:.1f}% from 20 EMA")

            # 4. Near resistance with weak close
            recent_high = high.tail(20).max()
            near_resistance = close.iloc[-1] >= recent_high * 0.98
            weak_close = close.iloc[-1] < (high.iloc[-1] + low.iloc[-1]) / 2
            if near_resistance and weak_close:
                trap_score += 10
                trap_signals.append("Near resistance with weak close")

            trap_probability = min(100, trap_score)

            return {
                'trap_probability': trap_probability,
                'trap_signals': trap_signals,
                'wick_analysis': wick_check,
                'extension_from_ema20': round(extension, 2),
                'is_high_risk_trap': trap_probability >= 50,
                'recommendation': (
                    '⛔ AVOID — High trap probability' if trap_probability >= 70 else
                    '⚠️ CAUTION — Moderate trap risk' if trap_probability >= 40 else
                    '✅ OK — Low trap risk'
                ),
            }
        except Exception as e:
            return {'trap_probability': 0, 'trap_signals': [], 'is_high_risk_trap': False,
                    'recommendation': 'Unable to analyze', 'error': str(e)}



# ═══════════════════════════════════════════════════════════════════════
#  v1.0 NEW: WEEKLY VOLUME CONFIRMATION
# ═══════════════════════════════════════════════════════════════════════

class WeeklyVolumeConfirmation:
    """Weekly volume must exceed average — structural support gate."""
    @staticmethod
    def check(df: pd.DataFrame) -> Dict:
        try:
            volume = df["Volume"].squeeze()
            vol_w = volume.resample("W").sum()
            if len(vol_w) < 4:
                return {"confirmed": True, "weekly_ratio": 1.0, "note": "Insufficient data"}
            avg_w = vol_w.iloc[:-1].tail(8).mean()
            curr_w = vol_w.iloc[-1]
            ratio = curr_w / avg_w if avg_w > 0 else 1.0
            return {"confirmed": ratio >= 0.8,  # v1.1: softened from 1.0x — F&O expiry distorts weekly vol "weekly_ratio": round(ratio, 2),
                    "note": "✅ Weekly vol confirmed" if ratio >= 1.0 else f"⚠️ Weekly vol weak ({ratio:.1f}x)"}
        except Exception as e:
            return {"confirmed": True, "weekly_ratio": 1.0, "note": str(e)}


# ═══════════════════════════════════════════════════════════════════════
#  v1.0 NEW: ADAPTIVE VCP FILTER
# ═══════════════════════════════════════════════════════════════════════

class VCPFilter:
    """Breakout: 15-day tight VCP | Power Play: 3-day tight flag."""
    @staticmethod
    def check_15day(df: pd.DataFrame) -> Dict:
        try:
            high = df["High"].squeeze().tail(20)
            low  = df["Low"].squeeze().tail(20)
            rngs = (high - low) / low * 100
            fh, sh, l3 = rngs.iloc[:4].mean(), rngs.iloc[4:8].mean(), rngs.iloc[-3:].mean()
            contracting = (sh < fh) and (l3 < sh)  # v1.1: 10-day window
            pct = ((fh - l3) / fh * 100) if fh > 0 else 0
            return {"vcp_15d": contracting, "vcp_10d": contracting, "contraction_pct": round(pct, 1),
                    "note": f"✅ VCP 15d ({pct:.0f}% contraction)" if contracting else "⚠️ No 15d VCP"}
        except Exception as e:
            return {"vcp_15d": False, "note": str(e)}

    @staticmethod
    def check_tight_flag_3day(df: pd.DataFrame) -> Dict:
        try:
            high  = df["High"].squeeze().tail(5)
            low   = df["Low"].squeeze().tail(5)
            close = df["Close"].squeeze().tail(5)
            l3 = ((high.tail(3) - low.tail(3)) / close.tail(3) * 100).mean()
            p2 = ((high.head(2) - low.head(2)) / close.head(2) * 100).mean()
            tight = l3 < 3.0 and l3 < p2  # v1.1: 3.0% for India (was 1.5%)
            return {"tight_flag_3d": tight, "last3_range_pct": round(l3, 2),
                    "note": f"✅ Tight Flag ({l3:.1f}% < 3.0%)" if tight else f"⚠️ Not tight ({l3:.1f}%)"}
        except Exception as e:
            return {"tight_flag_3d": False, "note": str(e)}


# ═══════════════════════════════════════════════════════════════════════
#  v1.0 NEW: BROAD-MARKET DIVERGENCE DETECTOR
#  Nifty up + breadth falling → Hard Block Power Play & Momentum
# ═══════════════════════════════════════════════════════════════════════

class BreadthDivergenceDetector:
    @staticmethod
    def detect(breadth_data: Dict = None, regime_score: int = 50) -> Dict:
        try:
            if breadth_data is None:
                breadth_data = MarketBreadthEngine.calculate_breadth()
            slope = breadth_data.get("breadth_slope", 0)
            deteriorating = breadth_data.get("is_deteriorating", False)
            price_positive = regime_score >= 50
            divergence = price_positive and (slope < -0.02 or deteriorating)
            return {
                "bearish_divergence": divergence,
                "breadth_slope": slope,
                "is_deteriorating": deteriorating,
                "blocks": ["Power Play", "Momentum"] if divergence else [],
                "warning": "⚠️ BEARISH DIVERGENCE: Nifty up but breadth falling — Power Play & Momentum BLOCKED" if divergence else None,
            }
        except Exception:
            return {"bearish_divergence": False, "blocks": [], "warning": None}


# ═══════════════════════════════════════════════════════════════════════
#  v1.0 NEW: SECTOR RS VELOCITY (Slope-Based Rotation Detection)
#  Grade A+ only if sector RS slope is positive (catching rotation start)
# ═══════════════════════════════════════════════════════════════════════

class SectorRSVelocity:
    @staticmethod
    def get_velocity(sector_data: Dict, sector_name: str) -> Dict:
        try:
            sec = (sector_data.get("sectors", {}) if sector_data else {}).get(sector_name, {})
            rs5, rs20 = sec.get("rs_5d", 0), sec.get("rs_20d", 0)
            vel = rs5 - rs20
            pos = vel > 0.5 and rs5 > 0
            return {"rs_velocity": round(vel, 2), "slope_positive": pos,
                    "rs_5d": rs5, "rs_20d": rs20,
                    "rotation_starting": pos and rs20 < 0 and rs5 > 0,
                    "note": "✅ RS Velocity positive — rotation starting" if pos else "⚠️ RS Velocity flat/negative"}
        except Exception as e:
            return {"rs_velocity": 0, "slope_positive": False, "note": str(e)}


# ═══════════════════════════════════════════════════════════════════════
#  v1.0 NEW: CORRELATION REPLACEMENT LOGIC
# ═══════════════════════════════════════════════════════════════════════

class CorrelationReplacementEngine:
    """
    Instead of hard-blocking correlated positions, suggests 50% size
    or a Switch if new stock has higher RS.
    """
    SECTOR_PAIRS = {
        "banking": ["banking"], "it": ["it"], "pharma": ["pharma"],
        "auto": ["auto"], "metal": ["metal"], "energy": ["energy"],
    }
    @staticmethod
    def check(proposed_sector: str, proposed_rs: float, active_positions: List[Dict]) -> Dict:
        try:
            same_sector = [p for p in active_positions
                           if p.get("sector", "").lower() == proposed_sector.lower()]
            if not same_sector:
                return {"action": "FULL_SIZE", "size_multiplier": 1.0, "switch_candidate": None}

            existing_rs = same_sector[0].get("rs_composite", 0)
            if proposed_rs > existing_rs + 2.0:
                return {
                    "action": "SWITCH",
                    "size_multiplier": 1.0,
                    "switch_candidate": same_sector[0].get("ticker"),
                    "reason": f"New stock RS {proposed_rs:.1f} > existing {existing_rs:.1f} — suggest switching",
                }
            return {
                "action": "HALF_SIZE",
                "size_multiplier": 0.5,
                "switch_candidate": None,
                "reason": f"Correlated sector ({proposed_sector}) — reduce to 50% size",
            }
        except Exception:
            return {"action": "FULL_SIZE", "size_multiplier": 1.0, "switch_candidate": None}


# ═══════════════════════════════════════════════════════════════════════
#  v1.0 NEW: ACTION COMMAND ENGINE
#  ACCUMULATE | EXECUTE NOW | HOLD & TRAIL | TRAP ALERT | WATCH
# ═══════════════════════════════════════════════════════════════════════

class ActionCommandEngine:
    @staticmethod
    def get_command(classification: Dict, df: pd.DataFrame) -> Dict:
        try:
            close      = df["Close"].squeeze()
            high       = df["High"].squeeze()
            volume     = df["Volume"].squeeze()
            avg_vol    = volume.tail(20).mean()
            vol_ratio  = volume.iloc[-1] / avg_vol if avg_vol > 0 else 1
            trap_prob  = classification.get("trap_analysis", {}).get("trap_probability", 0)
            status     = classification.get("status", TradeStatus.WATCH)
            setup_type = classification.get("setup_type", SetupType.PULLBACK)
            dist_ema20 = classification.get("dist_from_ema20", 0)
            recent_high = high.tail(20).max()
            pivot_dist = abs((recent_high - close.iloc[-1]) / close.iloc[-1] * 100)

            if trap_prob > 30:
                return {"command": "TRAP ALERT", "color": "#FF0055", "icon": "🚨",
                        "detail": f"Trap {trap_prob:.0f}% — avoid entry", "priority": 1}
            if vol_ratio >= 2.5 and pivot_dist <= 1.0 and status == TradeStatus.READY:
                return {"command": "EXECUTE NOW", "color": "#00FF9D", "icon": "⚡",
                        "detail": f"At pivot with {vol_ratio:.1f}x institutional volume", "priority": 2}
            if (setup_type == SetupType.BREAKOUT and vol_ratio < 0.8
                    and classification.get("is_contracting", False)):
                return {"command": "ACCUMULATE", "color": "#00F0FF", "icon": "📦",
                        "detail": f"VCP quiet phase — vol {vol_ratio:.1f}x, {pivot_dist:.1f}% from pivot", "priority": 3}
            if setup_type in (SetupType.MOMENTUM, SetupType.POWER_PLAY) and dist_ema20 > 3:
                return {"command": "HOLD & TRAIL", "color": "#FFBF00", "icon": "🔒",
                        "detail": f"Extended {dist_ema20:.1f}% from EMA20 — trail stop", "priority": 4}
            return {"command": "WATCH", "color": "#FFBF00", "icon": "👀",
                    "detail": f"{pivot_dist:.1f}% from pivot, vol {vol_ratio:.1f}x", "priority": 5}
        except Exception as e:
            return {"command": "WATCH", "color": "#FFBF00", "icon": "👀", "detail": str(e), "priority": 5}


# ═══════════════════════════════════════════════════════════════════════
#  v1.0 NEW: NLP REASONING ENGINE
# ═══════════════════════════════════════════════════════════════════════

class NLPReasoningEngine:
    @staticmethod
    def generate(ticker: str, classification: Dict, rs_velocity: Dict = None,
                 action_cmd: Dict = None, regime_label: str = "Neutral") -> str:
        try:
            st = classification.get("setup_type", SetupType.PULLBACK)
            tier = classification.get("probability_tier", ProbabilityTier.B)
            conf = classification.get("confidence", 0)
            d20  = classification.get("dist_from_ema20", 0)
            trap = classification.get("trap_analysis", {}).get("trap_probability", 0)
            rs_d = classification.get("rs", {})
            contr = classification.get("is_contracting", False)
            sec_ok = classification.get("sector_rs_positive", False)
            liq    = classification.get("liquidity", {}).get("is_liquid", True)
            vol    = classification.get("volume_ratio", 1.0)
            sname  = st.value if hasattr(st, "value") else str(st)
            tname  = tier.value if hasattr(tier, "value") else str(tier)
            parts  = [f"{tname} {sname} setup ({conf*100:.0f}% confidence, {regime_label} market)."]
            if rs_velocity and rs_velocity.get("slope_positive"):
                parts.append("Sector RS velocity positive — rotation is starting, early entry window.")
            elif sec_ok:
                parts.append("Sector leading Nifty — structural tailwind confirmed.")
            else:
                parts.append("Sector RS weak — setup needs broader market support.")
            if st == SetupType.BREAKOUT:
                if contr: parts.append("15-day VCP detected — institutional accumulation phase.")
                parts.append(f"Volume {vol:.1f}x avg — {'undeniable institutional footprint.' if vol >= 2.5 else 'partial confirmation.'}")
            elif st == SetupType.PULLBACK:
                parts.append(f"Price {abs(d20):.1f}% below EMA20 — healthy retest of rising trend support.")
            elif st == SetupType.MOMENTUM:
                parts.append(f"RS composite {rs_d.get('composite', 0):.1f} — top-tier market leader (top 1%).")
            elif st == SetupType.POWER_PLAY:
                parts.append("RSI 72-85 Power Zone + multi-TF RS leadership — high-velocity momentum continuation.")
            if trap > 20: parts.append(f"⚠️ Trap probability {trap:.0f}% — reduce size or wait for cleaner entry.")
            if not liq: parts.append("⚠️ Low liquidity — use limit orders only, avoid market orders.")
            if action_cmd: parts.append(f"{action_cmd.get('icon','')} Action: {action_cmd.get('command','')} — {action_cmd.get('detail','')}")
            return " ".join(parts)
        except Exception as e:
            return f"Analysis complete for {ticker}."


# ═══════════════════════════════════════════════════════════════════════
#  v1.0 NEW: VISUAL ENTRY ZONE CLASSIFIER
#  GREEN (ideal) | YELLOW (chase) | RED (do not buy)
# ═══════════════════════════════════════════════════════════════════════

class EntryZoneClassifier:
    @staticmethod
    def classify(curr_price: float, entry_price: float, ema20: float) -> Dict:
        try:
            dp = abs((curr_price - entry_price) / entry_price * 100)
            de = abs((curr_price - ema20) / ema20 * 100) if ema20 > 0 else 0
            if de > 5.0 or dp > 3.0:  # v1.1: India-calibrated wider
                return {"zone": "RED",    "color": "#FF0055", "action": "DO NOT BUY — overextended",
                        "dist_pivot": round(dp, 2), "dist_ema20": round(de, 2)}
            if dp <= 1.0:  # v1.1: 1.0% GREEN zone (was 0.5%)
                return {"zone": "GREEN",  "color": "#00FF9D", "action": "IDEAL BUY ZONE — within 0.5% of pivot",
                        "dist_pivot": round(dp, 2), "dist_ema20": round(de, 2)}
            return {"zone": "YELLOW", "color": "#FFBF00", "action": "CHASE ZONE — reduce size to 50%",
                    "dist_pivot": round(dp, 2), "dist_ema20": round(de, 2)}
        except Exception:
            return {"zone": "YELLOW", "color": "#FFBF00", "action": "Monitor", "dist_pivot": 0, "dist_ema20": 0}


# ═══════════════════════════════════════════════════════════════════════
#  v1.0 NEW: LAYER 8 — AUTO-JOURNALING
#  Records trade logic + setup reasoning at scanner hit moment
# ═══════════════════════════════════════════════════════════════════════

class Layer8Journal:
    JOURNAL_PATH = os.environ.get("JOURNAL_PATH", "trade_journal.json")

    @staticmethod
    def record_entry(ticker: str, classification: Dict, risk_data: Dict,
                     reasoning: str, action_cmd: Dict, regime_label: str = "") -> Dict:
        try:
            st = classification.get("setup_type", SetupType.UNKNOWN)
            pt = classification.get("probability_tier", ProbabilityTier.C)
            entry = {
                "id": f"{ticker}_{int(time.time())}",
                "ticker": ticker,
                "timestamp": datetime.now(IST_TZ).strftime("%Y-%m-%d %H:%M IST"),
                "setup_type": st.value if hasattr(st, "value") else str(st),
                "grade": pt.value if hasattr(pt, "value") else str(pt),
                "confidence": round(classification.get("confidence", 0) * 100, 1),
                "entry_price": risk_data.get("entry"),
                "stop_loss": risk_data.get("stop_loss"),
                "target1": risk_data.get("target1"),
                "rr1": risk_data.get("rr1"),
                "action": action_cmd.get("command", "WATCH"),
                "regime": regime_label,
                "reasoning": reasoning,
                "status": "PENDING",
            }
            journal = []
            try:
                with open(Layer8Journal.JOURNAL_PATH, "r") as f:
                    journal = json.load(f)
            except Exception:
                pass
            journal.append(entry)
            journal = journal[-500:]
            try:
                with open(Layer8Journal.JOURNAL_PATH, "w") as f:
                    json.dump(journal, f, indent=2, default=str)
            except Exception:
                pass
            return entry
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def get_recent(limit: int = 20) -> List[Dict]:
        try:
            with open(Layer8Journal.JOURNAL_PATH, "r") as f:
                return list(reversed(json.load(f)[-limit:]))
        except Exception:
            return []



# ═══════════════════════════════════════════════════════════════════════
#  v1.1 NEW: INDIA VIX ENGINE
#  VIX regime gate: >20 reduce, >25 half, >30 stop new trades
# ═══════════════════════════════════════════════════════════════════════

class IndiaVIXEngine:
    """India VIX-based position sizing gate. ^INDIAVIX from yfinance."""
    LEVELS = [
        (30, 0.25, "EXTREME", "🔴"),
        (25, 0.50, "HIGH",    "🔴"),
        (20, 0.75, "ELEVATED","🟡"),
        (17, 0.90, "MODERATE","🟡"),
        (13, 1.00, "LOW",     "🟢"),
        (0,  1.00, "VERY_LOW","🟢"),
    ]

    @staticmethod
    def get_vix() -> Dict:
        try:
            df = _yf_download("^INDIAVIX", period="5d")
            if df.empty:
                return {"vix": 15.0, "level": "LOW", "size_multiplier": 1.0, "icon": "🟢", "note": "VIX unavailable — default used"}
            vix = float(df["Close"].squeeze().iloc[-1])
            for threshold, mult, level, icon in IndiaVIXEngine.LEVELS:
                if vix >= threshold:
                    note = (
                        f"⛔ EXTREME VIX {vix:.1f} — no new positions" if level == "EXTREME" else
                        f"⚠️ HIGH VIX {vix:.1f} — half position size" if level == "HIGH" else
                        f"⚠️ ELEVATED VIX {vix:.1f} — reduce size to 75%" if level == "ELEVATED" else
                        f"🟡 MODERATE VIX {vix:.1f} — size at 90%" if level == "MODERATE" else
                        f"✅ VIX {vix:.1f} — normal sizing"
                    )
                    return {"vix": round(vix, 2), "level": level, "size_multiplier": mult, "icon": icon, "note": note}
            return {"vix": round(vix, 2), "level": "VERY_LOW", "size_multiplier": 1.0, "icon": "🟢", "note": f"VIX {vix:.1f} — very low, normal sizing"}
        except Exception as e:
            return {"vix": 15.0, "level": "LOW", "size_multiplier": 1.0, "icon": "🟢", "note": str(e)}


# ═══════════════════════════════════════════════════════════════════════
#  v1.1 NEW: F&O EXPIRY GUARD
#  Weekly expiry every Thursday. Monthly = last Thursday.
#  Suppresses volume-based signals on expiry day.
# ═══════════════════════════════════════════════════════════════════════

class FNOExpiryGuard:
    """Detects NSE F&O expiry weeks and adjusts volume signal confidence."""

    @staticmethod
    def check() -> Dict:
        try:
            today = datetime.now(IST_TZ)
            weekday = today.weekday()  # 0=Mon, 3=Thu, 6=Sun
            day_of_month = today.day

            # Find last Thursday of this month
            import calendar
            _, last_day = calendar.monthrange(today.year, today.month)
            last_thu = last_day - ((calendar.weekday(today.year, today.month, last_day) - 3) % 7)

            is_weekly_expiry_day  = weekday == 3  # Thursday
            is_monthly_expiry_day = weekday == 3 and day_of_month == last_thu
            days_to_weekly_expiry = (3 - weekday) % 7  # days until next Thursday
            is_expiry_week = days_to_weekly_expiry <= 2  # Mon-Thu of expiry week

            if is_monthly_expiry_day:
                label = "MONTHLY EXPIRY DAY"
                vol_confidence = 0.4   # volume today is F&O rollover noise
                caution = "⚠️ Monthly F&O Expiry — volume unreliable. Avoid new entries today."
            elif is_weekly_expiry_day:
                label = "WEEKLY EXPIRY DAY"
                vol_confidence = 0.55
                caution = "⚠️ Weekly F&O Expiry — intraday vol distorted. Wait for 2:30 PM move."
            elif is_expiry_week:
                label = "EXPIRY WEEK"
                vol_confidence = 0.75
                caution = "🟡 Expiry week — expect elevated vol noise Thu. Plan around it."
            else:
                label = "NORMAL"
                vol_confidence = 1.0
                caution = None

            return {
                "label": label,
                "is_expiry_day": is_weekly_expiry_day,
                "is_monthly_expiry": is_monthly_expiry_day,
                "is_expiry_week": is_expiry_week,
                "days_to_weekly_expiry": days_to_weekly_expiry,
                "vol_confidence_multiplier": vol_confidence,
                "caution": caution,
            }
        except Exception as e:
            return {"label": "NORMAL", "vol_confidence_multiplier": 1.0, "caution": None, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
#  v1.1 NEW: DELIVERY QUALITY ESTIMATOR
#  Proxy for delivery % using close position in range + consistency.
#  Actual CDSL/NSE delivery data not available via yfinance.
# ═══════════════════════════════════════════════════════════════════════

class DeliveryQualityEstimator:
    """
    Proxy for delivery-based buying quality.
    High delivery = genuine accumulation (not intraday speculation).
    Estimated via: close position in range + multi-day consistency.
    Score 0-100. >= 60 = good delivery quality.
    """

    @staticmethod
    def estimate(df: pd.DataFrame) -> Dict:
        try:
            close  = df["Close"].squeeze()
            high   = df["High"].squeeze()
            low    = df["Low"].squeeze()
            volume = df["Volume"].squeeze()

            # Metric 1: Close position in daily range (0=low, 1=high)
            # High close-position = buyers held till end of day = delivery
            range_ = (high - low)
            close_pos = ((close - low) / range_.replace(0, np.nan)).fillna(0.5)
            avg_close_pos = close_pos.tail(5).mean()

            # Metric 2: Up-days with above-avg volume (institutional buying pattern)
            avg_vol = volume.tail(20).mean()
            up_days_high_vol = ((close > close.shift(1)) & (volume > avg_vol)).tail(5).sum()

            # Metric 3: Body consistency (large bodies = decisive moves = delivery)
            body_size = abs(close - df["Open"].squeeze())
            body_ratio = (body_size / range_.replace(0, np.nan)).fillna(0).tail(5).mean()

            # Metric 4: Volume trend during price rise (accumulation signature)
            recent_price_up = close.iloc[-1] > close.iloc[-5] if len(close) >= 5 else False
            vol_rising = volume.tail(3).mean() > volume.tail(10).mean()
            accumulation_sig = recent_price_up and vol_rising

            # Composite score
            score = (avg_close_pos * 35) + (up_days_high_vol / 5 * 30) + (body_ratio * 25) + (10 if accumulation_sig else 0)
            score = min(100, max(0, score * 100 if score < 1 else score))

            quality = "HIGH" if score >= 65 else ("MEDIUM" if score >= 40 else "LOW")
            return {
                "delivery_score": round(score, 1),
                "quality": quality,
                "avg_close_position": round(float(avg_close_pos), 3),
                "up_days_high_vol": int(up_days_high_vol),
                "body_consistency": round(float(body_ratio), 3),
                "accumulation_signal": accumulation_sig,
                "is_ok": score >= 40,
                "note": f"✅ Delivery quality {quality} ({score:.0f}/100)" if score >= 40 else f"⚠️ Low delivery quality ({score:.0f}/100) — intraday speculation"
            }
        except Exception as e:
            return {"delivery_score": 50, "quality": "MEDIUM", "is_ok": True, "note": str(e)}


# ═══════════════════════════════════════════════════════════════════════
#  v1.1 NEW: WEEKLY TREND CONFIRMATION
#  Weekly EMA20 must be rising. Weekly RSI must be in bullish zone.
#  Prevents buying stocks that are in weekly downtrends.
# ═══════════════════════════════════════════════════════════════════════

class WeeklyTrendConfirmation:
    """Weekly chart alignment — prevents buying into weekly downtrends."""

    @staticmethod
    def check(df: pd.DataFrame) -> Dict:
        try:
            close_d = df["Close"].squeeze()
            # Resample to weekly
            close_w = close_d.resample("W").last().dropna()
            if len(close_w) < 12:
                return {"weekly_ok": True, "note": "Insufficient weekly data", "weekly_rsi": 50}

            ema20w   = close_w.ewm(span=20, adjust=False).mean()
            ema10w   = close_w.ewm(span=10, adjust=False).mean()

            # Weekly RSI (simplified)
            delta = close_w.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, np.nan)
            rsi_w = (100 - 100 / (1 + rs)).iloc[-1]

            ema20w_rising = ema20w.iloc[-1] > ema20w.iloc[-3]
            ema10w_rising = ema10w.iloc[-1] > ema10w.iloc[-3]
            price_above_weekly_ema = close_w.iloc[-1] > ema20w.iloc[-1]
            weekly_rsi_ok = rsi_w >= 40  # Not in weekly downtrend

            weekly_ok = ema20w_rising and price_above_weekly_ema and weekly_rsi_ok

            return {
                "weekly_ok": weekly_ok,
                "ema20w_rising": ema20w_rising,
                "ema10w_rising": ema10w_rising,
                "price_above_weekly_ema": price_above_weekly_ema,
                "weekly_rsi": round(float(rsi_w), 1),
                "weekly_ema20": round(float(ema20w.iloc[-1]), 2),
                "note": "✅ Weekly trend confirmed" if weekly_ok else "⚠️ Weekly trend weak — do not buy",
            }
        except Exception as e:
            return {"weekly_ok": True, "note": str(e), "weekly_rsi": 50}


# ═══════════════════════════════════════════════════════════════════════
#  v1.1 NEW: 52-WEEK HIGH BREAKOUT DETECTOR
#  Most reliable India setup. Institutional momentum buying triggers here.
# ═══════════════════════════════════════════════════════════════════════

class FiftyTwoWeekHighDetector:
    """Detects new 12-week high breakout — strongest signal in Indian markets."""

    @staticmethod
    def check(df: pd.DataFrame) -> Dict:
        try:
            close  = df["Close"].squeeze()
            high   = df["High"].squeeze()
            volume = df["Volume"].squeeze()

            # 12-week high from rolling 60 bars
            lookback = min(60, len(close) - 2)
            high_12w = high.iloc[-(lookback+1):-1].max()
            curr_price = close.iloc[-1]
            prev_close = close.iloc[-2]
            avg_vol = volume.tail(20).mean()
            vol_ratio = volume.iloc[-1] / avg_vol if avg_vol > 0 else 1

            # New 12W high: today's close > previous 12W high
            is_new_12w_high = curr_price > high_12w
            # Near 12W high: within 2% (about to break)
            near_12w_high = (high_12w - curr_price) / high_12w * 100 <= 2.0
            # Distance from 12W high
            dist_pct = round((curr_price - high_12w) / high_12w * 100, 2)

            # ATH check (all-time in available data)
            ath = high.max()
            is_ath = curr_price >= ath * 0.995

            return {
                "is_new_12w_high": is_new_12w_high,
                "near_12w_high": near_12w_high,
                "is_ath": is_ath,
                "high_12w": round(float(high_12w), 2),
                "dist_from_12w_pct": dist_pct,
                "volume_on_breakout": round(vol_ratio, 2),
                "valid_breakout": is_new_12w_high and vol_ratio >= 1.2,
                "note": (
                    "🚀 ATH BREAKOUT — strongest signal!" if is_ath and is_new_12w_high else
                    "✅ New 12W High — institutional momentum buy signal" if is_new_12w_high else
                    f"⏳ Near 12W High — {abs(dist_pct):.1f}% away, watch for break" if near_12w_high else
                    f"📊 {abs(dist_pct):.1f}% below 12W High"
                ),
            }
        except Exception as e:
            return {"is_new_12w_high": False, "valid_breakout": False, "note": str(e)}


# ═══════════════════════════════════════════════════════════════════════
#  v1.1 NEW: RESULTS MOMENTUM DETECTOR
#  Post-earnings gap + hold setup. Very reliable in India Q-result seasons.
# ═══════════════════════════════════════════════════════════════════════

class ResultsMomentumDetector:
    """
    Detects post-earnings gap-and-hold pattern.
    If stock gapped >5% on any day in last 10 sessions and held,
    it's a Results Momentum candidate.
    """

    @staticmethod
    def detect(df: pd.DataFrame) -> Dict:
        try:
            close  = df["Close"].squeeze()
            open_  = df["Open"].squeeze()
            low    = df["Low"].squeeze()
            volume = df["Volume"].squeeze()
            avg_vol = volume.tail(20).mean()

            # Scan last 10 days for a gap event
            results_gap_day = None
            gap_pct = 0.0

            for i in range(-10, -1):
                gap = (open_.iloc[i] - close.iloc[i-1]) / close.iloc[i-1] * 100
                if gap >= 5.0 and volume.iloc[i] >= avg_vol * 1.5:
                    results_gap_day = len(close) + i
                    gap_pct = gap
                    break

            if results_gap_day is None:
                return {"detected": False, "gap_pct": 0, "note": "No results gap detected in last 10 sessions"}

            # Check if price held above 50% of gap zone since then
            gap_open  = float(open_.iloc[results_gap_day - len(close)])
            gap_ref   = float(close.iloc[results_gap_day - len(close) - 1])
            gap_50pct = gap_ref + (gap_open - gap_ref) * 0.5
            days_since = abs(results_gap_day - len(close)) + 1  # simplified
            price_held = close.iloc[-1] >= gap_50pct
            vol_normalized = volume.iloc[-1] < volume.iloc[results_gap_day - len(close)] * 0.8  # vol quieting down

            return {
                "detected": True,
                "gap_pct": round(gap_pct, 2),
                "gap_50pct_level": round(gap_50pct, 2),
                "price_held": price_held,
                "vol_normalized": vol_normalized,
                "valid_setup": price_held and vol_normalized,
                "note": (
                    f"✅ Results Momentum: {gap_pct:.1f}% gap held — vol normalizing, continuation likely" if price_held and vol_normalized else
                    f"⚠️ {gap_pct:.1f}% gap detected but price filling gap — avoid" if not price_held else
                    f"📊 {gap_pct:.1f}% gap detected, vol still high — wait for normalization"
                ),
            }
        except Exception as e:
            return {"detected": False, "gap_pct": 0, "note": str(e)}


# ═══════════════════════════════════════════════════════════════════════
#  v1.1 NEW: OPERATOR ALERT DETECTOR
#  Detects mid/smallcap manipulation patterns (operator-driven stocks).
#  These stocks show huge moves on thin liquidity — avoid.
# ═══════════════════════════════════════════════════════════════════════

class OperatorAlertDetector:
    """
    Operator-driven stock detection for NSE mid/smallcap.
    Signals: thin avg volume + sudden 5x+ spikes + price far from all EMAs.
    """

    @staticmethod
    def check(df: pd.DataFrame, market_cap_cat: str = "midcap") -> Dict:
        try:
            close  = df["Close"].squeeze()
            high   = df["High"].squeeze()
            low    = df["Low"].squeeze()
            volume = df["Volume"].squeeze()

            if market_cap_cat == "largecap":
                return {"is_operator": False, "risk_score": 0, "note": "Large cap — operator risk minimal"}

            avg_vol_20 = volume.tail(20).mean()
            avg_vol_5  = volume.tail(5).mean()
            max_vol_20 = volume.tail(20).max()
            curr_vol   = volume.iloc[-1]

            # Signal 1: Average volume is thin (< 1 lakh shares/day for midcap)
            thin_liquidity = avg_vol_20 < 100000

            # Signal 2: Sudden massive volume spike (5x+ in one day)
            vol_spike_ratio = max_vol_20 / avg_vol_20 if avg_vol_20 > 0 else 1
            suspicious_spike = vol_spike_ratio > 5.0

            # Signal 3: Price far from all EMAs simultaneously (operator pump)
            ema20 = ta.ema(close, 20).iloc[-1]
            ema50 = ta.ema(close, 50).iloc[-1]
            cp = close.iloc[-1]
            far_from_ema20 = (cp - ema20) / ema20 * 100 > 20 if ema20 > 0 else False
            far_from_ema50 = (cp - ema50) / ema50 * 100 > 30 if ema50 > 0 else False

            # Signal 4: Price range instability (huge candles = pump volatility)
            ranges_pct = ((high - low) / close * 100).tail(10)
            extreme_ranges = (ranges_pct > 8).sum()
            range_instability = extreme_ranges >= 3

            # Signal 5: Circuit limit proximity (stock hitting consecutive upper circuits)
            consecutive_green = (close.diff().tail(5) > 0).sum()
            possible_circuit_run = consecutive_green == 5 and vol_spike_ratio > 3

            operator_score = (
                (30 if thin_liquidity else 0) +
                (25 if suspicious_spike else 0) +
                (20 if far_from_ema20 else 0) +
                (15 if range_instability else 0) +
                (10 if possible_circuit_run else 0)
            )

            is_operator = operator_score >= 50

            return {
                "is_operator": is_operator,
                "risk_score": operator_score,
                "thin_liquidity": thin_liquidity,
                "vol_spike_ratio": round(vol_spike_ratio, 1),
                "far_from_ema20": far_from_ema20,
                "range_instability": range_instability,
                "possible_circuit_run": possible_circuit_run,
                "note": (
                    f"🚨 OPERATOR ALERT: Risk score {operator_score}/100 — avoid this stock" if is_operator else
                    f"⚠️ Moderate operator risk ({operator_score}/100) — use small size" if operator_score >= 25 else
                    "✅ No operator signals detected"
                ),
            }
        except Exception as e:
            return {"is_operator": False, "risk_score": 0, "note": str(e)}


# ═══════════════════════════════════════════════════════════════════════
#  v1.1 NEW: GLOBAL CUES ENGINE
#  Asian market context for pre-market direction assessment.
# ═══════════════════════════════════════════════════════════════════════

class GlobalCuesEngine:
    """
    Fetches Asian/Global indices for pre-market context.
    Nikkei, Hang Seng, SGX Nifty proxy, US futures proxy.
    """
    INDICES = {
        "Nikkei 225":   "^N225",
        "Hang Seng":    "^HSI",
        "KOSPI":        "^KS11",
        "Nasdaq Fut":   "NQ=F",
        "S&P500 Fut":   "ES=F",
        "Crude Oil":    "CL=F",
        "USD/INR":      "USDINR=X",
    }

    @staticmethod
    def get_cues() -> Dict:
        try:
            results = {}
            positive_count = 0
            negative_count = 0

            for name, ticker in GlobalCuesEngine.INDICES.items():
                try:
                    df = _yf_download(ticker, period="3d")
                    if df.empty or len(df) < 2:
                        continue
                    close = df["Close"].squeeze()
                    chg = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
                    val = float(close.iloc[-1])
                    results[name] = {
                        "value": round(val, 2),
                        "change_pct": round(float(chg), 2),
                        "direction": "UP" if chg > 0 else "DOWN",
                        "icon": "🟢" if chg > 0.3 else ("🔴" if chg < -0.3 else "⚪"),
                    }
                    if chg > 0.3: positive_count += 1
                    elif chg < -0.3: negative_count += 1
                except Exception:
                    continue

            # Nifty opening bias
            if positive_count >= 4:
                bias = "POSITIVE"; bias_strength = "Strong"
            elif positive_count >= 2:
                bias = "POSITIVE"; bias_strength = "Moderate"
            elif negative_count >= 4:
                bias = "NEGATIVE"; bias_strength = "Strong"
            elif negative_count >= 2:
                bias = "NEGATIVE"; bias_strength = "Moderate"
            else:
                bias = "NEUTRAL"; bias_strength = "Mixed"

            return {
                "indices": results,
                "nifty_open_bias": bias,
                "bias_strength": bias_strength,
                "positive_markets": positive_count,
                "negative_markets": negative_count,
                "summary": f"{bias_strength} {bias} global cues — {positive_count} markets up, {negative_count} down",
            }
        except Exception as e:
            return {"indices": {}, "nifty_open_bias": "NEUTRAL", "bias_strength": "Unknown",
                    "summary": "Global cues unavailable", "error": str(e)}


class BreakoutConfirmation:
    """
    2-day breakout confirmation with v4 Wick Rejection integration.
    """

    @staticmethod
    def check(df: pd.DataFrame) -> Dict:
        try:
            close = df['Close'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            open_ = df['Open'].squeeze()
            volume = df['Volume'].squeeze()

            resistance = high.iloc[:-2].tail(20).max()
            avg_volume = volume.iloc[:-2].tail(20).mean()
            vol_std = volume.iloc[:-2].tail(20).std()

            vol_zscore_d1 = ((volume.iloc[-2] - avg_volume) / vol_std) if vol_std > 0 else 0
            vol_zscore_d2 = ((volume.iloc[-1] - avg_volume) / vol_std) if vol_std > 0 else 0
            vol_breakout_quality = (
                'Strong' if vol_zscore_d1 > 2.0 else
                'Good' if vol_zscore_d1 > 1.0 else
                'Weak'
            )

            d1_volume_ratio = volume.iloc[-2] / avg_volume if avg_volume > 0 else 1
            d1_close = close.iloc[-2]
            d1_broke = d1_close > resistance and vol_zscore_d1 > 0.5

            d2_close = close.iloc[-1]
            d2_held = d2_close > resistance

            confirmed = d1_broke and d2_held

            breakout_midpoint = (high.iloc[-2] + low.iloc[-2]) / 2
            follow_through_ok = d2_close > breakout_midpoint
            vol_contracting_d2 = vol_zscore_d2 < vol_zscore_d1

            # Bearish engulfing guard
            bearish_engulf = False
            if len(df) >= 3:
                for i in range(-3, 0):
                    body_prev = close.iloc[i-1] - open_.iloc[i-1]
                    body_curr = open_.iloc[i] - close.iloc[i]
                    if body_prev > 0 and body_curr > 0 and body_curr > body_prev:
                        bearish_engulf = True
                        break

            # V4: Wick Rejection Check on breakout day
            wick_check = WickTrapFilter.check(df)
            wick_rejection_ok = not wick_check['is_trap']

            # Enhanced confirmation with wick rejection
            confirmed_strict = confirmed and follow_through_ok and not bearish_engulf and wick_rejection_ok

            return {
                'resistance_level': round(resistance, 2),
                'day1_close': round(d1_close, 2),
                'day1_volume_ratio': round(d1_volume_ratio, 2),
                'day1_volume_zscore': round(vol_zscore_d1, 2),
                'day1_broke_out': d1_broke,
                'day2_close': round(d2_close, 2),
                'day2_held': d2_held,
                'confirmed': confirmed,
                'confirmed_strict': confirmed_strict,
                'follow_through_ok': follow_through_ok,
                'vol_contraction_pullback': vol_contracting_d2,
                'volume_quality': vol_breakout_quality,
                'bearish_engulf_detected': bearish_engulf,
                'wick_rejection_ok': wick_rejection_ok,
                'wick_analysis': wick_check,
                'status': (
                    '✅ Strict Confirmed' if confirmed_strict else
                    '⚠️ Confirmed (wick concern)' if confirmed and not wick_rejection_ok else
                    '⚠️ Confirmed (weak follow-through)' if confirmed else
                    ('⏳ Pending D2' if d1_broke else '❌ Not confirmed')
                ),
            }
        except Exception:
            return {'confirmed': False, 'confirmed_strict': False,
                    'status': '❌ Error checking confirmation'}


class RSIClassifier:
    """Flexible RSI classification including Power Play zone."""

    @staticmethod
    def classify(rsi_value: float) -> Dict:
        if rsi_value >= 80:
            zone = "extreme_overbought"
            strength = "Exhaustion likely"
            power_play = False
        elif rsi_value >= 72:
            zone = "super_momentum"
            strength = "Power Play zone — Indian bull runs stay here"
            power_play = True
        elif rsi_value >= 60:
            zone = "strong"
            strength = "Healthy momentum"
            power_play = False
        elif rsi_value >= 45:
            zone = "neutral"
            strength = "No clear bias"
            power_play = False
        elif rsi_value >= 30:
            zone = "weak"
            strength = "Pullback zone"
            power_play = False
        else:
            zone = "oversold"
            strength = "Potential reversal"
            power_play = False

        if 55 <= rsi_value <= 72:
            rsi_score = 100
        elif 72 < rsi_value <= 80:
            rsi_score = 85
        elif 45 <= rsi_value < 55:
            rsi_score = 70
        elif rsi_value > 80:
            rsi_score = 30
        elif 30 <= rsi_value < 45:
            rsi_score = 60
        else:
            rsi_score = 40

        return {
            'value': round(rsi_value, 2),
            'zone': zone,
            'strength': strength,
            'is_power_play': power_play,
            'rsi_score': rsi_score,
        }


class SetupClassifier:
    """
    Enhanced classifier with Trap Detection and Wick Rejection.
    Never returns UNKNOWN — always returns best setup with tier adjustment.
    """

    @staticmethod
    def classify_setup(df: pd.DataFrame, ticker: str,
                       sector: str = "", market_cap_cat: str = "midcap",
                       days_to_earnings: int = 999, beta: float = 1.0,
                       sector_data: Dict = None) -> Dict:
        try:
            close = df['Close'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            volume = df['Volume'].squeeze()

            ema20 = ta.ema(close, 20)
            ema50 = ta.ema(close, 50)
            rsi = ta.rsi(close, 14)
            atr = ta.atr(high, low, close, 14)

            curr_price = close.iloc[-1]
            curr_rsi = rsi.iloc[-1]
            curr_atr = atr.iloc[-1]

            recent_high_20 = high.tail(20).max()
            recent_low_20 = low.tail(20).min()
            range_pct = ((recent_high_20 - recent_low_20) / recent_low_20) * 100

            dist_from_ema20 = ((curr_price - ema20.iloc[-1]) / ema20.iloc[-1]) * 100

            avg_volume_20 = volume.tail(20).mean()
            curr_volume = volume.iloc[-1]
            volume_ratio = curr_volume / avg_volume_20 if avg_volume_20 > 0 else 1

            atr_20_ago = atr.iloc[-20] if len(atr) >= 20 else curr_atr
            atr_change_pct = ((curr_atr - atr_20_ago) / atr_20_ago) * 100 if atr_20_ago > 0 else 0
            is_contracting = atr_change_pct < -10

            rsi_info = RSIClassifier.classify(curr_rsi)
            rs_data = MultiTimeframeRS.calculate(close)
            liq_data = LiquidityFilter.check_liquidity(df, market_cap_cat)
            gap_data = GapRiskModel.calculate(df, days_to_earnings, beta)
            sector_rs = SectorLeadershipEngine.get_sector_rs_for_stock(sector, sector_data)
            sector_rs_positive = sector_rs.get('is_leading', False)
            breakout_confirm = BreakoutConfirmation.check(df)
            trap_analysis = TrapDetector.analyze(df)

            # SCORING
            breakout_score = 0
            pullback_score = 0
            momentum_score = 0
            power_play_score = 0

            # BREAKOUT signals
            near_resistance = curr_price >= recent_high_20 * 0.98
            if near_resistance: breakout_score += 25
            if is_contracting: breakout_score += 20
            if volume_ratio > 1.5: breakout_score += 15
            if range_pct < 8: breakout_score += 10
            if abs(dist_from_ema20) < 3: breakout_score += 5
            if sector_rs_positive: breakout_score += 10
            if liq_data['is_liquid']: breakout_score += 5
            if breakout_confirm['confirmed_strict']: breakout_score += 10

            # PULLBACK signals
            is_pullback = -5 <= dist_from_ema20 <= -1.5
            ema_structure_good = ema20.iloc[-1] > ema50.iloc[-1]
            if is_pullback: pullback_score += 25
            if ema_structure_good: pullback_score += 20
            if 35 <= curr_rsi <= 55: pullback_score += 15
            if volume_ratio < 1.0: pullback_score += 10
            if ema20.iloc[-1] > ema20.iloc[-5]: pullback_score += 5
            if sector_rs_positive: pullback_score += 10
            if liq_data['is_liquid']: pullback_score += 5

            # MOMENTUM signals
            strong_uptrend = curr_price > ema20.iloc[-1] > ema50.iloc[-1]
            if strong_uptrend: momentum_score += 25
            if 55 <= curr_rsi <= 80: momentum_score += 20
            if dist_from_ema20 > 3: momentum_score += 15
            if volume_ratio > 1.2: momentum_score += 10
            momentum_5d = ((curr_price - close.iloc[-5]) / close.iloc[-5]) * 100
            if momentum_5d > 3: momentum_score += 5
            if rs_data['composite'] > 4: momentum_score += 10
            if sector_rs_positive: momentum_score += 10
            if liq_data['is_liquid']: momentum_score += 5

            # POWER PLAY signals
            if rsi_info['is_power_play']:
                power_play_score += 30
            if strong_uptrend: power_play_score += 20
            if rs_data['composite'] > 6: power_play_score += 20
            if volume_ratio > 1.3: power_play_score += 10
            if sector_rs_positive: power_play_score += 10
            if liq_data['is_liquid']: power_play_score += 5
            if momentum_5d > 5: power_play_score += 5

            # DETERMINE SETUP
            scores = {
                SetupType.BREAKOUT: breakout_score,
                SetupType.PULLBACK: pullback_score,
                SetupType.MOMENTUM: momentum_score,
                SetupType.POWER_PLAY: power_play_score,
            }

            setup_type = max(scores, key=scores.get)
            max_possible = 100
            confidence = scores[setup_type] / max_possible

            if confidence >= 0.75:
                tier = ProbabilityTier.A_PLUS
            elif confidence >= 0.60:
                tier = ProbabilityTier.A
            elif confidence >= 0.45:
                tier = ProbabilityTier.B
            else:
                tier = ProbabilityTier.C

            if confidence >= 0.60 and liq_data['is_liquid']:
                status = TradeStatus.READY
            elif confidence >= 0.40:
                status = TradeStatus.WATCH
            else:
                status = TradeStatus.AVOID

            # Downgrade if sector RS negative for breakout
            if setup_type == SetupType.BREAKOUT and not sector_rs_positive:
                if tier in (ProbabilityTier.A_PLUS, ProbabilityTier.A):
                    tier = ProbabilityTier.B
                    status = TradeStatus.WATCH

            # v1.0: Trap Suppression — >40% trap probability → AVOID (hidden from dashboard)
            if trap_analysis['trap_probability'] > 40:
                status = TradeStatus.AVOID
                tier = ProbabilityTier.C
            elif trap_analysis['is_high_risk_trap']:
                if tier in (ProbabilityTier.A_PLUS, ProbabilityTier.A):
                    tier = ProbabilityTier.B
                if status == TradeStatus.READY:
                    status = TradeStatus.WATCH

            # v1.1: Intraday Chase Guard — India-calibrated: 5% large, 8% midcap
            chase_threshold = 8.0 if market_cap_cat == 'midcap' else (10.0 if market_cap_cat == 'smallcap' else 5.0)
            if dist_from_ema20 > chase_threshold and status == TradeStatus.READY:
                status = TradeStatus.WATCH
                if tier == ProbabilityTier.A_PLUS:
                    tier = ProbabilityTier.A

            # V4: Apply Wick Rejection penalty for breakouts
            if setup_type == SetupType.BREAKOUT and not breakout_confirm.get('wick_rejection_ok', True):
                if tier in (ProbabilityTier.A_PLUS, ProbabilityTier.A):
                    tier = ProbabilityTier.B
                status = TradeStatus.WATCH

            # HARD KILL CONDITIONS
            hard_kills = []
            kill_triggered = False

            if gap_data.get('days_to_earnings', 999) <= 1:
                hard_kills.append("⛔ Earnings tomorrow — avoid new entry")
                kill_triggered = True
            if gap_data.get('risk_level') == 'HIGH' and setup_type == SetupType.BREAKOUT:
                hard_kills.append("⛔ High gap risk + breakout = operator trap zone")
                kill_triggered = True
            if not liq_data['is_liquid']:
                hard_kills.append("⛔ Insufficient liquidity for safe exit")
                kill_triggered = True
            if gap_data.get('atr_gap_multiplier_triggered', False) and beta > 1.3:
                hard_kills.append("⛔ Liquidity vacuum risk on high-beta stock")
                kill_triggered = True
            if trap_analysis['trap_probability'] >= 70:
                hard_kills.append("⛔ High trap probability detected")
                kill_triggered = True

            if kill_triggered:
                tier = ProbabilityTier.C
                status = TradeStatus.AVOID

            # TREND CONTEXT TAG
            try:
                ema100 = ta.ema(close, 100)
                dist_from_ema100 = ((curr_price - ema100.iloc[-1]) / ema100.iloc[-1]) * 100
                rs_composite_val = rs_data.get('composite', 0)
                if dist_from_ema100 < 5 and rs_composite_val < 3:
                    trend_context = "Early_Trend"
                elif dist_from_ema100 < 15 and rs_composite_val < 8:
                    trend_context = "Mid_Trend"
                else:
                    trend_context = "Late_Trend"
                if trend_context == "Late_Trend" and setup_type == SetupType.BREAKOUT:
                    if status == TradeStatus.READY:
                        status = TradeStatus.WATCH
                    if tier == ProbabilityTier.A_PLUS:
                        tier = ProbabilityTier.A
            except Exception:
                trend_context = "Mid_Trend"

            return {
                'setup_type': setup_type,
                'confidence': round(confidence, 3),
                'probability_tier': tier,
                'status': status,
                'scores': {k.value: v for k, v in scores.items()},
                'rsi': rsi_info,
                'rs': rs_data,
                'liquidity': liq_data,
                'gap_risk': gap_data,
                'sector_rs': sector_rs,
                'sector_rs_positive': sector_rs_positive,
                'breakout_confirmation': breakout_confirm,
                'trap_analysis': trap_analysis,
                'volume_ratio': round(volume_ratio, 2),
                'dist_from_ema20': round(dist_from_ema20, 2),
                'is_contracting': is_contracting,
                'allocation_r': TIER_ALLOCATION[tier] * gap_data.get('position_size_factor', 1.0),
                'trend_context': trend_context,
                'hard_kill_conditions': hard_kills,
                'kill_triggered': kill_triggered,
            }

        except Exception as e:
            logger.error(f"Classification error for {ticker}: {e}")
            return {
                'setup_type': SetupType.UNKNOWN,
                'confidence': 0.0,
                'probability_tier': ProbabilityTier.C,
                'status': TradeStatus.AVOID,
                'scores': {},
                'error': str(e),
            }

    classify = classify_setup


# ═══════════════════════════════════════════════════════════════════════
#  LAYER 3b: SETUP SCORER
# ═══════════════════════════════════════════════════════════════════════

class SetupScorer:
    """Score setup with sector RS, breadth, and liquidity weights."""

    @staticmethod
    def score_setup(df: pd.DataFrame, setup_type: SetupType,
                    benchmark_ticker: str = "^NSEI",
                    sector_rs_positive: bool = False,
                    breadth_healthy: bool = True,
                    liquidity_score: float = 50) -> Dict:

        if setup_type == SetupType.BREAKOUT:
            return SetupScorer._score_breakout(df, benchmark_ticker, sector_rs_positive, breadth_healthy, liquidity_score)
        elif setup_type == SetupType.PULLBACK:
            return SetupScorer._score_pullback(df, benchmark_ticker, sector_rs_positive, breadth_healthy, liquidity_score)
        elif setup_type in (SetupType.MOMENTUM, SetupType.POWER_PLAY):
            return SetupScorer._score_momentum(df, benchmark_ticker, sector_rs_positive, breadth_healthy, liquidity_score, setup_type)
        else:
            return {'total_score': 0, 'details': {}, 'passed_filters': False}

    @staticmethod
    def _score_breakout(df, bench, sector_ok, breadth_ok, liq_score) -> Dict:
        config = SETUP_CONFIGS[SetupType.BREAKOUT]
        weights = config.scoring_weights
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        volume = df['Volume'].squeeze()
        atr = ta.atr(high, low, close, 14)
        curr_price = close.iloc[-1]
        curr_atr = atr.iloc[-1]
        avg_volume = volume.tail(20).mean()
        curr_volume = volume.iloc[-1]

        scores = {}
        mandatory_passed = True

        atr_20_ago = atr.iloc[-20] if len(atr) >= 20 else curr_atr
        atr_change = ((curr_atr - atr_20_ago) / atr_20_ago) * 100 if atr_20_ago > 0 else 0
        is_contracting = atr_change < -10
        scores['volatility_contraction'] = weights['volatility_contraction'] if is_contracting else 0
        if not is_contracting: mandatory_passed = False

        volume_ratio = curr_volume / avg_volume if avg_volume > 0 else 1
        scores['volume_expansion'] = min(weights['volume_expansion'], weights['volume_expansion'] * (volume_ratio / 1.5))
        if volume_ratio < 1.5: mandatory_passed = False

        recent_high = high.tail(20).max()
        res_dist = (recent_high - curr_price) / curr_atr if curr_atr > 0 else 0
        scores['location_quality'] = weights['location_quality'] if res_dist >= 1.5 else 0

        regime_info = MarketRegimeEngine.detect_regime()
        regime_aligned = regime_info['type'] != RegimeType.RISK_OFF
        scores['regime_alignment'] = weights['regime_alignment'] if regime_aligned else 0
        if not regime_aligned: mandatory_passed = False

        candle_body = abs(close.iloc[-1] - df['Open'].iloc[-1])
        candle_range = high.iloc[-1] - low.iloc[-1]
        body_ratio = candle_body / candle_range if candle_range > 0 else 0
        scores['candle_anatomy'] = weights['candle_anatomy'] * body_ratio

        rs_data = MultiTimeframeRS.calculate(close, bench)
        scores['relative_strength'] = min(weights['relative_strength'],
                                          weights['relative_strength'] * max(0, rs_data['composite']) / 10)

        scores['sector_rs'] = weights['sector_rs'] if sector_ok else 0
        scores['breadth_confirm'] = weights['breadth_confirm'] if breadth_ok else 0
        scores['liquidity_score'] = weights['liquidity_score'] * (liq_score / 100)

        total = sum(scores.values())
        return {
            'total_score': round(total, 2),
            'max_score': sum(weights.values()),
            'scores': {k: round(v, 2) for k, v in scores.items()},
            'passed_filters': mandatory_passed,
            'details': {
                'volatility_contraction': is_contracting,
                'volume_ratio': round(volume_ratio, 2),
                'resistance_distance_atr': round(res_dist, 2),
                'regime': regime_info['type'].value,
                'rs_composite': rs_data['composite'],
            }
        }

    @staticmethod
    def _score_pullback(df, bench, sector_ok, breadth_ok, liq_score) -> Dict:
        config = SETUP_CONFIGS[SetupType.PULLBACK]
        weights = config.scoring_weights
        close = df['Close'].squeeze()
        volume = df['Volume'].squeeze()
        ema20 = ta.ema(close, 20)
        ema50 = ta.ema(close, 50)
        rsi = ta.rsi(close, 14)
        curr_price = close.iloc[-1]
        curr_rsi = rsi.iloc[-1]

        scores = {}
        mandatory_passed = True

        ema_structure = ema20.iloc[-1] > ema50.iloc[-1]
        ema50_rising = ema50.iloc[-1] > ema50.iloc[-5]
        scores['trend_structure'] = weights['trend_structure'] if (ema_structure and ema50_rising) else 0
        if not (ema_structure and ema50_rising): mandatory_passed = False

        dist = ((curr_price - ema20.iloc[-1]) / ema20.iloc[-1]) * 100
        scores['pullback_quality'] = weights['pullback_quality'] if -5 <= dist <= -1 else 0

        avg_vol = volume.tail(20).mean()
        vol_ratio = volume.iloc[-1] / avg_vol if avg_vol > 0 else 1
        scores['volume_pattern'] = min(weights['volume_pattern'], weights['volume_pattern'] * vol_ratio) if vol_ratio > 1.1 else 0
        scores['weekly_alignment'] = weights['weekly_alignment'] * 0.8

        rsi_ok = 35 <= curr_rsi <= 55
        scores['rsi_zone'] = weights['rsi_zone'] if rsi_ok else 0

        regime_info = MarketRegimeEngine.detect_regime()
        scores['regime'] = weights['regime'] * (regime_info['score'] / 100)

        scores['sector_rs'] = weights['sector_rs'] if sector_ok else 0
        scores['breadth_confirm'] = weights['breadth_confirm'] if breadth_ok else 0
        scores['liquidity_score'] = weights['liquidity_score'] * (liq_score / 100)

        total = sum(scores.values())
        return {
            'total_score': round(total, 2),
            'max_score': sum(weights.values()),
            'scores': {k: round(v, 2) for k, v in scores.items()},
            'passed_filters': mandatory_passed,
            'details': {
                'ema_structure': ema_structure,
                'ema50_rising': ema50_rising,
                'pullback_distance': round(dist, 2),
                'rsi': round(curr_rsi, 2),
            }
        }

    @staticmethod
    def _score_momentum(df, bench, sector_ok, breadth_ok, liq_score, setup_type) -> Dict:
        config = SETUP_CONFIGS.get(setup_type, SETUP_CONFIGS[SetupType.MOMENTUM])
        weights = config.scoring_weights
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        volume = df['Volume'].squeeze()
        ema20 = ta.ema(close, 20)
        ema50 = ta.ema(close, 50)
        rsi = ta.rsi(close, 14)
        curr_price = close.iloc[-1]
        curr_rsi = rsi.iloc[-1]

        scores = {}
        mandatory_passed = True

        rs_data = MultiTimeframeRS.calculate(close, bench)
        scores['relative_strength'] = min(weights['relative_strength'],
                                          weights['relative_strength'] * rs_data['composite'] / 10)
        if rs_data['composite'] < 4: mandatory_passed = False

        avg_vol = volume.tail(20).mean()
        vol_ratio = volume.iloc[-1] / avg_vol if avg_vol > 0 else 1
        scores['volume'] = min(weights['volume'], weights['volume'] * (vol_ratio / 1.2))
        if vol_ratio < 1.2: mandatory_passed = False

        above_emas = curr_price > ema20.iloc[-1] > ema50.iloc[-1]
        trend_key = 'trend_alignment' if 'trend_alignment' in weights else 'trend_strength'
        scores[trend_key] = weights[trend_key] if above_emas else 0
        if not above_emas: mandatory_passed = False

        rsi_key = 'rsi_zone' if 'rsi_zone' in weights else 'rsi_power'
        rsi_range = config.mandatory_filters.get('rsi_range', (55, 80))
        rsi_ok = rsi_range[0] <= curr_rsi <= rsi_range[1]
        scores[rsi_key] = weights[rsi_key] if rsi_ok else 0

        regime_info = MarketRegimeEngine.detect_regime()
        scores['regime'] = weights['regime'] * (regime_info['score'] / 100)

        if 'candle' in weights:
            cb = abs(close.iloc[-1] - df['Open'].iloc[-1])
            cr = high.iloc[-1] - low.iloc[-1]
            scores['candle'] = weights['candle'] * (cb / cr if cr > 0 else 0)

        scores['sector_rs'] = weights['sector_rs'] if sector_ok else 0
        if 'breadth_confirm' in weights:
            scores['breadth_confirm'] = weights['breadth_confirm'] if breadth_ok else 0
        scores['liquidity_score'] = weights['liquidity_score'] * (liq_score / 100)

        total = sum(scores.values())
        return {
            'total_score': round(total, 2),
            'max_score': sum(weights.values()),
            'scores': {k: round(v, 2) for k, v in scores.items()},
            'passed_filters': mandatory_passed,
            'details': {
                'relative_strength': rs_data['composite'],
                'volume_ratio': round(vol_ratio, 2),
                'price_above_emas': above_emas,
                'rsi': round(curr_rsi, 2),
            }
        }


# ═══════════════════════════════════════════════════════════════════════
#  LAYER 4: RISK LAYER
# ═══════════════════════════════════════════════════════════════════════

class RiskCalculator:
    """Enhanced risk with gap-aware position sizing."""

    @staticmethod
    def calculate_risk_params(df: pd.DataFrame, setup_type: SetupType,
                              entry_price: float, gap_risk: Dict = None) -> Dict:
        config = SETUP_CONFIGS.get(setup_type, SETUP_CONFIGS[SetupType.MOMENTUM])
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        atr = ta.atr(high, low, close, 14).iloc[-1]

        if setup_type == SetupType.BREAKOUT:
            sl_mult = np.mean(config.risk_profile['sl_atr_multiplier'])
            sl = entry_price - (atr * sl_mult)
            t1 = entry_price + (atr * 2.0)
            t2 = entry_price + (atr * 3.5)
        elif setup_type == SetupType.PULLBACK:
            recent_low = low.tail(10).min()
            sl_mult = np.mean(config.risk_profile['sl_atr_multiplier'])
            sl = min(recent_low, entry_price - (atr * sl_mult))
            t1 = entry_price + (atr * 1.5)
            t2 = entry_price + (atr * 2.5)
        elif setup_type == SetupType.POWER_PLAY:
            ema10 = ta.ema(close, 10).iloc[-1]
            sl_mult = np.mean(config.risk_profile['sl_atr_multiplier'])
            sl = max(ema10 - atr * 0.5, entry_price - (atr * sl_mult))
            t1 = entry_price + (atr * 2.0)
            t2 = entry_price + (atr * 3.5)
        else:
            ema20 = ta.ema(close, 20).iloc[-1]
            sl_mult = np.mean(config.risk_profile['sl_atr_multiplier'])
            sl = max(ema20 - atr, entry_price - (atr * sl_mult))
            t1 = entry_price + (atr * 2.5)
            t2 = entry_price + (atr * 4.0)

        risk_pct = ((entry_price - sl) / entry_price) * 100
        reward1_pct = ((t1 - entry_price) / entry_price) * 100
        reward2_pct = ((t2 - entry_price) / entry_price) * 100
        rr1 = reward1_pct / risk_pct if risk_pct > 0 else 0
        rr2 = reward2_pct / risk_pct if risk_pct > 0 else 0

        return {
            'entry': round(entry_price, 2),
            'stop_loss': round(sl, 2),
            'target1': round(t1, 2),
            'target2': round(t2, 2),
            'risk_pct': round(risk_pct, 2),
            'reward1_pct': round(reward1_pct, 2),
            'reward2_pct': round(reward2_pct, 2),
            'rr1': round(rr1, 2),
            'rr2': round(rr2, 2),
            'atr': round(atr, 2),
            'max_risk_pct': config.risk_profile['max_risk_pct'],
            'gap_adjusted': gap_risk is not None,
        }


# ═══════════════════════════════════════════════════════════════════════
#  LAYER 5: CAPITAL LAYER
# ═══════════════════════════════════════════════════════════════════════

class PositionSizingEngine:
    """Position size tied to probability tier."""

    @staticmethod
    def calculate_position(total_capital: float, risk_per_trade_pct: float,
                           entry: float, stop_loss: float,
                           tier: ProbabilityTier,
                           gap_risk_factor: float = 1.0,
                           vix_size_factor: float = 1.0) -> Dict:
        risk_per_share = abs(entry - stop_loss)
        if risk_per_share <= 0:
            return {'shares': 0, 'capital_required': 0, 'risk_amount': 0}

        r_multiplier = TIER_ALLOCATION[tier]
        # v1.1: VIX size adjustment applied here
        vix_factor   = vix_size_factor
        effective_r  = r_multiplier * gap_risk_factor * vix_factor

        base_risk = total_capital * (risk_per_trade_pct / 100)
        adjusted_risk = base_risk * effective_r

        shares = int(adjusted_risk / risk_per_share)
        capital_required = shares * entry
        actual_risk = shares * risk_per_share

        return {
            'shares': shares,
            'capital_required': round(capital_required, 2),
            'capital_pct': round((capital_required / total_capital) * 100, 2) if total_capital > 0 else 0,
            'risk_amount': round(actual_risk, 2),
            'risk_pct_of_capital': round((actual_risk / total_capital) * 100, 3) if total_capital > 0 else 0,
            'tier': tier.value,
            'r_multiplier': round(effective_r, 2),
            'gap_adjusted': gap_risk_factor < 1.0,
        }


# ═══════════════════════════════════════════════════════════════════════
#  LAYER 6: EVOLUTION LAYER — Expectancy Tracker
# ═══════════════════════════════════════════════════════════════════════

class TradeExpectancyTracker:
    """Per-setup expectancy tracking with edge stability."""

    def __init__(self, storage_path: str = "expectancy_data.json"):
        self._path = storage_path
        self._data = self._load()

    def _load(self) -> Dict:
        try:
            with open(self._path, 'r') as f:
                return json.load(f)
        except:
            return {}

    def _save(self):
        with open(self._path, 'w') as f:
            json.dump(self._data, f, indent=2)

    def record_trade(self, setup_type: str, r_result: float, won: bool):
        if setup_type not in self._data:
            self._data[setup_type] = {
                'total': 0, 'wins': 0, 'losses': 0,
                'total_r_won': 0, 'total_r_lost': 0,
                'history': [],
            }

        d = self._data[setup_type]
        d['total'] += 1
        if won:
            d['wins'] += 1
            d['total_r_won'] += abs(r_result)
        else:
            d['losses'] += 1
            d['total_r_lost'] += abs(r_result)
        d['history'].append({
            'r': round(r_result, 2), 'won': won,
            'date': datetime.now().strftime('%Y-%m-%d'),
        })
        self._save()

    def get_expectancy(self, setup_type: str) -> Dict:
        d = self._data.get(setup_type, {})
        total = d.get('total', 0)
        if total == 0:
            return {'expectancy': 0, 'win_rate': 0, 'trades': 0, 'is_active': True,
                    'edge_stability': 'UNKNOWN', 'confidence_meter': 'Gray'}

        wins = d.get('wins', 0)
        losses = d.get('losses', 0)
        wr = wins / total
        lr = losses / total
        avg_win = d.get('total_r_won', 0) / max(wins, 1)
        avg_loss = d.get('total_r_lost', 0) / max(losses, 1)
        expectancy = (wr * avg_win) - (lr * avg_loss)

        is_active = True
        if total >= 10 and expectancy < -0.1:
            is_active = False

        history = d.get('history', [])
        rolling_exp = 0.0
        edge_stability = 'UNKNOWN'
        confidence_meter = 'Gray'

        if len(history) >= 5:
            recent = history[-20:]
            r_wins = [h for h in recent if h.get('won')]
            r_losses = [h for h in recent if not h.get('won')]
            r_wr = len(r_wins) / len(recent)
            r_lr = len(r_losses) / len(recent)
            r_avg_win = np.mean([abs(h['r']) for h in r_wins]) if r_wins else 0
            r_avg_loss = np.mean([abs(h['r']) for h in r_losses]) if r_losses else 0
            rolling_exp = (r_wr * r_avg_win) - (r_lr * r_avg_loss)

            if len(history) >= 10:
                old_exp = self._calc_slice_expectancy(history[-10:-5])
                new_exp = self._calc_slice_expectancy(history[-5:])
                trend = new_exp - old_exp
            else:
                trend = 0.0

            edge_stability = (
                'STRONG' if rolling_exp > 0.3 else
                'STABLE' if rolling_exp > 0 else
                'DEGRADING' if rolling_exp > -0.2 else
                'COLLAPSED'
            )
            confidence_meter = (
                'Green' if rolling_exp > 0 and trend >= 0 else
                'Orange' if rolling_exp >= -0.1 else
                'Red'
            )
            size_reduction = 0.25 if edge_stability in ('DEGRADING', 'COLLAPSED') else 0.0
        else:
            rolling_exp = expectancy
            size_reduction = 0.0

        return {
            'expectancy': round(expectancy, 3),
            'win_rate': round(wr * 100, 1),
            'avg_win_r': round(avg_win, 2),
            'avg_loss_r': round(avg_loss, 2),
            'trades': total,
            'is_active': is_active,
            'recommendation': 'ACTIVE' if is_active else '⚠️ DISABLED — negative expectancy',
            'rolling_expectancy_20': round(rolling_exp, 3),
            'edge_stability': edge_stability,
            'confidence_meter': confidence_meter,
            'suggested_size_reduction': size_reduction,
        }

    def _calc_slice_expectancy(self, history_slice: list) -> float:
        if not history_slice:
            return 0.0
        wins = [h for h in history_slice if h.get('won')]
        losses = [h for h in history_slice if not h.get('won')]
        n = len(history_slice)
        wr = len(wins) / n
        lr = len(losses) / n
        avg_win = np.mean([abs(h['r']) for h in wins]) if wins else 0
        avg_loss = np.mean([abs(h['r']) for h in losses]) if losses else 0
        return (wr * avg_win) - (lr * avg_loss)

    def get_all_stats(self) -> Dict:
        return {st: self.get_expectancy(st) for st in self._data}


# ═══════════════════════════════════════════════════════════════════════
#  PLAYBOOK GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class PlaybookGenerator:

    @staticmethod
    def generate_playbook(ticker: str, setup_type: SetupType, score_data: Dict,
                          risk_data: Dict, regime_info: Dict,
                          classification: Dict = None) -> Dict:
        config = SETUP_CONFIGS.get(setup_type, SETUP_CONFIGS[SetupType.MOMENTUM])

        tier = classification.get('probability_tier', ProbabilityTier.B) if classification else ProbabilityTier.B
        status = classification.get('status', TradeStatus.WATCH) if classification else TradeStatus.WATCH

        # V4: Add entry recommendation from gap risk
        gap_risk = classification.get('gap_risk', {}) if classification else {}
        entry_recommendation = gap_risk.get('entry_recommendation', 'Buy at Market')

        playbook = {
            'ticker': ticker,
            'setup_type': setup_type.value,
            'probability_tier': tier.value,
            'status': status.value,
            'confidence': PlaybookGenerator._calc_confidence(score_data, regime_info),
            'allocation_r': TIER_ALLOCATION.get(tier, 0.5),
            'why_selected': PlaybookGenerator._explain(setup_type, score_data, regime_info),
            'entry_plan': PlaybookGenerator._entry_plan(setup_type, risk_data, entry_recommendation),
            'what_to_watch': PlaybookGenerator._watch_items(setup_type),
            'invalidation_rules': PlaybookGenerator._invalidation(setup_type, risk_data),
            'risk_comment': PlaybookGenerator._risk_comment(setup_type, risk_data),
            'time_decay': config.time_decay_days,
            'risk_data': risk_data,
            'regime_aligned': regime_info['type'] != RegimeType.RISK_OFF,
            'rr_ratio': risk_data.get('rr1', 0),
            'event_risk': classification.get('gap_risk', {}).get('risk_level', 'LOW') if classification else 'LOW',
            'sector_rs_aligned': classification.get('sector_rs_positive', False) if classification else False,
            'liquidity_ok': classification.get('liquidity', {}).get('is_liquid', True) if classification else True,
            'breakout_confirmed': classification.get('breakout_confirmation', {}).get('confirmed', True) if classification else True,
            'trap_probability': classification.get('trap_analysis', {}).get('trap_probability', 0) if classification else 0,
        }
        return playbook

    @staticmethod
    def _calc_confidence(score_data, regime_info):
        if not score_data.get('max_score'):
            return 50
        ratio = score_data['total_score'] / score_data['max_score']
        boost = regime_info.get('score', 50) / 100
        return min(100, max(0, int((ratio * 0.7 + boost * 0.3) * 100)))

    @staticmethod
    def _explain(setup_type, score_data, regime_info):
        base = {
            SetupType.BREAKOUT: f"Volatility contraction + volume expansion. Score: {score_data['total_score']}/{score_data['max_score']}",
            SetupType.PULLBACK: f"Healthy pullback in strong trend. Score: {score_data['total_score']}/{score_data['max_score']}",
            SetupType.MOMENTUM: f"Strong relative strength + rising momentum. Score: {score_data['total_score']}/{score_data['max_score']}",
            SetupType.POWER_PLAY: f"🔥 Super-momentum Power Play. RSI sustaining 72+. Score: {score_data['total_score']}/{score_data['max_score']}",
        }
        return base.get(setup_type, "Setup identified") + f"\n{regime_info.get('message', '')}"

    @staticmethod
    def _entry_plan(setup_type, risk_data, entry_recommendation):
        plans = {
            SetupType.BREAKOUT: {
                'primary': f"Buy above ₹{risk_data['entry']} with vol > 1.5x AFTER 2-day confirmation",
                'alternate': "If intraday pullback to breakout level with volume support",
                'avoid': "Gap up > 2.5%, late-day breakout, wick rejection, no 2-day confirmation",
                'entry_mode': entry_recommendation,
            },
            SetupType.PULLBACK: {
                'primary': f"Buy on bounce above ₹{risk_data['entry']} with volume expansion",
                'alternate': "Enter on reversal candle at EMA20 support",
                'avoid': "Continued breakdown below EMA20, volume spike on decline",
                'entry_mode': entry_recommendation,
            },
            SetupType.MOMENTUM: {
                'primary': f"Buy on continuation above ₹{risk_data['entry']} with strong candle",
                'alternate': "Scale in on shallow pullback to EMA10/20",
                'avoid': "Exhaustion gap, distribution candle",
                'entry_mode': entry_recommendation,
            },
            SetupType.POWER_PLAY: {
                'primary': f"Buy on strength above ₹{risk_data['entry']} if RSI stays 72+",
                'alternate': "Enter on minor dip if RSI holds 68+",
                'avoid': "RSI divergence, distribution day",
                'entry_mode': entry_recommendation,
            },
        }
        return plans.get(setup_type, {
            'primary': f"Buy near ₹{risk_data['entry']}",
            'alternate': "Monitor",
            'avoid': "Adverse conditions",
            'entry_mode': entry_recommendation,
        })

    @staticmethod
    def _watch_items(setup_type):
        items = {
            SetupType.BREAKOUT: [
                "Volume sustained above average",
                "Day 2 holds above resistance",
                "No wick rejection (< 40% upper wick)",
                "Sector continues to lead",
                "Market breadth intact",
            ],
            SetupType.PULLBACK: [
                "Bounce from EMA20/50 with volume",
                "RSI turning up from 35-45 zone",
                "Weekly trend intact",
                "Sector still leading",
            ],
            SetupType.MOMENTUM: [
                "Continuation of strong price action",
                "Multi-TF RS still positive (5d/20d/60d)",
                "Volume staying elevated",
                "No bearish reversal patterns",
                "Sector leadership intact",
            ],
            SetupType.POWER_PLAY: [
                "RSI sustaining 72-85 zone",
                "Volume not drying up",
                "5-day RS accelerating",
                "No distribution days",
                "Trail EMA10 aggressively",
            ],
        }
        return items.get(setup_type, ["Monitor"])

    @staticmethod
    def _invalidation(setup_type, risk_data):
        rules = {
            SetupType.BREAKOUT: [
                f"Day 2 close back inside range (below ₹{risk_data.get('entry', 0)})",
                "Wick rejection > 40% of candle range",
                "Volume dries up on breakout attempt",
                "Market regime flips to Risk-Off",
                f"Stop loss at ₹{risk_data['stop_loss']}",
            ],
            SetupType.PULLBACK: [
                "Breaks below EMA50",
                "Volume spikes on decline",
                f"Stop loss at ₹{risk_data['stop_loss']}",
            ],
            SetupType.MOMENTUM: [
                "Closes below EMA20",
                "RSI drops below 50",
                "RS turns negative",
                f"Stop loss at ₹{risk_data['stop_loss']}",
            ],
            SetupType.POWER_PLAY: [
                "RSI drops below 65",
                "Closes below EMA10",
                "Distribution candle (high volume bearish)",
                f"Stop loss at ₹{risk_data['stop_loss']}",
            ],
        }
        return rules.get(setup_type, [f"SL: ₹{risk_data['stop_loss']}"])

    @staticmethod
    def _risk_comment(setup_type, risk_data):
        r = risk_data['risk_pct']
        if r > 5:
            return f"⚠️ High risk ({r}%). Reduce position size."
        elif r > 3:
            return f"Moderate risk ({r}%). Standard sizing."
        return f"Low risk ({r}%). Favorable R:R."


# ═══════════════════════════════════════════════════════════════════════
#  ACTIVE TRADE EVALUATOR
# ═══════════════════════════════════════════════════════════════════════

class ActiveTradeEvaluator:

    @staticmethod
    def evaluate_trade(ticker: str, entry_price: float, stop_loss: float,
                       target: float, setup_type: SetupType,
                       entry_date: datetime) -> Dict:
        try:
            df = _yf_download(ticker, period="3mo")
            if df.empty:
                return {'error': 'Unable to fetch data'}

            close = df['Close'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            volume = df['Volume'].squeeze()
            curr_price = close.iloc[-1]
            days_held = (datetime.now() - entry_date).days
            pnl_pct = ((curr_price - entry_price) / entry_price) * 100

            health_factors = {
                'trend': ActiveTradeEvaluator._check_trend(df, setup_type),
                'volume': ActiveTradeEvaluator._check_volume(volume),
                'regime': ActiveTradeEvaluator._check_regime(),
                'candle': ActiveTradeEvaluator._check_candle(df),
                'structure': ActiveTradeEvaluator._check_structure(df, setup_type),
            }
            health_score = sum(v for v in health_factors.values()) / len(health_factors)

            if health_score >= 70:
                status, color = "Strong", "#00FF9D"
            elif health_score >= 50:
                status, color = "Warning", "#FFBF00"
            else:
                status, color = "Weak", "#FF0055"

            atr = ta.atr(high, low, close, 14).iloc[-1]
            new_sl = ActiveTradeEvaluator._calc_trailing_sl(
                entry_price, curr_price, stop_loss, pnl_pct, atr, setup_type
            )

            return {
                'status': status,
                'status_color': color,
                'health_score': round(health_score, 1),
                'health_factors': health_factors,
                'pnl_pct': round(pnl_pct, 2),
                'days_held': days_held,
                'current_price': round(curr_price, 2),
                'suggested_sl': round(new_sl, 2),
                'action': ActiveTradeEvaluator._gen_action(health_score, pnl_pct, days_held, setup_type, curr_price, stop_loss, target),
                'phase': ActiveTradeEvaluator._phase(days_held, pnl_pct),
            }
        except Exception as e:
            return {'error': str(e)}

    @staticmethod
    def _check_trend(df, setup_type):
        close = df['Close'].squeeze()
        ema20 = ta.ema(close, 20)
        ema50 = ta.ema(close, 50)
        cp = close.iloc[-1]
        if setup_type in (SetupType.MOMENTUM, SetupType.POWER_PLAY):
            return 100 if cp > ema20.iloc[-1] > ema50.iloc[-1] else (70 if cp > ema20.iloc[-1] else 30)
        return 100 if cp > ema20.iloc[-1] else 40

    @staticmethod
    def _check_volume(volume):
        avg = volume.tail(20).mean()
        ratio = volume.iloc[-1] / avg if avg > 0 else 1
        if ratio > 1.5: return 100
        if ratio > 1.0: return 80
        if ratio > 0.7: return 60
        return 30

    @staticmethod
    def _check_regime():
        return MarketRegimeEngine.detect_regime()['score']

    @staticmethod
    def _check_candle(df):
        close = df['Close'].squeeze()
        op = df['Open'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        body = close.iloc[-1] - op.iloc[-1]
        rng = high.iloc[-1] - low.iloc[-1]
        if body > 0:
            return min(100, (body / rng * 120)) if rng > 0 else 50
        return 30

    @staticmethod
    def _check_structure(df, setup_type):
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        hh = high.iloc[-1] > high.iloc[-5]
        hl = low.iloc[-1] > low.iloc[-5]
        if hh and hl: return 100
        if hh or hl: return 70
        return 40

    @staticmethod
    def _gen_action(health, pnl, days, setup_type, price, sl, target):
        if price <= sl:
            return "🚨 EXIT — Stop loss hit"
        if health < 40:
            return "🔴 EXIT — Structure deteriorating"
        config = SETUP_CONFIGS.get(setup_type)
        max_d = config.time_decay_days if config else 5
        if days > max_d and pnl < 1:
            return f"⏰ EXIT — {days}d with no progress"
        if pnl > 10: return "✅ TRAIL — Protect gains"
        if pnl > 5: return "🟢 MOVE SL TO BE"
        if pnl > 3 and health > 70: return "✓ HOLD — Developing well"
        if pnl < -2: return "⚠️ CAUTION — Near stop"
        if health < 50: return "⚠️ TIGHTEN SL"
        return "⚪ HOLD — Monitor"

    @staticmethod
    def _calc_trailing_sl(entry, current, original_sl, pnl, atr, setup_type, df=None):
        config = SETUP_CONFIGS.get(setup_type, SETUP_CONFIGS[SetupType.MOMENTUM])
        rr_trigger = config.risk_profile.get('structure_trail_rr', 2.0)
        risk = entry - original_sl
        rr_reached = (current - entry) / risk if risk > 0 else 0

        # v1.0: Switch to structure-based trailing (swing-low) once 2:1 RR reached
        if rr_reached >= rr_trigger and df is not None:
            try:
                low = df['Low'].squeeze()
                # Use last 5-bar swing low as structure stop
                swing_low = low.tail(5).min()
                return max(original_sl, swing_low - atr * 0.2)
            except Exception:
                pass

        if pnl < 3:
            return original_sl
        if setup_type == SetupType.POWER_PLAY:
            return current - (atr * 0.8)
        if setup_type == SetupType.MOMENTUM:
            return current - (atr * 1.0)
        if setup_type == SetupType.BREAKOUT:
            return current - (atr * 1.5)
        return max(entry, current - (atr * 1.2))

    @staticmethod
    def _phase(days, pnl):
        if days <= 2: return "Entry Phase"
        if pnl > 8: return "Resistance Phase"
        if pnl > 5: return "Expansion Phase"
        if days <= 4 and pnl > 0: return "Confirmation Phase"
        if pnl < 0: return "Risk Phase"
        return "Development Phase"


# ═══════════════════════════════════════════════════════════════════════
#  PORTFOLIO-LEVEL RISK ENGINE — V4 HARD BLOCKS
# ═══════════════════════════════════════════════════════════════════════

class PortfolioRiskEngine:
    """
    Portfolio-level risk controls with HARD BLOCKS (v4 improvement).
    Returns "BLOCKED (Sector Cap)" status instead of allowing trade.
    """

    MAX_POSITIONS = 6
    MAX_PER_SECTOR = 2
    MAX_HIGH_BETA_PCT = 40.0
    DRAWDOWN_TRIGGER_PCT = 8.0
    DRAWDOWN_REDUCTION = 0.75

    @staticmethod
    def validate_new_trade(
        proposed_ticker: str,
        proposed_sector: str,
        proposed_beta: float,
        proposed_capital_pct: float,
        active_positions: List[Dict],
        portfolio_current_drawdown_pct: float = 0.0
    ) -> Dict:
        """
        Check whether a new trade violates portfolio-level rules.
        Returns {allowed: bool, block_reason: str, ...}
        """
        violations = []
        warnings = []
        allowed = True
        size_multiplier = 1.0
        block_reason = None

        # Rule 1: Max position count
        if len(active_positions) >= PortfolioRiskEngine.MAX_POSITIONS:
            violations.append(f"⛔ Max {PortfolioRiskEngine.MAX_POSITIONS} positions reached ({len(active_positions)} open)")
            allowed = False
            block_reason = "BLOCKED (Max Positions)"

        # Rule 2: Max per sector
        sector_count = sum(1 for p in active_positions
                           if p.get('sector', '').lower() == proposed_sector.lower())
        if sector_count >= PortfolioRiskEngine.MAX_PER_SECTOR:
            violations.append(f"⛔ Sector '{proposed_sector}' already has {sector_count} positions (max {PortfolioRiskEngine.MAX_PER_SECTOR})")
            allowed = False
            block_reason = f"BLOCKED (Sector Cap: {proposed_sector})"

        # Rule 3: High-beta exposure cap
        if proposed_beta > 1.3:
            current_high_beta_pct = sum(
                p.get('capital_pct', 0) for p in active_positions if p.get('beta', 1.0) > 1.3
            )
            if current_high_beta_pct + proposed_capital_pct > PortfolioRiskEngine.MAX_HIGH_BETA_PCT:
                violations.append(
                    f"⛔ High-beta exposure would reach {current_high_beta_pct + proposed_capital_pct:.1f}% "
                    f"(max {PortfolioRiskEngine.MAX_HIGH_BETA_PCT}%)"
                )
                allowed = False
                block_reason = "BLOCKED (High-Beta Cap)"

        # Rule 4: Portfolio drawdown trigger
        if portfolio_current_drawdown_pct >= PortfolioRiskEngine.DRAWDOWN_TRIGGER_PCT:
            size_multiplier = PortfolioRiskEngine.DRAWDOWN_REDUCTION
            warnings.append(
                f"⚠️ Portfolio drawdown {portfolio_current_drawdown_pct:.1f}% triggered size reduction to "
                f"{PortfolioRiskEngine.DRAWDOWN_REDUCTION * 100:.0f}%"
            )

        total_deployed_pct = sum(p.get('capital_pct', 0) for p in active_positions)
        high_beta_deployed = sum(p.get('capital_pct', 0) for p in active_positions if p.get('beta', 1.0) > 1.3)
        sectors_used = {}
        for p in active_positions:
            s = p.get('sector', 'Unknown')
            sectors_used[s] = sectors_used.get(s, 0) + 1

        return {
            'allowed': allowed,
            'block_reason': block_reason,
            'violations': violations,
            'warnings': warnings,
            'size_multiplier': size_multiplier,
            'adjusted_capital_pct': round(proposed_capital_pct * size_multiplier, 2),
            'portfolio_stats': {
                'open_positions': len(active_positions),
                'max_positions': PortfolioRiskEngine.MAX_POSITIONS,
                'total_deployed_pct': round(total_deployed_pct, 1),
                'high_beta_exposure_pct': round(high_beta_deployed, 1),
                'sectors_used': sectors_used,
                'portfolio_drawdown_pct': portfolio_current_drawdown_pct,
            }
        }

    @staticmethod
    def check_portfolio_constraint(
        proposed_sector: str,
        active_positions: List[Dict]
    ) -> Dict:
        """
        V4 IMPROVEMENT: Quick check for portfolio constraints.
        Returns constraint status to embed in TradeStatus.
        """
        sector_count = sum(1 for p in active_positions
                           if p.get('sector', '').lower() == proposed_sector.lower())

        if len(active_positions) >= PortfolioRiskEngine.MAX_POSITIONS:
            return {
                'is_blocked': True,
                'status': TradeStatus.BLOCKED,
                'reason': "Max Positions",
                'message': f"⛔ BLOCKED — Max {PortfolioRiskEngine.MAX_POSITIONS} positions reached"
            }

        if sector_count >= PortfolioRiskEngine.MAX_PER_SECTOR:
            return {
                'is_blocked': True,
                'status': TradeStatus.BLOCKED,
                'reason': f"Sector Cap ({proposed_sector})",
                'message': f"⛔ BLOCKED — {proposed_sector} sector already has {sector_count} positions"
            }

        return {
            'is_blocked': False,
            'status': None,
            'reason': None,
            'message': None
        }

    @staticmethod
    def get_operating_mode(regime_type: RegimeType, breadth_slope: float,
                           vol_regime: str) -> Dict:
        """3-Mode Adaptive Operating System."""
        if (regime_type == RegimeType.RISK_ON and
                breadth_slope > 0 and
                vol_regime in ('Low_Vol_Expansion', 'High_Vol_Compression')):
            mode = 'Aggressive'
            max_positions = 6
            size_multiplier = 1.0
            description = "🟢 Early Bull conditions — full deployment permitted"
        elif (regime_type == RegimeType.RISK_OFF or
              breadth_slope < -2.0 or
              vol_regime == 'Panic_Vol_Spike'):
            mode = 'Conservative'
            max_positions = 3
            size_multiplier = 0.5
            description = "🔴 Late Cycle / Risk-Off — half sizing, 3 positions max"
        else:
            mode = 'Balanced'
            max_positions = 4
            size_multiplier = 0.75
            description = "🟡 Balanced — standard sizing, 4 positions max"

        return {
            'mode': mode,
            'max_positions': max_positions,
            'size_multiplier': size_multiplier,
            'description': description,
        }


# ═══════════════════════════════════════════════════════════════════════
#  MASTER ORCHESTRATOR — Swing Bull Trader Engine
# ═══════════════════════════════════════════════════════════════════════

class SwingBullEngine:
    """
    Master orchestrator (renamed from FreddyEngine).
    Runs all 7 layers for full analysis with portfolio hard blocks.
    """

    def __init__(self, total_capital: float = 1000000.0, risk_per_trade: float = 2.0):
        self._auth = AuthManager()
        self._expectancy = TradeExpectancyTracker()
        self.total_capital = total_capital
        self.risk_per_trade = risk_per_trade
        self._sector_cache = {}
        self._sector_cache_time = 0
        self._active_positions: List[Dict] = []

    def login(self, username: str, password: str, ip: str = "127.0.0.1") -> Optional[str]:
        return self._auth.login(username, password, ip)

    def logout(self, token: str):
        self._auth.logout(token)

    def set_active_positions(self, positions: List[Dict]):
        """Update active positions for portfolio constraint checks."""
        self._active_positions = positions

    def get_market_regime_public(self) -> Dict:
        """Public market regime — no auth required. Safe: read-only data."""
        try:
            breadth = MarketBreadthEngine.calculate_breadth()
            volatility = VolatilityRegimeEngine.detect_state()
            regime = MarketRegimeEngine.detect_regime(breadth_data=breadth, volatility_data=volatility)
            sectors = self._get_sector_data()

            # v1.1: India VIX + F&O expiry + global cues
            india_vix   = IndiaVIXEngine.get_vix()
            fno_expiry  = FNOExpiryGuard.check()
            global_cues = GlobalCuesEngine.get_cues()

            return {
                'market_pulse': {
                    'mpi': regime['mpi'],
                    'degrees': regime['gauge_degrees'],
                    'regime': regime['type'].value,
                    'zone': regime['gauge_zone'],
                    'zone_label': regime['gauge_zone_label'],
                    'label': 'Risk-Off' if regime['mpi'] < 35 else ('Risk-On' if regime['mpi'] > 65 else 'Neutral'),
                },
                'breadth': breadth,
                'volatility': {
                    'state': volatility['state'].value,
                    'vol_regime': volatility.get('vol_regime', 'Unknown'),
                    'favored': volatility.get('favored_strategies', []),
                },
                'sectors': sectors,
                'regime_details': regime['details'],
                # v1.1 additions
                'india_vix': india_vix,
                'fno_expiry': fno_expiry,
                'global_cues': global_cues,
                'timestamp': datetime.now(IST_TZ).strftime('%Y-%m-%d %H:%M IST'),
                'dashboard_meta': DASHBOARD_META,
            }
        except Exception as e:
            logger.error(f"Market regime public error: {e}")
            return {'error': str(e), 'regime': 'Neutral', 'mpi': 50, 'dashboard_meta': DASHBOARD_META}

    def scan_stocks_public(self, tickers: List[str],
                           sector_map: Dict[str, str] = None,
                           market_cap_map: Dict[str, str] = None) -> List[Dict]:
        """Public scanner — no auth required."""
        sector_data = self._get_sector_data()
        results = []
        for i, ticker in enumerate(tickers):
            # Small inter-ticker delay to avoid rate-limit bursts on Render.
            # Cached tickers return instantly; only uncached ones hit the network.
            if i > 0:
                time.sleep(random.uniform(0.3, 0.8))
            try:
                df = _yf_download(ticker, period="3mo")
                if df.empty:
                    continue
                sector = (sector_map or {}).get(ticker, "")
                cap_cat = (market_cap_map or {}).get(ticker, "midcap")
                classification = SetupClassifier.classify_setup(
                    df, ticker, sector, cap_cat, 999, 1.0, sector_data
                )

                # V4: Check portfolio constraints
                portfolio_check = PortfolioRiskEngine.check_portfolio_constraint(
                    sector, self._active_positions
                )

                entry_price = float(df['Close'].squeeze().iloc[-1])
                risk_data = RiskCalculator.calculate_risk_params(
                    df, classification['setup_type'], entry_price,
                    classification.get('gap_risk')
                )

                # Override status if blocked
                final_status = classification['status']
                if portfolio_check['is_blocked']:
                    final_status = TradeStatus.BLOCKED

                # v1.1: All filters + new India-specific checks
                action_cmd    = ActionCommandEngine.get_command(classification, df)
                rs_velocity   = SectorRSVelocity.get_velocity(sector_data, sector)
                ema20_val     = float(ta.ema(df['Close'].squeeze(), 20).iloc[-1])
                entry_zone    = EntryZoneClassifier.classify(entry_price, risk_data['entry'], ema20_val)
                reasoning     = NLPReasoningEngine.generate(ticker, classification, rs_velocity, action_cmd)
                weekly_vol    = WeeklyVolumeConfirmation.check(df)
                vcp_check     = VCPFilter.check_15day(df)
                tight_flag    = VCPFilter.check_tight_flag_3day(df)
                # v1.1 new
                delivery      = DeliveryQualityEstimator.estimate(df)
                weekly_trend  = WeeklyTrendConfirmation.check(df)
                fifty2w       = FiftyTwoWeekHighDetector.check(df)
                results_mom   = ResultsMomentumDetector.detect(df)
                operator_chk  = OperatorAlertDetector.check(df, cap_cat)

                # v1.1: Kill operator stocks immediately
                if operator_chk['is_operator']:
                    final_status = TradeStatus.AVOID

                results.append({
                    'ticker': ticker,
                    'setup_type': classification['setup_type'].value,
                    'tier': classification['probability_tier'].value,
                    'status': final_status.value,
                    'confidence': classification['confidence'],
                    'entry': risk_data['entry'],
                    'stop_loss': risk_data['stop_loss'],
                    'target1': risk_data['target1'],
                    'rr1': risk_data['rr1'],
                    'sector_rs': classification.get('sector_rs_positive', False),
                    'liquidity_ok': classification.get('liquidity', {}).get('is_liquid', False),
                    'volume_ratio': classification.get('volume_ratio', 1.0),
                    'trap_probability': classification.get('trap_analysis', {}).get('trap_probability', 0),
                    'intraday_chase_alert': classification.get('gap_risk', {}).get('intraday_chase_alert', False),
                    'morning_spike_valid': classification.get('gap_risk', {}).get('morning_spike_valid', False),
                    'portfolio_blocked': portfolio_check['is_blocked'],
                    'block_reason': portfolio_check.get('reason'),
                    # v1.0 fields
                    'action_command': action_cmd,
                    'reasoning': reasoning,
                    'entry_zone': entry_zone,
                    'rs_velocity': rs_velocity,
                    'weekly_vol_confirmed': weekly_vol.get('confirmed', True),
                    'weekly_vol_ratio': weekly_vol.get('weekly_ratio', 1.0),
                    'vcp_10d': vcp_check.get('vcp_15d', False),
                    'tight_flag_3d': tight_flag.get('tight_flag_3d', False),
                    # v1.1 new fields
                    'delivery_quality': delivery.get('quality', 'MEDIUM'),
                    'delivery_score': delivery.get('delivery_score', 50),
                    'weekly_trend_ok': weekly_trend.get('weekly_ok', True),
                    'weekly_rsi': weekly_trend.get('weekly_rsi', 50),
                    'is_12w_high': fifty2w.get('is_new_12w_high', False),
                    'near_12w_high': fifty2w.get('near_12w_high', False),
                    'is_ath': fifty2w.get('is_ath', False),
                    'results_momentum': results_mom.get('valid_setup', False),
                    'results_gap_pct': results_mom.get('gap_pct', 0),
                    'operator_alert': operator_chk['is_operator'],
                    'operator_risk_score': operator_chk['risk_score'],
                    'ui_hints': {
                        'status_color': DASHBOARD_META['status_colors'].get(final_status.value, '#64748B'),
                        'action_color': action_cmd.get('color', '#64748B'),
                        'zone_color': entry_zone.get('color', '#64748B'),
                        'delivery_color': '#00FF9D' if delivery.get('quality') == 'HIGH' else ('#FFBF00' if delivery.get('quality') == 'MEDIUM' else '#FF0055'),
                    },
                })
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
        return sorted(results, key=lambda x: x['confidence'], reverse=True)

    def _get_sector_data(self) -> Dict:
        if time.time() - self._sector_cache_time > 900:
            self._sector_cache = SectorLeadershipEngine.analyze_sectors()
            self._sector_cache_time = time.time()
        return self._sector_cache

    @require_auth
    def full_analysis(self, ticker: str, sector: str = "",
                      market_cap_cat: str = "midcap",
                      days_to_earnings: int = 999,
                      beta: float = 1.0,
                      auth_token: str = "") -> Dict:
        """Complete 7-layer analysis with portfolio constraints."""
        # Layer 1: Market
        breadth = MarketBreadthEngine.calculate_breadth()
        volatility = VolatilityRegimeEngine.detect_state()
        regime = MarketRegimeEngine.detect_regime(breadth_data=breadth, volatility_data=volatility)

        # Layer 2: Sector
        sector_data = self._get_sector_data()

        # Fetch stock data
        df = _yf_download(ticker, period="3mo")
        if df.empty:
            return {'error': f'No data for {ticker}'}

        # Layer 3: Stock classification
        classification = SetupClassifier.classify_setup(
            df, ticker, sector, market_cap_cat,
            days_to_earnings, beta, sector_data
        )

        setup_type = classification['setup_type']
        if setup_type == SetupType.UNKNOWN:
            setup_type = SetupType.PULLBACK

        # Score
        score_data = SetupScorer.score_setup(
            df, setup_type, "^NSEI",
            classification.get('sector_rs_positive', False),
            breadth.get('is_healthy', True),
            classification.get('liquidity', {}).get('liquidity_score', 50)
        )

        # Layer 4: Risk
        entry_price = df['Close'].squeeze().iloc[-1]
        risk_data = RiskCalculator.calculate_risk_params(
            df, setup_type, entry_price,
            classification.get('gap_risk')
        )

        # Layer 5: Capital sizing
        tier = classification['probability_tier']
        gap_factor = classification.get('gap_risk', {}).get('position_size_factor', 1.0)
        position = PositionSizingEngine.calculate_position(
            self.total_capital, self.risk_per_trade,
            risk_data['entry'], risk_data['stop_loss'],
            tier, gap_factor
        )

        # Layer 6: Expectancy check
        expectancy = self._expectancy.get_expectancy(setup_type.value)

        if not expectancy['is_active']:
            classification['status'] = TradeStatus.AVOID
            classification['probability_tier'] = ProbabilityTier.C

        # Layer 7: Portfolio constraints (V4)
        portfolio_check = PortfolioRiskEngine.check_portfolio_constraint(
            sector, self._active_positions
        )
        if portfolio_check['is_blocked']:
            classification['status'] = TradeStatus.BLOCKED
            classification['probability_tier'] = ProbabilityTier.C

        # Playbook
        playbook = PlaybookGenerator.generate_playbook(
            ticker, setup_type, score_data, risk_data, regime, classification
        )

        return {
            'ticker': ticker,
            'timestamp': datetime.now(IST_TZ).strftime('%Y-%m-%d %H:%M IST'),
            'layers': {
                'market': {
                    'regime': regime['type'].value,
                    'score': regime['score'],
                    'mpi': regime['mpi'],
                    'gauge_degrees': regime['gauge_degrees'],
                    'gauge_zone': regime['gauge_zone'],
                    'breadth': regime['breadth'],
                    'volatility': {
                        'state': volatility['state'].value,
                        'vol_regime': volatility.get('vol_regime', 'Unknown'),
                        'atr_ratio': volatility.get('atr_ratio', 1.0),
                    },
                    'favored_strategies': regime.get('favored_strategies', []),
                },
                'sector': {
                    'stock_sector': sector,
                    'sector_rs': classification.get('sector_rs', {}),
                    'sector_rs_positive': classification.get('sector_rs_positive', False),
                    'all_sectors': sector_data,
                },
                'stock': {
                    'setup_type': setup_type.value,
                    'confidence': classification['confidence'],
                    'probability_tier': classification['probability_tier'].value,
                    'status': classification['status'].value,
                    'scores': classification['scores'],
                    'rsi': classification.get('rsi', {}),
                    'rs': classification.get('rs', {}),
                    'breakout_confirmation': classification.get('breakout_confirmation', {}),
                    'trap_analysis': classification.get('trap_analysis', {}),
                    'trend_context': classification.get('trend_context', 'Mid_Trend'),
                    'hard_kill_conditions': classification.get('hard_kill_conditions', []),
                    'kill_triggered': classification.get('kill_triggered', False),
                },
                'risk': {
                    **risk_data,
                    'gap_risk': classification.get('gap_risk', {}),
                    'liquidity': classification.get('liquidity', {}),
                },
                'capital': position,
                'evolution': expectancy,
                'portfolio': portfolio_check,
            },
            'playbook': playbook,
            'trade_card': {
                'setup_badge': setup_type.value,
                'tier_badge': classification['probability_tier'].value,
                'regime_badge': regime['type'].value,
                'rr_ratio': risk_data.get('rr1', 0),
                'time_decay_days': SETUP_CONFIGS.get(setup_type, SETUP_CONFIGS[SetupType.MOMENTUM]).time_decay_days,
                'event_risk': classification.get('gap_risk', {}).get('risk_level', 'LOW'),
                'status': classification['status'].value,
                'color': DASHBOARD_META['status_colors'].get(classification['status'].value, '#64748B'),
            },
            'dashboard_meta': DASHBOARD_META,
        }

    @require_auth
    def market_overview(self, auth_token: str = "") -> Dict:
        """Dashboard market overview — Market Pulse + sector gauges."""
        breadth = MarketBreadthEngine.calculate_breadth()
        volatility = VolatilityRegimeEngine.detect_state()
        regime = MarketRegimeEngine.detect_regime(breadth_data=breadth, volatility_data=volatility)
        sectors = self._get_sector_data()

        return {
            'market_pulse': {  # Renamed from freddy_gauge
                'mpi': regime['mpi'],
                'degrees': regime['gauge_degrees'],
                'zone': regime['gauge_zone'],
                'zone_label': regime['gauge_zone_label'],
                'regime': regime['type'].value,
                'label': 'Risk-Off' if regime['mpi'] < 35 else ('Risk-On' if regime['mpi'] > 65 else 'Neutral'),
            },
            'breadth': breadth,
            'volatility': {
                'state': volatility['state'].value,
                'vol_regime': volatility.get('vol_regime', 'Unknown'),
                'favored': volatility.get('favored_strategies', []),
            },
            'sectors': sectors,
            'regime_details': regime['details'],
            'timestamp': datetime.now(IST_TZ).strftime('%Y-%m-%d %H:%M IST'),
            'dashboard_meta': DASHBOARD_META,
        }

    @require_auth
    def record_trade_result(self, setup_type: str, r_result: float,
                            won: bool, auth_token: str = ""):
        self._expectancy.record_trade(setup_type, r_result, won)

    @require_auth
    def get_performance_stats(self, auth_token: str = "") -> Dict:
        return self._expectancy.get_all_stats()


# Backward compatibility alias
FreddyEngine = SwingBullEngine


# ═══════════════════════════════════════════════════════════════════════
#  EXPORTS
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    'SwingBullEngine', 'FreddyEngine', 'AuthManager', 'SecurityConfig',
    'SetupType', 'RegimeType', 'VolatilityState', 'ProbabilityTier', 'TradeStatus',
    'MarketRegimeEngine', 'MarketBreadthEngine', 'VolatilityRegimeEngine',
    'SectorLeadershipEngine', 'MultiTimeframeRS',
    'SetupClassifier', 'SetupScorer', 'RSIClassifier',
    'LiquidityFilter', 'GapRiskModel', 'BreakoutConfirmation',
    'WickTrapFilter', 'TrapDetector',
    'RiskCalculator', 'PositionSizingEngine',
    'PlaybookGenerator', 'ActiveTradeEvaluator',
    'TradeExpectancyTracker',
    'PortfolioRiskEngine',
    # v1.0 exports
    'WeeklyVolumeConfirmation', 'VCPFilter', 'BreadthDivergenceDetector',
    'SectorRSVelocity', 'CorrelationReplacementEngine',
    'ActionCommandEngine', 'NLPReasoningEngine', 'EntryZoneClassifier',
    'Layer8Journal',
    # v1.1 new exports
    'IndiaVIXEngine', 'FNOExpiryGuard', 'DeliveryQualityEstimator',
    'WeeklyTrendConfirmation', 'FiftyTwoWeekHighDetector',
    'ResultsMomentumDetector', 'OperatorAlertDetector', 'GlobalCuesEngine',
    'DASHBOARD_META',
]
