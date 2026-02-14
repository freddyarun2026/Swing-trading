"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          FREDDY EVOLUTION ENGINE™ v3.0 — Trading Engine                    ║
║          Architected by Freddy — Personal Use Only                         ║
║          Indian Market Optimized • NSE/BSE Focus                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

LAYERED ARCHITECTURE:
  Layer 1: Market   (Regime + REAL Breadth + Volatility State)
  Layer 2: Sector   (Leadership + Multi-TF Relative Strength)
  Layer 3: Stock    (Setup Detection + Structure + Power Play)
  Layer 4: Risk     (ATR + Beta + Gap Risk + Event Risk)
  Layer 5: Capital  (Probability Tiered Sizing + Exposure Mgmt)
  Layer 6: Evolution(Expectancy Tracking + Self-Protective Logic)
  Layer 7: Portfolio(Risk Engine — NEW in v3)

BUG FIXES (v2 → v3):
  🔧 FIX 1: SetupClassifier.classify alias added (was: classify_setup only)
             api_server.py calling .classify() caused 500 on all stock scans.
  🔧 FIX 2: /api/market-regime 500 fixed — added get_market_regime_public()
             and scan_stocks_public() that bypass @require_auth for read-only data.

STAGE-1 IMPROVEMENTS INCORPORATED (v3):
  ✅ 1. Real breadth (% above 50 EMA, A/D ratio, % 20d highs)
  ✅ 2. Multi-timeframe RS (5d / 20d / 60d)
  ✅ 3. Flexible RSI — no rigid 72 cap, graded classification + Power Play
  ✅ 4. 2-day breakout confirmation to avoid NSE traps
  ✅ 5. Liquidity filter (₹ crore traded value basis)
  ✅ 6. Volatility regime layer (Compression / Expansion)
  ✅ 7. Sector Leadership Engine (RS vs Nifty, sector wave tracking)
  ✅ 8. Position Sizing Engine (A+/A/B/C tiers → 1.0R/0.75R/0.5R/0.25R)
  ✅ 9. Gap Risk Model (historical gaps, earnings proximity, beta)
  ✅ 10. Market Participation Index — True "Freddy Gauge™"
  ✅ 11. Trade Expectancy Tracker (self-protective evolution)
  ✅ 12. Eliminated rigid setup logic — always return best setup, just size smaller
  ✅ 13. Security — session auth, rate limiting, IP binding, personal use only

  NEW IN v3 (Stage-1 Review Improvements):
  🆕 14. Slope-Based Breadth Deterioration Detector (Nifty500/Nifty50 RS slope)
  🆕 15. Sector Leadership Concentration Index (top-3 concentration → FRAGILE/ROBUST)
  🆕 16. Follow-Through Rule for breakouts (Day+2 midpoint test + engulfing guard)
  🆕 17. Volume Z-Score (adaptive vs fixed 1.5x, midcap/largecap aware)
  🆕 18. ATR-Based Gap Risk Multiplier (liquidity vacuum detection)
  🆕 19. Granular Volatility Regime (Low_Vol_Expansion, Panic_Vol_Spike, etc.)
  🆕 20. Hard Kill Conditions Layer (earnings, liquidity, ATR-gap veto)
  🆕 21. Setup Context Tag (Early_Trend / Mid_Trend / Late_Trend)
  🆕 22. Edge Stability Score + Strategy Confidence Meter (Green/Orange/Red)
  🆕 23. Portfolio-Level Risk Engine (max positions, sector caps, drawdown trigger)
  🆕 24. 3-Mode Adaptive Operating System (Conservative / Balanced / Aggressive)
  🆕 25. Public API helpers (get_market_regime_public, scan_stocks_public)
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("freddy.engine")


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
        stored_hash = self._hash_password(
            SecurityConfig.OWNER_PASSWORD_HASH
        ) if SecurityConfig.OWNER_PASSWORD_HASH.startswith("change") else SecurityConfig.OWNER_PASSWORD_HASH

        # For initial setup, compare against default
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
    BREAKOUT = "Breakout"
    PULLBACK = "Pullback"
    MOMENTUM = "Momentum"
    POWER_PLAY = "Power Play"    # NEW: super-momentum RSI 72-85 zone
    UNKNOWN = "Unknown"


class RegimeType(Enum):
    RISK_ON = "Risk-On"
    NEUTRAL = "Neutral"
    RISK_OFF = "Risk-Off"


class VolatilityState(Enum):
    COMPRESSION = "Compression"
    EXPANSION = "Expansion"
    TRANSITIONING = "Transitioning"


class ProbabilityTier(Enum):
    A_PLUS = "A+"   # 1.00R
    A = "A"         # 0.75R
    B = "B"         # 0.50R
    C = "C"         # 0.25R


class TradeStatus(Enum):
    READY = "Ready"       # Green
    WATCH = "Watch"       # Orange
    AVOID = "Avoid"       # Red
    EXPIRED = "Expired"   # Grey


# ═══════════════════════════════════════════════════════════════════════
#  SETUP CONFIGS (Enhanced)
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
            'min_volume_ratio': 1.5,
            'min_resistance_distance': 1.5,
            'max_gap_pct': 1.5,
            'sector_rs_positive': True,
            'regime_not_risk_off': True,
            'liquidity_ok': True,           # NEW
            'breakout_confirmed_2d': True,  # NEW: 2-day confirmation
        },
        scoring_weights={
            'volatility_contraction': 15,
            'volume_expansion': 15,
            'location_quality': 15,
            'regime_alignment': 10,
            'candle_anatomy': 10,
            'relative_strength': 10,
            'sector_rs': 10,          # NEW
            'breadth_confirm': 8,     # NEW
            'liquidity_score': 7,     # NEW
        },
        risk_profile={
            'sl_atr_multiplier': (1.8, 2.2),
            'max_risk_pct': 4.0,
            'trailing_trigger_atr': 1.5,
            'move_required_days': 3
        },
        time_decay_days=3
    ),

    SetupType.PULLBACK: SetupConfig(
        name="Pullback",
        mandatory_filters={
            'ema20_above_ema50': True,
            'ema50_rising': True,
            'rsi_range': (35, 55),      # Slightly wider
            'pullback_volume_low': True,
            'bounce_volume_expansion': True,
            'weekly_aligned': True,
            'liquidity_ok': True,        # NEW
        },
        scoring_weights={
            'trend_structure': 20,
            'pullback_quality': 15,
            'volume_pattern': 15,
            'weekly_alignment': 10,
            'rsi_zone': 10,
            'regime': 10,
            'sector_rs': 10,             # NEW
            'breadth_confirm': 5,        # NEW
            'liquidity_score': 5,        # NEW
        },
        risk_profile={
            'sl_atr_multiplier': (1.2, 1.5),
            'max_risk_pct': 3.0,
            'trailing_trigger_atr': 1.0,
            'move_required_days': 5
        },
        time_decay_days=5
    ),

    SetupType.MOMENTUM: SetupConfig(
        name="Momentum",
        mandatory_filters={
            'min_rs_vs_benchmark': 4.0,
            'rsi_range': (55, 80),        # WIDENED — was 55-72
            'min_volume_ratio': 1.2,
            'price_above_emas': True,
            'no_bearish_reversal': True,
            'liquidity_ok': True,          # NEW
        },
        scoring_weights={
            'relative_strength': 25,
            'volume': 15,
            'trend_alignment': 15,
            'rsi_zone': 10,
            'regime': 10,
            'candle': 5,
            'sector_rs': 10,               # NEW
            'breadth_confirm': 5,          # NEW
            'liquidity_score': 5,          # NEW
        },
        risk_profile={
            'sl_atr_multiplier': (1.4, 1.8),
            'max_risk_pct': 3.5,
            'trailing_trigger_atr': 0.8,
            'move_required_days': 2,
            'trail_below_ema': 20
        },
        time_decay_days=2
    ),

    # NEW: Power Play — super momentum that stays RSI 72-85
    SetupType.POWER_PLAY: SetupConfig(
        name="Power Play",
        mandatory_filters={
            'rsi_range': (72, 85),
            'min_rs_vs_benchmark': 6.0,
            'price_above_emas': True,
            'sector_rs_positive': True,
            'regime_risk_on': True,
            'liquidity_ok': True,
        },
        scoring_weights={
            'relative_strength': 30,
            'volume': 15,
            'trend_strength': 15,
            'rsi_power': 15,
            'regime': 10,
            'sector_rs': 10,
            'liquidity_score': 5,
        },
        risk_profile={
            'sl_atr_multiplier': (1.0, 1.4),
            'max_risk_pct': 2.5,
            'trailing_trigger_atr': 0.6,
            'move_required_days': 1,
            'trail_below_ema': 10
        },
        time_decay_days=2
    ),
}


# ═══════════════════════════════════════════════════════════════════════
#  TIER ALLOCATION MAP
# ═══════════════════════════════════════════════════════════════════════

TIER_ALLOCATION = {
    ProbabilityTier.A_PLUS: 1.00,
    ProbabilityTier.A:      0.75,
    ProbabilityTier.B:      0.50,
    ProbabilityTier.C:      0.25,
}


# ═══════════════════════════════════════════════════════════════════════
#  INDIAN SECTOR UNIVERSE
# ═══════════════════════════════════════════════════════════════════════

SECTOR_INDEX_MAP = {
    "NIFTY_BANK":   "^NSEBANK",
    "NIFTY_IT":     "^CNXIT",
    "NIFTY_PHARMA": "^CNXPHARMA",
    "NIFTY_AUTO":   "^CNXAUTO",
    "NIFTY_METAL":  "^CNXMETAL",
    "NIFTY_REALTY":  "^CNXREALTY",
    "NIFTY_FMCG":   "^CNXFMCG",
    "NIFTY_ENERGY": "^CNXENERGY",
    "NIFTY_INFRA":  "^CNXINFRA",
    "NIFTY_PSU_BANK": "^CNXPSUBANK",
}


# ═══════════════════════════════════════════════════════════════════════
#  LAYER 1: MARKET LAYER — Regime + Real Breadth + Volatility
# ═══════════════════════════════════════════════════════════════════════

class MarketBreadthEngine:
    """
    IMPROVEMENT #1: Real breadth calculation.
    Not just index score — actual market participation metrics.
    """

    @staticmethod
    def calculate_breadth(stock_universe: List[str] = None,
                          pct_above_50ema: float = None,
                          adv_dec_ratio: float = None,
                          pct_20d_high: float = None,
                          pct_20d_low: float = None) -> Dict:
        """
        Calculate REAL market breadth.
        Can accept pre-computed values or compute from stock universe.
        """
        # If pre-computed values provided, use them directly
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
            # Fallback: estimate from Nifty 50 components
            breadth = MarketBreadthEngine._estimate_from_index()

        # Composite breadth score
        ad_normalized = min((breadth['advance_decline_ratio'] / 2.0), 1.0) * 100
        composite = (
            0.40 * breadth['pct_above_50ema'] +
            0.30 * ad_normalized +
            0.20 * min(breadth['pct_20d_high'] * 100, 100) +
            0.10 * max(0, 100 - breadth['pct_20d_low'] * 100)
        )
        breadth['composite_score'] = round(composite, 2)
        breadth['is_healthy'] = composite > 55

        # ── STAGE-1 IMPROVEMENT: Slope-Based Breadth Deterioration Detector ──
        # Instead of threshold-only, detect gradual distribution BEFORE breakdown.
        breadth['breadth_slope'] = MarketBreadthEngine._calc_breadth_slope()
        slope = breadth['breadth_slope']
        # Combine static threshold + slope for better early warning
        breadth['is_deteriorating'] = slope < -2.0         # slope falling fast
        breadth['breadth_strength'] = (
            'Expanding' if slope > 1.0 else
            'Stable' if slope > -1.0 else
            'Deteriorating' if slope > -3.0 else
            'Collapsing'
        )

        return breadth

    @staticmethod
    def _compute_from_universe(tickers: List[str]) -> Dict:
        """Compute breadth from actual stock list."""
        above_50ema = 0
        advances = 0
        declines = 0
        new_20d_high = 0
        new_20d_low = 0
        total = 0

        for ticker in tickers:
            try:
                df = yf.download(ticker, period="6mo", progress=False)
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
        """
        STAGE-1 IMPROVEMENT: Compute 5-day slope of Nifty500 RS vs Nifty50.
        Detects breadth deterioration BEFORE price breakdown.
        Returns slope as %/day — negative means deteriorating.
        """
        try:
            n50 = yf.download("^NSEI", period="3mo", progress=False)
            n500 = yf.download("^CRSLDX", period="3mo", progress=False)  # Nifty500
            if n50.empty or n500.empty or len(n50) < 10:
                return 0.0
            c50 = n50['Close'].squeeze()
            c500 = n500['Close'].squeeze()
            # Align on common dates
            common = c50.index.intersection(c500.index)
            if len(common) < 10:
                return 0.0
            c50 = c50.loc[common]
            c500 = c500.loc[common]
            # RS ratio: 500/50
            rs_ratio = (c500 / c50).tail(10)
            # Linear slope of last 5 days (percentage per day)
            x = np.arange(len(rs_ratio))
            slope = np.polyfit(x, rs_ratio.values, 1)[0]
            # Normalize to percentage-of-mean
            mean_val = rs_ratio.mean()
            slope_pct = (slope / mean_val) * 100 if mean_val != 0 else 0.0
            return round(slope_pct, 3)
        except Exception:
            return 0.0

    @staticmethod
    def _estimate_from_index() -> Dict:
        """Fallback estimation from Nifty index behavior."""
        try:
            df = yf.download("^NSEI", period="3mo", progress=False)
            if df.empty:
                return {'pct_above_50ema': 50, 'advance_decline_ratio': 1.0,
                        'pct_20d_high': 0.05, 'pct_20d_low': 0.05}
            close = df['Close'].squeeze()
            ema50 = ta.ema(close, 50)

            # Rough estimation based on index position vs EMA
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
    """
    IMPROVEMENT #6 (Advanced): Volatility state detection.
    Compression → favor breakout/pullback
    Expansion → favor momentum
    """

    @staticmethod
    def detect_state(ticker: str = "^NSEI") -> Dict:
        try:
            df = yf.download(ticker, period="6mo", progress=False)
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

            # Strategy recommendation
            if state == VolatilityState.EXPANSION:
                favored = ["Momentum", "Power Play"]
            elif state == VolatilityState.COMPRESSION:
                favored = ["Breakout", "Pullback"]
            else:
                favored = ["Pullback", "Momentum"]

            # ── STAGE-1 IMPROVEMENT: Granular Volatility Regime Classification ──
            # Compression→Expansion transition = highest breakout expectancy window
            atr_hist = atr.tail(60)
            atr_percentile = float(np.percentile(atr_hist.dropna(), [20, 50, 80]))
            curr_atr_val = atr.iloc[-1]

            if ratio > 1.50:
                vol_regime = "Panic_Vol_Spike"           # Avoid new positions
            elif ratio > 1.20 and curr_atr_val < float(np.percentile(atr_hist.dropna(), 50)):
                vol_regime = "Low_Vol_Expansion"         # BEST for breakout entry
            elif ratio > 1.20:
                vol_regime = "High_Vol_Expansion"        # Momentum only, tight size
            elif ratio < 0.80:
                vol_regime = "High_Vol_Compression"      # Prime breakout setup area
            else:
                vol_regime = "Neutral_Transitioning"

            ideal_for_breakout = vol_regime in ("Low_Vol_Expansion", "High_Vol_Compression")

            # Volatility score: 0=very volatile, 100=very calm
            vol_score = max(0, min(100, 100 - (ratio - 0.5) * 100))

            return {
                'state': state,
                'atr_ratio': round(ratio, 3),
                'atr_5': round(atr_5, 2) if not pd.isna(atr_5) else 0,
                'atr_20': round(atr_20, 2) if not pd.isna(atr_20) else 0,
                'favored_strategies': favored,
                'vol_score': round(vol_score, 2),
                # Stage-1 additions
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
    """Enhanced regime detection with real breadth + volatility integration."""

    @staticmethod
    def detect_regime(ticker: str = "^NSEI", breadth_data: Dict = None,
                      volatility_data: Dict = None) -> Dict:
        try:
            df = yf.download(ticker, period="1y", progress=False)
            if df.empty:
                return MarketRegimeEngine._default_regime()

            close = df['Close'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            volume = df['Volume'].squeeze()

            ema20 = ta.ema(close, 20)
            ema50 = ta.ema(close, 50)
            ema200 = ta.ema(close, 200)
            rsi = ta.rsi(close, 14)
            macd = ta.macd(close)

            curr_price = close.iloc[-1]
            score = 0
            details = {}

            # 1. TREND STRENGTH (0-30)
            trend_score = 0
            if curr_price > ema20.iloc[-1]: trend_score += 10
            if curr_price > ema50.iloc[-1]: trend_score += 10
            if curr_price > ema200.iloc[-1]: trend_score += 10
            score += trend_score
            details['trend_score'] = trend_score
            details['price_vs_ema200'] = round(((curr_price - ema200.iloc[-1]) / ema200.iloc[-1]) * 100, 2)

            # 2. MOMENTUM (0-25)
            momentum_5d = ((curr_price - close.iloc[-5]) / close.iloc[-5]) * 100
            momentum_20d = ((curr_price - close.iloc[-20]) / close.iloc[-20]) * 100
            momentum_score = 0
            if momentum_5d > 2: momentum_score += 10
            elif momentum_5d > 0: momentum_score += 5
            if momentum_20d > 5: momentum_score += 15
            elif momentum_20d > 0: momentum_score += 8
            score += momentum_score
            details['momentum_5d'] = round(momentum_5d, 2)
            details['momentum_20d'] = round(momentum_20d, 2)

            # 3. RSI & MACD (0-20)
            curr_rsi = rsi.iloc[-1]
            indicator_score = 0
            if 50 < curr_rsi < 70: indicator_score += 10
            elif 45 < curr_rsi <= 50: indicator_score += 7
            if macd['MACD_12_26_9'].iloc[-1] > macd['MACDs_12_26_9'].iloc[-1]:
                indicator_score += 10
            score += indicator_score
            details['rsi'] = round(curr_rsi, 2)

            # 4. VOLATILITY (0-15)
            recent_returns = close.pct_change().tail(20)
            volatility = recent_returns.std() * np.sqrt(252)
            vol_score_pts = 15 if volatility < 0.15 else (10 if volatility < 0.25 else 5)
            score += vol_score_pts
            details['volatility'] = round(volatility * 100, 2)

            # 5. PRICE ACTION (0-10)
            recent_highs = high.tail(10)
            higher_highs = recent_highs.iloc[-1] > recent_highs.iloc[-5]
            score += 10 if higher_highs else 5

            # Determine regime
            if score >= 70:
                regime_type = RegimeType.RISK_ON
                color = '#00e676'
            elif score >= 40:
                regime_type = RegimeType.NEUTRAL
                color = '#ff9100'
            else:
                regime_type = RegimeType.RISK_OFF
                color = '#ff1744'

            # REAL BREADTH (not just score)
            if breadth_data is None:
                breadth_data = MarketBreadthEngine.calculate_breadth()
            breadth_composite = breadth_data.get('composite_score', score)

            # VOLATILITY STATE
            if volatility_data is None:
                volatility_data = VolatilityRegimeEngine.detect_state(ticker)

            # ═══ MARKET PARTICIPATION INDEX (MPI) — Freddy Gauge™ ═══
            regime_normalized = score  # 0-100
            sector_avg = 50  # Will be overridden by sector layer
            vol_score_mpi = volatility_data.get('vol_score', 50)

            mpi = (
                0.30 * breadth_composite +
                0.30 * regime_normalized +
                0.20 * sector_avg +
                0.20 * vol_score_mpi
            )
            gauge_degrees = (mpi / 100.0) * 180.0  # 0°=Risk-Off, 90°=Neutral, 180°=Risk-On

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
            'color': '#ff9100', 'breadth': {}, 'breadth_composite': 50,
            'volatility': {}, 'mpi': 50, 'gauge_degrees': 90,
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
    """
    IMPROVEMENT #2 (multi-TF RS) + #7 (Sector Leadership).
    Tracks sector waves: PSU → Pharma → IT → Auto → Capital Goods
    """

    @staticmethod
    def analyze_sectors(benchmark: str = "^NSEI") -> Dict:
        """Rank all sectors by multi-timeframe RS vs Nifty."""
        results = {}
        try:
            bench_df = yf.download(benchmark, period="6mo", progress=False)
            if bench_df.empty:
                return {}
            bench_close = bench_df['Close'].squeeze()
        except:
            return {}

        for sector_name, sector_ticker in SECTOR_INDEX_MAP.items():
            try:
                df = yf.download(sector_ticker, period="6mo", progress=False)
                if df.empty or len(df) < 60:
                    continue
                close = df['Close'].squeeze()

                # Multi-timeframe RS
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

        # Rank sectors
        sorted_sectors = sorted(results.items(), key=lambda x: x[1]['rs_composite'], reverse=True)
        for rank, (name, data) in enumerate(sorted_sectors, 1):
            results[name]['rank'] = rank

        # ── STAGE-1 IMPROVEMENT: Sector Leadership Concentration Index ──
        # If top-3 sectors drive >60% of total gains → regime is FRAGILE
        # If 6+ sectors are rising → regime is ROBUST (broad participation)
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
        """Check if stock's sector has positive RS."""
        if sector_data is None:
            sector_data = SectorLeadershipEngine.analyze_sectors()

        # Map common sector names to index names
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
    """IMPROVEMENT #2: 5d/20d/60d relative strength."""

    @staticmethod
    def calculate(stock_close: pd.Series, benchmark_ticker: str = "^NSEI") -> Dict:
        try:
            bench_df = yf.download(benchmark_ticker, period="6mo", progress=False)
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
    """
    IMPROVEMENT #5: Liquidity filter on ₹ crore traded value basis.
    Prevents entering illiquid midcap/smallcap traps.
    """

    THRESHOLDS = {
        'largecap': 50.0,   # ₹50 Cr minimum
        'midcap': 10.0,     # ₹10 Cr minimum
        'smallcap': 3.0,    # ₹3 Cr minimum
    }

    @staticmethod
    def check_liquidity(df: pd.DataFrame, market_cap_cat: str = "midcap") -> Dict:
        try:
            close = df['Close'].squeeze()
            volume = df['Volume'].squeeze()

            # Average traded value in ₹ (approximate, convert to crore)
            traded_value = (close * volume).tail(20).mean()
            traded_value_cr = traded_value / 1e7  # 1 crore = 10 million

            threshold = LiquidityFilter.THRESHOLDS.get(market_cap_cat, 10.0)
            is_liquid = traded_value_cr >= threshold

            # Liquidity score 0-100
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
    IMPROVEMENT #9: Gap risk model for Indian midcaps.
    Scores gap risk based on historical gaps, earnings proximity, beta.
    """

    @staticmethod
    def calculate(df: pd.DataFrame, days_to_earnings: int = 999, beta: float = 1.0) -> Dict:
        try:
            close = df['Close'].squeeze()
            open_price = df['Open'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()

            # Historical gap analysis (% of days with gap > 2%)
            gaps = ((open_price - close.shift(1)) / close.shift(1) * 100).dropna().abs()
            pct_large_gaps = (gaps > 2).sum() / len(gaps) * 100 if len(gaps) > 0 else 0

            # Gap risk components
            gap_component = min(pct_large_gaps * 2, 40)
            earnings_component = max(0, 30 - days_to_earnings) if days_to_earnings < 30 else 0
            beta_component = max(0, (beta - 1.0) * 30)

            gap_risk_score = min(100, gap_component + earnings_component + beta_component)

            # Position size reduction factor (1.0 = no reduction, 0.5 = halve)
            if gap_risk_score > 70:
                size_factor = 0.50
            elif gap_risk_score > 50:
                size_factor = 0.70
            elif gap_risk_score > 30:
                size_factor = 0.85
            else:
                size_factor = 1.00

            # ── STAGE-1 IMPROVEMENT: ATR-Based Gap Risk Multiplier ────────────
            # If average overnight gap > 1.2× ATR → stock has liquidity vacuum risk
            # This catches Indian midcap gaps that pure historical % misses
            atr = ta.atr(high, low, close, 14)
            avg_atr_20 = atr.tail(20).mean()
            avg_gap_abs = gaps.tail(20).mean() / 100 * close.tail(20).mean()  # in price terms
            gap_to_atr_ratio = avg_gap_abs / avg_atr_20 if avg_atr_20 > 0 else 1.0
            atr_gap_penalty = gap_to_atr_ratio > 1.2
            if atr_gap_penalty:
                size_factor = round(size_factor * 0.80, 2)  # Additional 20% reduction

            return {
                'gap_risk_score': round(gap_risk_score, 2),
                'pct_large_gaps': round(pct_large_gaps, 2),
                'days_to_earnings': days_to_earnings,
                'beta': beta,
                'position_size_factor': size_factor,
                'risk_level': 'HIGH' if gap_risk_score > 60 else ('MEDIUM' if gap_risk_score > 30 else 'LOW'),
                # Stage-1 additions
                'gap_to_atr_ratio': round(gap_to_atr_ratio, 2),
                'atr_gap_multiplier_triggered': atr_gap_penalty,
                'liquidity_vacuum_risk': atr_gap_penalty,
            }
        except:
            return {'gap_risk_score': 50, 'position_size_factor': 0.7, 'risk_level': 'MEDIUM'}


class BreakoutConfirmation:
    """
    IMPROVEMENT #4: 2-day breakout confirmation to avoid NSE traps.
    Day1: Close > resistance with 1.5x volume
    Day2: Close still above resistance (not back inside range)

    STAGE-1 ADDITIONS:
    - Follow-Through Rule: Day+2 must not close below breakout midpoint
    - Volume Z-Score: adaptive threshold instead of fixed 1.5x
    - Bearish engulfing guard within 3 sessions
    """

    @staticmethod
    def check(df: pd.DataFrame) -> Dict:
        try:
            close = df['Close'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            open_ = df['Open'].squeeze()
            volume = df['Volume'].squeeze()

            # Resistance = 20-day high (excluding last 2 days)
            resistance = high.iloc[:-2].tail(20).max()
            avg_volume = volume.iloc[:-2].tail(20).mean()
            vol_std = volume.iloc[:-2].tail(20).std()

            # ── STAGE-1: Volume Z-Score (adaptive vs fixed 1.5x) ──────────────
            # More meaningful across midcap/largecap differences
            vol_zscore_d1 = ((volume.iloc[-2] - avg_volume) / vol_std) if vol_std > 0 else 0
            vol_zscore_d2 = ((volume.iloc[-1] - avg_volume) / vol_std) if vol_std > 0 else 0
            # Breakout quality threshold: z-score > 1.0 ≈ top 16% of days (≈1.5x avg)
            vol_breakout_quality = (
                'Strong' if vol_zscore_d1 > 2.0 else
                'Good' if vol_zscore_d1 > 1.0 else
                'Weak'
            )

            # Original volume ratio (kept for backward compat)
            d1_volume_ratio = volume.iloc[-2] / avg_volume if avg_volume > 0 else 1

            # Day 1 (second-to-last bar)
            d1_close = close.iloc[-2]
            d1_broke = d1_close > resistance and vol_zscore_d1 > 0.5  # Z>0.5 ≈ above avg

            # Day 2 (latest bar)
            d2_close = close.iloc[-1]
            d2_held = d2_close > resistance

            confirmed = d1_broke and d2_held

            # ── STAGE-1: Follow-Through Rule ─────────────────────────────────
            # Day+2 must NOT close below the midpoint of the breakout candle
            breakout_midpoint = (high.iloc[-2] + low.iloc[-2]) / 2
            follow_through_ok = d2_close > breakout_midpoint

            # Volume contraction on pullback (healthy sign)
            vol_contracting_d2 = vol_zscore_d2 < vol_zscore_d1

            # ── STAGE-1: Bearish engulfing guard (3-session look-back) ────────
            bearish_engulf = False
            if len(df) >= 3:
                for i in range(-3, 0):
                    body_prev = close.iloc[i-1] - open_.iloc[i-1]
                    body_curr = open_.iloc[i] - close.iloc[i]
                    if body_prev > 0 and body_curr > 0 and body_curr > body_prev:
                        bearish_engulf = True
                        break

            # Enhanced confirmation: original 2-day + follow-through + no engulf
            confirmed_strict = confirmed and follow_through_ok and not bearish_engulf

            return {
                'resistance_level': round(resistance, 2),
                'day1_close': round(d1_close, 2),
                'day1_volume_ratio': round(d1_volume_ratio, 2),
                'day1_volume_zscore': round(vol_zscore_d1, 2),
                'day1_broke_out': d1_broke,
                'day2_close': round(d2_close, 2),
                'day2_held': d2_held,
                'confirmed': confirmed,                          # Original 2-day check
                'confirmed_strict': confirmed_strict,           # + follow-through + no engulf
                'follow_through_ok': follow_through_ok,
                'vol_contraction_pullback': vol_contracting_d2,
                'volume_quality': vol_breakout_quality,
                'bearish_engulf_detected': bearish_engulf,
                'status': (
                    '✅ Strict Confirmed' if confirmed_strict else
                    '⚠️ Confirmed (weak follow-through)' if confirmed else
                    ('⏳ Pending D2' if d1_broke else '❌ Not confirmed')
                ),
            }
        except Exception:
            return {'confirmed': False, 'confirmed_strict': False,
                    'status': '❌ Error checking confirmation'}


class RSIClassifier:
    """
    IMPROVEMENT #3: Flexible RSI — no rigid 72 cap.
    Graded classification including Power Play zone.
    """

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

        # Score: how favorable is this RSI for entry
        if 55 <= rsi_value <= 72:
            rsi_score = 100
        elif 72 < rsi_value <= 80:
            rsi_score = 85   # Still good — Power Play
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
    Enhanced classifier — IMPROVEMENT #12: Never return UNKNOWN.
    Always return best setup. Just downgrade probability tier.

    NOTE: Both classify() and classify_setup() are valid — kept for API compatibility.
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

            # RSI classification (flexible)
            rsi_info = RSIClassifier.classify(curr_rsi)

            # Multi-TF RS
            rs_data = MultiTimeframeRS.calculate(close)

            # Liquidity check
            liq_data = LiquidityFilter.check_liquidity(df, market_cap_cat)

            # Gap risk
            gap_data = GapRiskModel.calculate(df, days_to_earnings, beta)

            # Sector RS
            sector_rs = SectorLeadershipEngine.get_sector_rs_for_stock(sector, sector_data)
            sector_rs_positive = sector_rs.get('is_leading', False)

            # 2-day breakout confirmation
            breakout_confirm = BreakoutConfirmation.check(df)

            # ═══ SCORING — 4 setups ═══
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
            if sector_rs_positive: breakout_score += 10       # NEW
            if liq_data['is_liquid']: breakout_score += 5     # NEW
            if breakout_confirm['confirmed']: breakout_score += 10  # NEW

            # PULLBACK signals
            is_pullback = -5 <= dist_from_ema20 <= -1.5
            ema_structure_good = ema20.iloc[-1] > ema50.iloc[-1]
            if is_pullback: pullback_score += 25
            if ema_structure_good: pullback_score += 20
            if 35 <= curr_rsi <= 55: pullback_score += 15
            if volume_ratio < 1.0: pullback_score += 10
            if ema20.iloc[-1] > ema20.iloc[-5]: pullback_score += 5
            if sector_rs_positive: pullback_score += 10       # NEW
            if liq_data['is_liquid']: pullback_score += 5     # NEW

            # MOMENTUM signals
            strong_uptrend = curr_price > ema20.iloc[-1] > ema50.iloc[-1]
            if strong_uptrend: momentum_score += 25
            if 55 <= curr_rsi <= 80: momentum_score += 20     # WIDENED from 72
            if dist_from_ema20 > 3: momentum_score += 15
            if volume_ratio > 1.2: momentum_score += 10
            momentum_5d = ((curr_price - close.iloc[-5]) / close.iloc[-5]) * 100
            if momentum_5d > 3: momentum_score += 5
            if rs_data['composite'] > 4: momentum_score += 10 # NEW: multi-TF RS
            if sector_rs_positive: momentum_score += 10        # NEW
            if liq_data['is_liquid']: momentum_score += 5      # NEW

            # POWER PLAY signals (NEW)
            if rsi_info['is_power_play']:
                power_play_score += 30
            if strong_uptrend: power_play_score += 20
            if rs_data['composite'] > 6: power_play_score += 20
            if volume_ratio > 1.3: power_play_score += 10
            if sector_rs_positive: power_play_score += 10
            if liq_data['is_liquid']: power_play_score += 5
            if momentum_5d > 5: power_play_score += 5

            # ═══ DETERMINE SETUP — always return best, never UNKNOWN ═══
            scores = {
                SetupType.BREAKOUT: breakout_score,
                SetupType.PULLBACK: pullback_score,
                SetupType.MOMENTUM: momentum_score,
                SetupType.POWER_PLAY: power_play_score,
            }

            setup_type = max(scores, key=scores.get)
            max_possible = 100
            confidence = scores[setup_type] / max_possible

            # Probability tier (IMPROVEMENT #8 + #12)
            if confidence >= 0.75:
                tier = ProbabilityTier.A_PLUS
            elif confidence >= 0.60:
                tier = ProbabilityTier.A
            elif confidence >= 0.45:
                tier = ProbabilityTier.B
            else:
                tier = ProbabilityTier.C  # Still return setup, just size small

            # Trade status
            if confidence >= 0.60 and liq_data['is_liquid']:
                status = TradeStatus.READY
            elif confidence >= 0.40:
                status = TradeStatus.WATCH
            else:
                status = TradeStatus.AVOID

            # If sector RS negative and it's breakout, downgrade
            if setup_type == SetupType.BREAKOUT and not sector_rs_positive:
                if tier in (ProbabilityTier.A_PLUS, ProbabilityTier.A):
                    tier = ProbabilityTier.B
                    status = TradeStatus.WATCH

            # ── STAGE-1 IMPROVEMENT: Hard Kill Conditions Layer ───────────────
            # These are VETO conditions — override scoring regardless of grade.
            # One critical flaw should invalidate even a B-grade setup.
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

            if kill_triggered:
                tier = ProbabilityTier.C
                status = TradeStatus.AVOID

            # ── STAGE-1 IMPROVEMENT: Setup Context Tag ────────────────────────
            # Late trend breakouts in Indian markets fail more frequently.
            # Detect whether stock is in Early / Mid / Late trend phase.
            try:
                ema200 = ta.ema(close, 200)
                dist_from_ema200 = ((curr_price - ema200.iloc[-1]) / ema200.iloc[-1]) * 100
                rs_composite_val = rs_data.get('composite', 0)
                if dist_from_ema200 < 5 and rs_composite_val < 3:
                    trend_context = "Early_Trend"
                elif dist_from_ema200 < 15 and rs_composite_val < 8:
                    trend_context = "Mid_Trend"
                else:
                    trend_context = "Late_Trend"
                # Late trend breakouts get an extra caution flag
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
                'volume_ratio': round(volume_ratio, 2),
                'dist_from_ema20': round(dist_from_ema20, 2),
                'is_contracting': is_contracting,
                'allocation_r': TIER_ALLOCATION[tier] * gap_data.get('position_size_factor', 1.0),
                # Stage-1 additions
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

    # ── API compatibility alias ─────────────────────────────────────────
    # api_server.py calls SetupClassifier.classify(...) — this alias makes
    # both names valid so the server never throws AttributeError.
    classify = classify_setup


# ═══════════════════════════════════════════════════════════════════════
#  LAYER 3b: SETUP SCORER (Enhanced with new weights)
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

        # Volatility contraction
        atr_20_ago = atr.iloc[-20] if len(atr) >= 20 else curr_atr
        atr_change = ((curr_atr - atr_20_ago) / atr_20_ago) * 100 if atr_20_ago > 0 else 0
        is_contracting = atr_change < -10
        scores['volatility_contraction'] = weights['volatility_contraction'] if is_contracting else 0
        if not is_contracting: mandatory_passed = False

        # Volume expansion
        volume_ratio = curr_volume / avg_volume if avg_volume > 0 else 1
        scores['volume_expansion'] = min(weights['volume_expansion'], weights['volume_expansion'] * (volume_ratio / 1.5))
        if volume_ratio < 1.5: mandatory_passed = False

        # Location quality
        recent_high = high.tail(20).max()
        res_dist = (recent_high - curr_price) / curr_atr if curr_atr > 0 else 0
        scores['location_quality'] = weights['location_quality'] if res_dist >= 1.5 else 0

        # Regime
        regime_info = MarketRegimeEngine.detect_regime()
        regime_aligned = regime_info['type'] != RegimeType.RISK_OFF
        scores['regime_alignment'] = weights['regime_alignment'] if regime_aligned else 0
        if not regime_aligned: mandatory_passed = False

        # Candle anatomy
        candle_body = abs(close.iloc[-1] - df['Open'].iloc[-1])
        candle_range = high.iloc[-1] - low.iloc[-1]
        body_ratio = candle_body / candle_range if candle_range > 0 else 0
        scores['candle_anatomy'] = weights['candle_anatomy'] * body_ratio

        # RS (multi-TF)
        rs_data = MultiTimeframeRS.calculate(close, bench)
        scores['relative_strength'] = min(weights['relative_strength'],
                                          weights['relative_strength'] * max(0, rs_data['composite']) / 10)

        # NEW: Sector RS
        scores['sector_rs'] = weights['sector_rs'] if sector_ok else 0

        # NEW: Breadth confirmation
        scores['breadth_confirm'] = weights['breadth_confirm'] if breadth_ok else 0

        # NEW: Liquidity
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
                'ema_structure': ema_structure, 'distance_from_ema20': round(dist, 2),
                'rsi': round(curr_rsi, 2), 'volume_ratio': round(vol_ratio, 2),
            }
        }

    @staticmethod
    def _score_momentum(df, bench, sector_ok, breadth_ok, liq_score, setup_type=SetupType.MOMENTUM) -> Dict:
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
                                          weights['relative_strength'] * max(0, rs_data['composite']) / 10)
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
#  LAYER 4: RISK LAYER — ATR + Beta + Gap + Event
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
        else:  # MOMENTUM
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
#  LAYER 5: CAPITAL LAYER — Probability Tiered Sizing
# ═══════════════════════════════════════════════════════════════════════

class PositionSizingEngine:
    """
    IMPROVEMENT #8: Position size tied to probability tier.
    A+ → 1.0R, A → 0.75R, B → 0.5R, C → 0.25R
    Then further adjusted by gap risk.
    """

    @staticmethod
    def calculate_position(total_capital: float, risk_per_trade_pct: float,
                           entry: float, stop_loss: float,
                           tier: ProbabilityTier,
                           gap_risk_factor: float = 1.0) -> Dict:
        risk_per_share = abs(entry - stop_loss)
        if risk_per_share <= 0:
            return {'shares': 0, 'capital_required': 0, 'risk_amount': 0}

        # Base R from tier
        r_multiplier = TIER_ALLOCATION[tier]

        # Adjust for gap risk
        effective_r = r_multiplier * gap_risk_factor

        # Max risk amount
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
    """
    IMPROVEMENT #11: Per-setup expectancy tracking.
    Disables setups temporarily if expectancy goes negative.
    """

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
        """Record a completed trade result."""
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

        # Disable if negative expectancy over 10+ trades
        is_active = True
        if total >= 10 and expectancy < -0.1:
            is_active = False

        # ── STAGE-1 IMPROVEMENT: Edge Stability Score ─────────────────────────
        # Track rolling 20-trade expectancy to detect edge degradation early.
        history = d.get('history', [])
        rolling_exp = 0.0
        edge_stability = 'UNKNOWN'
        confidence_meter = 'Gray'    # Gray=insufficient data, Green=good, Orange=flat, Red=degrading

        if len(history) >= 5:
            # Rolling last 20 trades
            recent = history[-20:]
            r_wins = [h for h in recent if h.get('won')]
            r_losses = [h for h in recent if not h.get('won')]
            r_wr = len(r_wins) / len(recent)
            r_lr = len(r_losses) / len(recent)
            r_avg_win = np.mean([abs(h['r']) for h in r_wins]) if r_wins else 0
            r_avg_loss = np.mean([abs(h['r']) for h in r_losses]) if r_losses else 0
            rolling_exp = (r_wr * r_avg_win) - (r_lr * r_avg_loss)

            # Trend: is expectancy improving or degrading?
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
            # Strategy Confidence Meter (displayed next to Freddy Gauge)
            confidence_meter = (
                'Green' if rolling_exp > 0 and trend >= 0 else
                'Orange' if rolling_exp >= -0.1 else
                'Red'
            )
            # Auto-reduce sizing if edge degrading
            size_reduction = 0.0
            if edge_stability in ('DEGRADING', 'COLLAPSED'):
                size_reduction = 0.25  # Suggest 25% size reduction

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
            # Stage-1 additions
            'rolling_expectancy_20': round(rolling_exp, 3),
            'edge_stability': edge_stability,
            'confidence_meter': confidence_meter,
            'suggested_size_reduction': size_reduction,
        }

    def _calc_slice_expectancy(self, history_slice: list) -> float:
        """Helper to compute expectancy for a slice of trade history."""
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
#  PLAYBOOK GENERATOR (Enhanced)
# ═══════════════════════════════════════════════════════════════════════

class PlaybookGenerator:

    @staticmethod
    def generate_playbook(ticker: str, setup_type: SetupType, score_data: Dict,
                          risk_data: Dict, regime_info: Dict,
                          classification: Dict = None) -> Dict:
        config = SETUP_CONFIGS.get(setup_type, SETUP_CONFIGS[SetupType.MOMENTUM])

        tier = classification.get('probability_tier', ProbabilityTier.B) if classification else ProbabilityTier.B
        status = classification.get('status', TradeStatus.WATCH) if classification else TradeStatus.WATCH

        playbook = {
            'ticker': ticker,
            'setup_type': setup_type.value,
            'probability_tier': tier.value,
            'status': status.value,
            'confidence': PlaybookGenerator._calc_confidence(score_data, regime_info),
            'allocation_r': TIER_ALLOCATION.get(tier, 0.5),
            'why_selected': PlaybookGenerator._explain(setup_type, score_data, regime_info),
            'entry_plan': PlaybookGenerator._entry_plan(setup_type, risk_data),
            'what_to_watch': PlaybookGenerator._watch_items(setup_type),
            'invalidation_rules': PlaybookGenerator._invalidation(setup_type, risk_data),
            'risk_comment': PlaybookGenerator._risk_comment(setup_type, risk_data),
            'time_decay': config.time_decay_days,
            'risk_data': risk_data,
            # NEW fields
            'regime_aligned': regime_info['type'] != RegimeType.RISK_OFF,
            'rr_ratio': risk_data.get('rr1', 0),
            'event_risk': classification.get('gap_risk', {}).get('risk_level', 'LOW') if classification else 'LOW',
            'sector_rs_aligned': classification.get('sector_rs_positive', False) if classification else False,
            'liquidity_ok': classification.get('liquidity', {}).get('is_liquid', True) if classification else True,
            'breakout_confirmed': classification.get('breakout_confirmation', {}).get('confirmed', True) if classification else True,
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
    def _entry_plan(setup_type, risk_data):
        plans = {
            SetupType.BREAKOUT: {
                'primary': f"Buy above ₹{risk_data['entry']} with vol > 1.5x AFTER 2-day confirmation",
                'alternate': "If intraday pullback to breakout level with volume support",
                'avoid': "Gap up > 2%, late-day breakout, no 2-day confirmation",
            },
            SetupType.PULLBACK: {
                'primary': f"Buy on bounce above ₹{risk_data['entry']} with volume expansion",
                'alternate': "Enter on reversal candle at EMA20 support",
                'avoid': "Continued breakdown below EMA20, volume spike on decline",
            },
            SetupType.MOMENTUM: {
                'primary': f"Buy on continuation above ₹{risk_data['entry']} with strong candle",
                'alternate': "Scale in on shallow pullback to EMA10/20",
                'avoid': "Parabolic move > 3 ATR in 2 days, bearish reversal candle",
            },
            SetupType.POWER_PLAY: {
                'primary': f"Buy on momentum continuation above ₹{risk_data['entry']}. Tight trail.",
                'alternate': "Scale in on any dip to EMA10 — these don't wait",
                'avoid': "Exhaustion gap > 3%, RSI > 85, parabolic blow-off",
            },
        }
        return plans.get(setup_type, {'primary': f"Buy at ₹{risk_data['entry']}"})

    @staticmethod
    def _watch_items(setup_type):
        items = {
            SetupType.BREAKOUT: [
                "2-day close above resistance (CONFIRMED?)",
                "Volume expansion continuation",
                "Sector RS still positive",
                "Breadth supporting move",
                "No gap-down reversal",
            ],
            SetupType.PULLBACK: [
                "Bounce candle with volume",
                "Holding above EMA20",
                "RSI turning up from 35-50 zone",
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
#  ACTIVE TRADE EVALUATOR (Enhanced)
# ═══════════════════════════════════════════════════════════════════════

class ActiveTradeEvaluator:

    @staticmethod
    def evaluate_trade(ticker: str, entry_price: float, stop_loss: float,
                       target: float, setup_type: SetupType,
                       entry_date: datetime) -> Dict:
        try:
            df = yf.download(ticker, period="3mo", progress=False)
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
                status, color = "Strong", "green"
            elif health_score >= 50:
                status, color = "Warning", "orange"
            else:
                status, color = "Weak", "red"

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
    def _calc_trailing_sl(entry, current, original_sl, pnl, atr, setup_type):
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
#  PORTFOLIO-LEVEL RISK ENGINE — Critical Missing Component (Stage-1)
# ═══════════════════════════════════════════════════════════════════════

class PortfolioRiskEngine:
    """
    STAGE-1 CRITICAL ADDITION: Portfolio-level risk controls.

    Indian markets correct sector-wide and gap on global cues.
    Trade-by-trade sizing is insufficient — portfolio exposure must be capped.

    Rules:
      • Max 6 open positions
      • Max 2 positions per sector
      • Max 40% capital in high-beta (β > 1.3) stocks
      • Portfolio max drawdown trigger → reduce all by 25%
    """

    MAX_POSITIONS = 6
    MAX_PER_SECTOR = 2
    MAX_HIGH_BETA_PCT = 40.0    # % of total capital
    DRAWDOWN_TRIGGER_PCT = 8.0  # Portfolio drawdown % that triggers size reduction
    DRAWDOWN_REDUCTION = 0.75   # Reduce all sizing to 75% on trigger

    @staticmethod
    def validate_new_trade(
        proposed_ticker: str,
        proposed_sector: str,
        proposed_beta: float,
        proposed_capital_pct: float,
        active_positions: List[Dict],  # [{ticker, sector, beta, capital_pct}]
        portfolio_current_drawdown_pct: float = 0.0
    ) -> Dict:
        """
        Check whether a new trade violates portfolio-level rules.
        Returns {allowed: bool, reasons: list, adjusted_size_pct: float}
        """
        violations = []
        warnings = []
        allowed = True
        size_multiplier = 1.0

        # Rule 1: Max position count
        if len(active_positions) >= PortfolioRiskEngine.MAX_POSITIONS:
            violations.append(f"⛔ Max {PortfolioRiskEngine.MAX_POSITIONS} positions reached ({len(active_positions)} open)")
            allowed = False

        # Rule 2: Max per sector
        sector_count = sum(1 for p in active_positions
                           if p.get('sector', '').lower() == proposed_sector.lower())
        if sector_count >= PortfolioRiskEngine.MAX_PER_SECTOR:
            violations.append(f"⛔ Sector '{proposed_sector}' already has {sector_count} positions (max {PortfolioRiskEngine.MAX_PER_SECTOR})")
            allowed = False

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

        # Rule 4: Portfolio drawdown trigger
        if portfolio_current_drawdown_pct >= PortfolioRiskEngine.DRAWDOWN_TRIGGER_PCT:
            size_multiplier = PortfolioRiskEngine.DRAWDOWN_REDUCTION
            warnings.append(
                f"⚠️ Portfolio drawdown {portfolio_current_drawdown_pct:.1f}% triggered size reduction to "
                f"{PortfolioRiskEngine.DRAWDOWN_REDUCTION * 100:.0f}%"
            )

        # Portfolio stats summary
        total_deployed_pct = sum(p.get('capital_pct', 0) for p in active_positions)
        high_beta_deployed = sum(p.get('capital_pct', 0) for p in active_positions if p.get('beta', 1.0) > 1.3)
        sectors_used = {}
        for p in active_positions:
            s = p.get('sector', 'Unknown')
            sectors_used[s] = sectors_used.get(s, 0) + 1

        return {
            'allowed': allowed,
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
    def get_operating_mode(regime_type: RegimeType, breadth_slope: float,
                           vol_regime: str) -> Dict:
        """
        STAGE-1 ADVANCED: 3-Mode Adaptive Operating System.
        Switches mode automatically based on Regime + Breadth + Volatility.

        Conservative Mode (Late Cycle): smaller sizes, fewer positions
        Balanced Mode: standard operation
        Aggressive Mode (Early Bull): larger sizes, full position count
        """
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
#  MASTER ORCHESTRATOR — Ties all layers together
# ═══════════════════════════════════════════════════════════════════════

class FreddyEngine:
    """
    Master orchestrator. Runs all 6 layers for full analysis.
    Protected by authentication for personal use only.
    """

    def __init__(self, total_capital: float = 1000000.0, risk_per_trade: float = 2.0):
        self._auth = AuthManager()
        self._expectancy = TradeExpectancyTracker()
        self.total_capital = total_capital
        self.risk_per_trade = risk_per_trade
        self._sector_cache = {}
        self._sector_cache_time = 0

    def login(self, username: str, password: str, ip: str = "127.0.0.1") -> Optional[str]:
        return self._auth.login(username, password, ip)

    def logout(self, token: str):
        self._auth.logout(token)

    # ── Public (no-auth) helpers for api_server.py endpoints ───────────
    # The market-regime endpoint was returning 500 because @require_auth
    # was rejecting unauthenticated calls before any session was established.
    # These public wrappers bypass auth for read-only market data.

    def get_market_regime_public(self) -> Dict:
        """Public market regime — no auth required. Safe: read-only data."""
        try:
            breadth = MarketBreadthEngine.calculate_breadth()
            volatility = VolatilityRegimeEngine.detect_state()
            regime = MarketRegimeEngine.detect_regime(breadth_data=breadth, volatility_data=volatility)
            sectors = self._get_sector_data()
            return {
                'freddy_gauge': {
                    'mpi': regime['mpi'],
                    'degrees': regime['gauge_degrees'],
                    'regime': regime['type'].value,
                    'label': 'Risk-Off' if regime['mpi'] < 35 else ('Risk-On' if regime['mpi'] > 65 else 'Neutral'),
                },
                'breadth': breadth,
                'volatility': {
                    'state': volatility['state'].value,
                    'favored': volatility.get('favored_strategies', []),
                },
                'sectors': sectors,
                'regime_details': regime['details'],
                'timestamp': datetime.now(IST_TZ).strftime('%Y-%m-%d %H:%M IST'),
            }
        except Exception as e:
            logger.error(f"Market regime public error: {e}")
            return {'error': str(e), 'regime': 'Neutral', 'mpi': 50}

    def scan_stocks_public(self, tickers: List[str],
                           sector_map: Dict[str, str] = None,
                           market_cap_map: Dict[str, str] = None) -> List[Dict]:
        """
        Public scanner — no auth required.
        Replaces the auth-gated scan that was causing 500s.
        """
        sector_data = self._get_sector_data()
        results = []
        for ticker in tickers:
            try:
                df = yf.download(ticker, period="6mo", progress=False)
                if df.empty:
                    continue
                sector = (sector_map or {}).get(ticker, "")
                cap_cat = (market_cap_map or {}).get(ticker, "midcap")
                classification = SetupClassifier.classify_setup(
                    df, ticker, sector, cap_cat, 999, 1.0, sector_data
                )
                entry_price = float(df['Close'].squeeze().iloc[-1])
                risk_data = RiskCalculator.calculate_risk_params(
                    df, classification['setup_type'], entry_price,
                    classification.get('gap_risk')
                )
                results.append({
                    'ticker': ticker,
                    'setup_type': classification['setup_type'].value,
                    'tier': classification['probability_tier'].value,
                    'status': classification['status'].value,
                    'confidence': classification['confidence'],
                    'entry': risk_data['entry'],
                    'stop_loss': risk_data['stop_loss'],
                    'target1': risk_data['target1'],
                    'rr1': risk_data['rr1'],
                    'sector_rs': classification.get('sector_rs_positive', False),
                    'liquidity_ok': classification.get('liquidity', {}).get('is_liquid', False),
                    'volume_ratio': classification.get('volume_ratio', 1.0),
                })
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
        return sorted(results, key=lambda x: x['confidence'], reverse=True)

    def _get_sector_data(self) -> Dict:
        """Cache sector data for 15 minutes."""
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
        """
        Complete 6-layer analysis for a single stock.
        Returns everything needed for a trade card.
        """
        # Layer 1: Market
        breadth = MarketBreadthEngine.calculate_breadth()
        volatility = VolatilityRegimeEngine.detect_state()
        regime = MarketRegimeEngine.detect_regime(breadth_data=breadth, volatility_data=volatility)

        # Layer 2: Sector
        sector_data = self._get_sector_data()

        # Fetch stock data
        df = yf.download(ticker, period="6mo", progress=False)
        if df.empty:
            return {'error': f'No data for {ticker}'}

        # Layer 3: Stock classification
        classification = SetupClassifier.classify_setup(
            df, ticker, sector, market_cap_cat,
            days_to_earnings, beta, sector_data
        )

        setup_type = classification['setup_type']
        if setup_type == SetupType.UNKNOWN:
            setup_type = SetupType.PULLBACK  # Fallback

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

        # Override status if setup has negative expectancy
        if not expectancy['is_active']:
            classification['status'] = TradeStatus.AVOID
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
                    'breadth': regime['breadth'],
                    'volatility': {
                        'state': volatility['state'].value,
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
                    # Stage-1 additions
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
                'color': {
                    TradeStatus.READY: '#00e676',
                    TradeStatus.WATCH: '#ff9100',
                    TradeStatus.AVOID: '#ff1744',
                    TradeStatus.EXPIRED: '#9e9e9e',
                }.get(classification['status'], '#9e9e9e'),
            },
        }

    @require_auth
    def market_overview(self, auth_token: str = "") -> Dict:
        """Dashboard market overview — Freddy Gauge + sector gauges."""
        breadth = MarketBreadthEngine.calculate_breadth()
        volatility = VolatilityRegimeEngine.detect_state()
        regime = MarketRegimeEngine.detect_regime(breadth_data=breadth, volatility_data=volatility)
        sectors = self._get_sector_data()

        return {
            'freddy_gauge': {
                'mpi': regime['mpi'],
                'degrees': regime['gauge_degrees'],
                'regime': regime['type'].value,
                'label': 'Risk-Off' if regime['mpi'] < 35 else ('Risk-On' if regime['mpi'] > 65 else 'Neutral'),
            },
            'breadth': breadth,
            'volatility': {
                'state': volatility['state'].value,
                'favored': volatility.get('favored_strategies', []),
            },
            'sectors': sectors,
            'regime_details': regime['details'],
            'timestamp': datetime.now(IST_TZ).strftime('%Y-%m-%d %H:%M IST'),
        }

    @require_auth
    def record_trade_result(self, setup_type: str, r_result: float,
                            won: bool, auth_token: str = ""):
        """Record completed trade for expectancy tracking."""
        self._expectancy.record_trade(setup_type, r_result, won)

    @require_auth
    def get_performance_stats(self, auth_token: str = "") -> Dict:
        """Performance evolution panel data."""
        return self._expectancy.get_all_stats()


# ═══════════════════════════════════════════════════════════════════════
#  EXPORTS
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    'FreddyEngine', 'AuthManager', 'SecurityConfig',
    'SetupType', 'RegimeType', 'VolatilityState', 'ProbabilityTier', 'TradeStatus',
    'MarketRegimeEngine', 'MarketBreadthEngine', 'VolatilityRegimeEngine',
    'SectorLeadershipEngine', 'MultiTimeframeRS',
    'SetupClassifier', 'SetupScorer', 'RSIClassifier',
    'LiquidityFilter', 'GapRiskModel', 'BreakoutConfirmation',
    'RiskCalculator', 'PositionSizingEngine',
    'PlaybookGenerator', 'ActiveTradeEvaluator',
    'TradeExpectancyTracker',
    # v3 additions
    'PortfolioRiskEngine',
]
