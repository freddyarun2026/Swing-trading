"""
Core Trading Engine - UI Agnostic
Implements setup-specific logic for Breakout, Pullback, and Momentum strategies
"""

import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pytz

IST_TZ = pytz.timezone('Asia/Kolkata')


class SetupType(Enum):
    """Trade setup classification"""
    BREAKOUT = "Breakout"
    PULLBACK = "Pullback"
    MOMENTUM = "Momentum"
    UNKNOWN = "Unknown"


class RegimeType(Enum):
    """Market regime types"""
    RISK_ON = "Risk-On"
    NEUTRAL = "Neutral"
    RISK_OFF = "Risk-Off"


@dataclass
class SetupConfig:
    """Configuration for each setup type"""
    name: str
    mandatory_filters: Dict[str, any]
    scoring_weights: Dict[str, int]
    risk_profile: Dict[str, any]
    time_decay_days: int


# Setup-specific configurations
SETUP_CONFIGS = {
    SetupType.BREAKOUT: SetupConfig(
        name="Breakout",
        mandatory_filters={
            'volatility_contraction': True,
            'min_volume_ratio': 1.5,
            'min_resistance_distance': 1.5,  # ATR
            'max_gap_pct': 1.5,
            'sector_rs_positive': True,
            'regime_not_risk_off': True
        },
        scoring_weights={
            'volatility_contraction': 20,
            'volume_expansion': 20,
            'location_quality': 20,
            'regime_alignment': 15,
            'candle_anatomy': 15,
            'relative_strength': 10
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
            'rsi_range': (40, 55),
            'pullback_volume_low': True,
            'bounce_volume_expansion': True,
            'weekly_aligned': True
        },
        scoring_weights={
            'trend_structure': 25,
            'pullback_quality': 20,
            'volume_pattern': 20,
            'weekly_alignment': 15,
            'rsi_zone': 10,
            'regime': 10
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
            'min_rs_vs_benchmark': 4.0,  # %
            'rsi_range': (55, 72),
            'min_volume_ratio': 1.2,
            'price_above_emas': True,
            'no_bearish_reversal': True
        },
        scoring_weights={
            'relative_strength': 30,
            'volume': 20,
            'trend_alignment': 20,
            'rsi_zone': 15,
            'regime': 10,
            'candle': 5
        },
        risk_profile={
            'sl_atr_multiplier': (1.4, 1.8),
            'max_risk_pct': 3.5,
            'trailing_trigger_atr': 0.8,
            'move_required_days': 2,
            'trail_below_ema': 20
        },
        time_decay_days=2
    )
}


class MarketRegimeEngine:
    """Enhanced market regime detection"""
    
    @staticmethod
    def detect_regime(ticker: str = "^NSEI", lookback_days: int = 60) -> Dict:
        """
        Detect market regime with comprehensive scoring
        Returns: regime_type, score, details, color
        """
        try:
            df = yf.download(ticker, period="1y", progress=False)
            if df.empty:
                return {
                    'type': RegimeType.NEUTRAL,
                    'score': 50,
                    'details': {},
                    'color': '#ffa500',
                    'breadth': 50
                }
            
            close = df['Close'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            volume = df['Volume'].squeeze()
            
            # Technical indicators
            ema20 = ta.ema(close, 20)
            ema50 = ta.ema(close, 50)
            ema200 = ta.ema(close, 200)
            rsi = ta.rsi(close, 14)
            macd = ta.macd(close)
            
            curr_price = close.iloc[-1]
            
            score = 0
            details = {}
            
            # 1. TREND STRENGTH (0-30 points)
            trend_score = 0
            if curr_price > ema20.iloc[-1]: trend_score += 10
            if curr_price > ema50.iloc[-1]: trend_score += 10
            if curr_price > ema200.iloc[-1]: trend_score += 10
            
            score += trend_score
            details['trend_score'] = trend_score
            details['price_vs_ema200'] = round(((curr_price - ema200.iloc[-1]) / ema200.iloc[-1]) * 100, 2)
            
            # 2. MOMENTUM (0-25 points)
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
            
            # 3. RSI & MACD (0-20 points)
            curr_rsi = rsi.iloc[-1]
            indicator_score = 0
            
            if 50 < curr_rsi < 70: indicator_score += 10
            elif 45 < curr_rsi <= 50: indicator_score += 7
            
            if macd['MACD_12_26_9'].iloc[-1] > macd['MACDs_12_26_9'].iloc[-1]:
                indicator_score += 10
            
            score += indicator_score
            details['rsi'] = round(curr_rsi, 2)
            
            # 4. VOLATILITY (0-15 points)
            recent_returns = close.pct_change().tail(20)
            volatility = recent_returns.std() * np.sqrt(252)
            
            vol_score = 15 if volatility < 0.15 else (10 if volatility < 0.25 else 5)
            score += vol_score
            details['volatility'] = round(volatility * 100, 2)
            
            # 5. PRICE ACTION (0-10 points)
            recent_highs = high.tail(10)
            higher_highs = recent_highs.iloc[-1] > recent_highs.iloc[-5]
            score += 10 if higher_highs else 5
            
            # Determine regime type
            if score >= 70:
                regime_type = RegimeType.RISK_ON
                color = '#00ff00'
            elif score >= 40:
                regime_type = RegimeType.NEUTRAL
                color = '#ffa500'
            else:
                regime_type = RegimeType.RISK_OFF
                color = '#ff0000'
            
            # Market breadth (simplified)
            breadth_pct = score  # Simplified - in production, calculate actual breadth
            
            return {
                'type': regime_type,
                'score': score,
                'details': details,
                'color': color,
                'breadth': breadth_pct,
                'message': MarketRegimeEngine._get_regime_message(regime_type, score)
            }
            
        except Exception as e:
            return {
                'type': RegimeType.NEUTRAL,
                'score': 50,
                'details': {'error': str(e)},
                'color': '#ffa500',
                'breadth': 50,
                'message': 'Unable to determine regime'
            }
    
    @staticmethod
    def _get_regime_message(regime_type: RegimeType, score: int) -> str:
        """Generate contextual regime message"""
        messages = {
            RegimeType.RISK_ON: f"Strong bullish environment (Score: {score}). Favor breakouts and momentum.",
            RegimeType.NEUTRAL: f"Mixed market conditions (Score: {score}). Focus on high-quality pullbacks.",
            RegimeType.RISK_OFF: f"Defensive environment (Score: {score}). Avoid new positions or trade pullbacks only."
        }
        return messages.get(regime_type, "Unknown regime")
    
    @staticmethod
    def get_strategy_allowance(regime_type: RegimeType) -> Dict[SetupType, bool]:
        """Determine which strategies are allowed in current regime"""
        allowance = {
            RegimeType.RISK_ON: {
                SetupType.BREAKOUT: True,
                SetupType.PULLBACK: True,
                SetupType.MOMENTUM: True
            },
            RegimeType.NEUTRAL: {
                SetupType.BREAKOUT: False,
                SetupType.PULLBACK: True,
                SetupType.MOMENTUM: False
            },
            RegimeType.RISK_OFF: {
                SetupType.BREAKOUT: False,
                SetupType.PULLBACK: True,
                SetupType.MOMENTUM: False
            }
        }
        return allowance.get(regime_type, {})


class SetupClassifier:
    """Classifies stock setups into Breakout, Pullback, or Momentum"""
    
    @staticmethod
    def classify_setup(df: pd.DataFrame, ticker: str) -> Tuple[SetupType, float]:
        """
        Analyze price action and classify setup type
        Returns: (setup_type, confidence_score)
        """
        try:
            close = df['Close'].squeeze()
            high = df['High'].squeeze()
            low = df['Low'].squeeze()
            volume = df['Volume'].squeeze()
            
            # Calculate indicators
            ema20 = ta.ema(close, 20)
            ema50 = ta.ema(close, 50)
            rsi = ta.rsi(close, 14)
            atr = ta.atr(high, low, close, 14)
            
            curr_price = close.iloc[-1]
            curr_rsi = rsi.iloc[-1]
            curr_atr = atr.iloc[-1]
            
            # Recent price action
            recent_high_20 = high.tail(20).max()
            recent_low_20 = low.tail(20).min()
            range_pct = ((recent_high_20 - recent_low_20) / recent_low_20) * 100
            
            # Distance from EMA20
            dist_from_ema20 = ((curr_price - ema20.iloc[-1]) / ema20.iloc[-1]) * 100
            
            # Volume analysis
            avg_volume_20 = volume.tail(20).mean()
            curr_volume = volume.iloc[-1]
            volume_ratio = curr_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            
            # Volatility contraction
            atr_20_ago = atr.iloc[-20] if len(atr) >= 20 else curr_atr
            atr_change_pct = ((curr_atr - atr_20_ago) / atr_20_ago) * 100 if atr_20_ago > 0 else 0
            is_contracting = atr_change_pct < -10
            
            # Classification scores
            breakout_score = 0
            pullback_score = 0
            momentum_score = 0
            
            # BREAKOUT signals
            near_resistance = curr_price >= recent_high_20 * 0.98
            if near_resistance: breakout_score += 30
            if is_contracting: breakout_score += 25
            if volume_ratio > 1.5: breakout_score += 20
            if range_pct < 8: breakout_score += 15  # Consolidation
            if abs(dist_from_ema20) < 3: breakout_score += 10
            
            # PULLBACK signals
            is_pullback = -5 <= dist_from_ema20 <= -2
            ema_structure_good = ema20.iloc[-1] > ema50.iloc[-1]
            if is_pullback: pullback_score += 30
            if ema_structure_good: pullback_score += 25
            if 40 <= curr_rsi <= 55: pullback_score += 20
            if volume_ratio < 1.0: pullback_score += 15  # Low volume pullback
            if ema20.iloc[-1] > ema20.iloc[-5]: pullback_score += 10  # Rising EMA
            
            # MOMENTUM signals
            strong_uptrend = curr_price > ema20.iloc[-1] > ema50.iloc[-1]
            if strong_uptrend: momentum_score += 30
            if 60 <= curr_rsi <= 72: momentum_score += 25
            if dist_from_ema20 > 3: momentum_score += 20
            if volume_ratio > 1.2: momentum_score += 15
            
            # Recent momentum
            momentum_5d = ((curr_price - close.iloc[-5]) / close.iloc[-5]) * 100
            if momentum_5d > 3: momentum_score += 10
            
            # Determine setup type
            scores = {
                SetupType.BREAKOUT: breakout_score,
                SetupType.PULLBACK: pullback_score,
                SetupType.MOMENTUM: momentum_score
            }
            
            setup_type = max(scores, key=scores.get)
            confidence = scores[setup_type] / 100.0
            
            # Require minimum confidence
            if confidence < 0.5:
                return SetupType.UNKNOWN, confidence
            
            return setup_type, confidence
            
        except Exception as e:
            return SetupType.UNKNOWN, 0.0


class SetupScorer:
    """Setup-specific scoring engine"""
    
    @staticmethod
    def score_setup(df: pd.DataFrame, setup_type: SetupType, benchmark_ticker: str = "^NSEI") -> Dict:
        """
        Score a setup based on its type-specific criteria
        Returns comprehensive scoring breakdown
        """
        if setup_type == SetupType.BREAKOUT:
            return SetupScorer._score_breakout(df, benchmark_ticker)
        elif setup_type == SetupType.PULLBACK:
            return SetupScorer._score_pullback(df, benchmark_ticker)
        elif setup_type == SetupType.MOMENTUM:
            return SetupScorer._score_momentum(df, benchmark_ticker)
        else:
            return {'total_score': 0, 'details': {}, 'passed_filters': False}
    
    @staticmethod
    def _score_breakout(df: pd.DataFrame, benchmark_ticker: str) -> Dict:
        """Breakout-specific scoring"""
        config = SETUP_CONFIGS[SetupType.BREAKOUT]
        weights = config.scoring_weights
        
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        volume = df['Volume'].squeeze()
        
        # Indicators
        atr = ta.atr(high, low, close, 14)
        ema20 = ta.ema(close, 20)
        
        curr_price = close.iloc[-1]
        curr_atr = atr.iloc[-1]
        curr_volume = volume.iloc[-1]
        avg_volume = volume.tail(20).mean()
        
        scores = {}
        mandatory_passed = True
        
        # 1. Volatility Contraction (20 points)
        atr_20_ago = atr.iloc[-20] if len(atr) >= 20 else curr_atr
        atr_change = ((curr_atr - atr_20_ago) / atr_20_ago) * 100 if atr_20_ago > 0 else 0
        is_contracting = atr_change < -10
        
        scores['volatility_contraction'] = weights['volatility_contraction'] if is_contracting else 0
        if not is_contracting:
            mandatory_passed = False
        
        # 2. Volume Expansion (20 points)
        volume_ratio = curr_volume / avg_volume if avg_volume > 0 else 1
        volume_score = min(weights['volume_expansion'], weights['volume_expansion'] * (volume_ratio / 1.5))
        scores['volume_expansion'] = volume_score
        
        if volume_ratio < 1.5:
            mandatory_passed = False
        
        # 3. Location Quality (20 points)
        recent_high = high.tail(20).max()
        resistance_distance = (recent_high - curr_price) / curr_atr if curr_atr > 0 else 0
        location_score = weights['location_quality'] if resistance_distance >= 1.5 else 0
        scores['location_quality'] = location_score
        
        if resistance_distance < 1.5:
            mandatory_passed = False
        
        # 4. Regime Alignment (15 points)
        regime_info = MarketRegimeEngine.detect_regime()
        regime_aligned = regime_info['type'] != RegimeType.RISK_OFF
        scores['regime_alignment'] = weights['regime_alignment'] if regime_aligned else 0
        
        if not regime_aligned:
            mandatory_passed = False
        
        # 5. Candle Anatomy (15 points)
        candle_body = abs(close.iloc[-1] - df['Open'].iloc[-1])
        candle_range = high.iloc[-1] - low.iloc[-1]
        body_ratio = candle_body / candle_range if candle_range > 0 else 0
        candle_score = weights['candle_anatomy'] * body_ratio
        scores['candle_anatomy'] = candle_score
        
        # 6. Relative Strength (10 points)
        rs_score = SetupScorer._calculate_relative_strength(close, benchmark_ticker)
        scores['relative_strength'] = min(weights['relative_strength'], weights['relative_strength'] * (rs_score / 100))
        
        total_score = sum(scores.values())
        
        return {
            'total_score': round(total_score, 2),
            'max_score': sum(weights.values()),
            'scores': scores,
            'passed_filters': mandatory_passed,
            'details': {
                'volatility_contraction': is_contracting,
                'volume_ratio': round(volume_ratio, 2),
                'resistance_distance_atr': round(resistance_distance, 2),
                'regime': regime_info['type'].value
            }
        }
    
    @staticmethod
    def _score_pullback(df: pd.DataFrame, benchmark_ticker: str) -> Dict:
        """Pullback-specific scoring"""
        config = SETUP_CONFIGS[SetupType.PULLBACK]
        weights = config.scoring_weights
        
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        volume = df['Volume'].squeeze()
        
        # Indicators
        ema20 = ta.ema(close, 20)
        ema50 = ta.ema(close, 50)
        rsi = ta.rsi(close, 14)
        
        curr_price = close.iloc[-1]
        curr_rsi = rsi.iloc[-1]
        
        scores = {}
        mandatory_passed = True
        
        # 1. Trend Structure (25 points)
        ema_structure = ema20.iloc[-1] > ema50.iloc[-1]
        ema50_rising = ema50.iloc[-1] > ema50.iloc[-5]
        
        if ema_structure and ema50_rising:
            scores['trend_structure'] = weights['trend_structure']
        else:
            scores['trend_structure'] = 0
            mandatory_passed = False
        
        # 2. Pullback Quality (20 points)
        dist_from_ema20 = ((curr_price - ema20.iloc[-1]) / ema20.iloc[-1]) * 100
        pullback_quality = -5 <= dist_from_ema20 <= -1
        
        if pullback_quality:
            scores['pullback_quality'] = weights['pullback_quality']
        else:
            scores['pullback_quality'] = 0
        
        # 3. Volume Pattern (20 points)
        avg_volume = volume.tail(20).mean()
        curr_volume = volume.iloc[-1]
        volume_ratio = curr_volume / avg_volume if avg_volume > 0 else 1
        
        # Looking for low volume on pullback, high volume on bounce
        # Simplified: check if current volume is expanding
        volume_score = min(weights['volume_pattern'], weights['volume_pattern'] * volume_ratio) if volume_ratio > 1.1 else 0
        scores['volume_pattern'] = volume_score
        
        # 4. Weekly Alignment (15 points)
        # Simplified - in production, fetch weekly data
        scores['weekly_alignment'] = weights['weekly_alignment'] * 0.8  # Assume mostly aligned
        
        # 5. RSI Zone (10 points)
        rsi_in_zone = 40 <= curr_rsi <= 55
        scores['rsi_zone'] = weights['rsi_zone'] if rsi_in_zone else 0
        
        if not rsi_in_zone:
            mandatory_passed = False
        
        # 6. Regime (10 points)
        regime_info = MarketRegimeEngine.detect_regime()
        scores['regime'] = weights['regime'] * (regime_info['score'] / 100)
        
        total_score = sum(scores.values())
        
        return {
            'total_score': round(total_score, 2),
            'max_score': sum(weights.values()),
            'scores': scores,
            'passed_filters': mandatory_passed,
            'details': {
                'ema_structure': ema_structure,
                'ema50_rising': ema50_rising,
                'distance_from_ema20': round(dist_from_ema20, 2),
                'rsi': round(curr_rsi, 2),
                'volume_ratio': round(volume_ratio, 2)
            }
        }
    
    @staticmethod
    def _score_momentum(df: pd.DataFrame, benchmark_ticker: str) -> Dict:
        """Momentum-specific scoring"""
        config = SETUP_CONFIGS[SetupType.MOMENTUM]
        weights = config.scoring_weights
        
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        volume = df['Volume'].squeeze()
        
        # Indicators
        ema20 = ta.ema(close, 20)
        ema50 = ta.ema(close, 50)
        rsi = ta.rsi(close, 14)
        
        curr_price = close.iloc[-1]
        curr_rsi = rsi.iloc[-1]
        
        scores = {}
        mandatory_passed = True
        
        # 1. Relative Strength (30 points)
        rs_score = SetupScorer._calculate_relative_strength(close, benchmark_ticker)
        scores['relative_strength'] = min(weights['relative_strength'], weights['relative_strength'] * (rs_score / 100))
        
        if rs_score < 4.0:
            mandatory_passed = False
        
        # 2. Volume (20 points)
        avg_volume = volume.tail(20).mean()
        curr_volume = volume.iloc[-1]
        volume_ratio = curr_volume / avg_volume if avg_volume > 0 else 1
        
        volume_score = min(weights['volume'], weights['volume'] * (volume_ratio / 1.2))
        scores['volume'] = volume_score
        
        if volume_ratio < 1.2:
            mandatory_passed = False
        
        # 3. Trend Alignment (20 points)
        price_above_emas = curr_price > ema20.iloc[-1] > ema50.iloc[-1]
        scores['trend_alignment'] = weights['trend_alignment'] if price_above_emas else 0
        
        if not price_above_emas:
            mandatory_passed = False
        
        # 4. RSI Zone (15 points)
        rsi_in_zone = 55 <= curr_rsi <= 72
        scores['rsi_zone'] = weights['rsi_zone'] if rsi_in_zone else 0
        
        if not rsi_in_zone:
            mandatory_passed = False
        
        # 5. Regime (10 points)
        regime_info = MarketRegimeEngine.detect_regime()
        scores['regime'] = weights['regime'] * (regime_info['score'] / 100)
        
        # 6. Candle (5 points)
        candle_body = abs(close.iloc[-1] - df['Open'].iloc[-1])
        candle_range = high.iloc[-1] - low.iloc[-1]
        body_ratio = candle_body / candle_range if candle_range > 0 else 0
        scores['candle'] = weights['candle'] * body_ratio
        
        total_score = sum(scores.values())
        
        return {
            'total_score': round(total_score, 2),
            'max_score': sum(weights.values()),
            'scores': scores,
            'passed_filters': mandatory_passed,
            'details': {
                'relative_strength': round(rs_score, 2),
                'volume_ratio': round(volume_ratio, 2),
                'price_above_emas': price_above_emas,
                'rsi': round(curr_rsi, 2)
            }
        }
    
    @staticmethod
    def _calculate_relative_strength(close: pd.Series, benchmark_ticker: str) -> float:
        """Calculate relative strength vs benchmark"""
        try:
            # Get benchmark data
            benchmark_df = yf.download(benchmark_ticker, period="3mo", progress=False)
            if benchmark_df.empty:
                return 0.0
            
            benchmark_close = benchmark_df['Close'].squeeze()
            
            # 20-day returns
            stock_return = ((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]) * 100
            benchmark_return = ((benchmark_close.iloc[-1] - benchmark_close.iloc[-20]) / benchmark_close.iloc[-20]) * 100
            
            rs = stock_return - benchmark_return
            return rs
            
        except:
            return 0.0


class RiskCalculator:
    """Setup-specific risk calculation"""
    
    @staticmethod
    def calculate_risk_params(df: pd.DataFrame, setup_type: SetupType, entry_price: float) -> Dict:
        """
        Calculate entry, stop loss, and targets based on setup type
        """
        config = SETUP_CONFIGS.get(setup_type)
        if not config:
            return {}
        
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        
        atr = ta.atr(high, low, close, 14).iloc[-1]
        
        # Setup-specific SL calculation
        if setup_type == SetupType.BREAKOUT:
            sl_multiplier = np.mean(config.risk_profile['sl_atr_multiplier'])
            sl = entry_price - (atr * sl_multiplier)
            target1 = entry_price + (atr * 2.0)
            target2 = entry_price + (atr * 3.5)
            
        elif setup_type == SetupType.PULLBACK:
            # Use swing low for pullback
            recent_low = low.tail(10).min()
            sl_multiplier = np.mean(config.risk_profile['sl_atr_multiplier'])
            sl = min(recent_low, entry_price - (atr * sl_multiplier))
            target1 = entry_price + (atr * 1.5)
            target2 = entry_price + (atr * 2.5)
            
        elif setup_type == SetupType.MOMENTUM:
            ema20 = ta.ema(close, 20).iloc[-1]
            sl_multiplier = np.mean(config.risk_profile['sl_atr_multiplier'])
            sl = max(ema20 - atr, entry_price - (atr * sl_multiplier))
            target1 = entry_price + (atr * 2.5)
            target2 = entry_price + (atr * 4.0)
        
        else:
            sl = entry_price - (atr * 1.5)
            target1 = entry_price + (atr * 2.0)
            target2 = entry_price + (atr * 3.0)
        
        risk_pct = ((entry_price - sl) / entry_price) * 100
        reward1_pct = ((target1 - entry_price) / entry_price) * 100
        reward2_pct = ((target2 - entry_price) / entry_price) * 100
        
        rr1 = reward1_pct / risk_pct if risk_pct > 0 else 0
        rr2 = reward2_pct / risk_pct if risk_pct > 0 else 0
        
        return {
            'entry': round(entry_price, 2),
            'stop_loss': round(sl, 2),
            'target1': round(target1, 2),
            'target2': round(target2, 2),
            'risk_pct': round(risk_pct, 2),
            'reward1_pct': round(reward1_pct, 2),
            'reward2_pct': round(reward2_pct, 2),
            'rr1': round(rr1, 2),
            'rr2': round(rr2, 2),
            'atr': round(atr, 2),
            'max_risk_pct': config.risk_profile['max_risk_pct']
        }


class PlaybookGenerator:
    """Generates execution playbooks for each setup"""
    
    @staticmethod
    def generate_playbook(ticker: str, setup_type: SetupType, score_data: Dict, risk_data: Dict, regime_info: Dict) -> Dict:
        """
        Generate comprehensive trade playbook
        """
        config = SETUP_CONFIGS.get(setup_type, None)
        
        playbook = {
            'ticker': ticker,
            'setup_type': setup_type.value,
            'confidence': PlaybookGenerator._calculate_confidence(score_data, regime_info),
            'why_selected': PlaybookGenerator._explain_selection(setup_type, score_data, regime_info),
            'entry_plan': PlaybookGenerator._create_entry_plan(setup_type, risk_data),
            'what_to_watch': PlaybookGenerator._what_to_watch(setup_type, config),
            'invalidation_rules': PlaybookGenerator._invalidation_rules(setup_type, risk_data),
            'risk_comment': PlaybookGenerator._risk_commentary(setup_type, risk_data),
            'time_decay': config.time_decay_days if config else 3,
            'probability_tier': PlaybookGenerator._probability_tier(score_data['total_score'], score_data['max_score'])
        }
        
        return playbook
    
    @staticmethod
    def _calculate_confidence(score_data: Dict, regime_info: Dict) -> int:
        """Calculate 0-100 confidence score"""
        score_ratio = score_data['total_score'] / score_data['max_score']
        regime_boost = regime_info['score'] / 100
        
        confidence = int((score_ratio * 0.7 + regime_boost * 0.3) * 100)
        return min(100, max(0, confidence))
    
    @staticmethod
    def _explain_selection(setup_type: SetupType, score_data: Dict, regime_info: Dict) -> str:
        """Explain why this setup was selected"""
        base_explanations = {
            SetupType.BREAKOUT: f"Volatility contraction with volume expansion signaling potential breakout. Score: {score_data['total_score']}/{score_data['max_score']}",
            SetupType.PULLBACK: f"Healthy pullback in strong trend with volume drying up. Score: {score_data['total_score']}/{score_data['max_score']}",
            SetupType.MOMENTUM: f"Strong relative strength with rising momentum. Score: {score_data['total_score']}/{score_data['max_score']}"
        }
        
        explanation = base_explanations.get(setup_type, "Technical setup identified")
        explanation += f"\n{regime_info['message']}"
        
        return explanation
    
    @staticmethod
    def _create_entry_plan(setup_type: SetupType, risk_data: Dict) -> Dict:
        """Create detailed entry plan"""
        plans = {
            SetupType.BREAKOUT: {
                'primary': f"Buy above ₹{risk_data['entry']} with volume > 1.5x on breakout candle",
                'alternate': f"If intraday pullback to breakout level with volume support",
                'avoid': "Gap up > 2%, late-day breakout, or breakout into resistance"
            },
            SetupType.PULLBACK: {
                'primary': f"Buy on bounce above ₹{risk_data['entry']} with volume expansion",
                'alternate': f"Enter on reversal candle at EMA20 support",
                'avoid': "Continued breakdown below EMA20, volume spike on decline"
            },
            SetupType.MOMENTUM: {
                'primary': f"Buy on continuation above ₹{risk_data['entry']} with strong candle",
                'alternate': f"Scale in on shallow pullback to EMA10/20",
                'avoid': "Parabolic move > 3 ATR in 2 days, bearish reversal candle"
            }
        }
        
        return plans.get(setup_type, {
            'primary': f"Buy at ₹{risk_data['entry']}",
            'alternate': "Watch for confirmation",
            'avoid': "Adverse market conditions"
        })
    
    @staticmethod
    def _what_to_watch(setup_type: SetupType, config: SetupConfig) -> List[str]:
        """What to watch tomorrow"""
        watch_items = {
            SetupType.BREAKOUT: [
                "Volume expansion confirmation",
                "Price holding above breakout level",
                "No gap up > 2%",
                "Sector strength continuation",
                "Index supporting move"
            ],
            SetupType.PULLBACK: [
                "Bounce candle with volume",
                "Holding above EMA20",
                "RSI turning up from 40-50 zone",
                "Weekly trend still intact",
                "No breakdown below swing low"
            ],
            SetupType.MOMENTUM: [
                "Continuation of strong price action",
                "Volume staying elevated",
                "RSI maintaining 60-70 zone",
                "No bearish reversal patterns",
                "Relative strength vs benchmark"
            ]
        }
        
        return watch_items.get(setup_type, ["Monitor price action", "Watch volume", "Check market regime"])
    
    @staticmethod
    def _invalidation_rules(setup_type: SetupType, risk_data: Dict) -> List[str]:
        """Setup invalidation rules"""
        rules = {
            SetupType.BREAKOUT: [
                f"Price closes below breakout level (₹{risk_data['entry']})",
                "Volume dries up on breakout attempt",
                "Immediate rejection at resistance",
                "Market regime flips to Risk-Off",
                f"Stop loss hit at ₹{risk_data['stop_loss']}"
            ],
            SetupType.PULLBACK: [
                "Breaks below EMA50",
                "Closes below swing low",
                "Bearish engulfing candle",
                "Volume spikes on decline",
                f"Stop loss hit at ₹{risk_data['stop_loss']}"
            ],
            SetupType.MOMENTUM: [
                "Closes below EMA20",
                "Bearish reversal candle",
                "RSI drops below 50",
                "Relative strength turns negative",
                f"Stop loss hit at ₹{risk_data['stop_loss']}"
            ]
        }
        
        return rules.get(setup_type, [f"Stop loss at ₹{risk_data['stop_loss']}"])
    
    @staticmethod
    def _risk_commentary(setup_type: SetupType, risk_data: Dict) -> str:
        """Risk management commentary"""
        if risk_data['risk_pct'] > 5:
            return f"⚠️ High risk setup ({risk_data['risk_pct']}%). Consider reducing position size."
        elif risk_data['risk_pct'] > 3:
            return f"Moderate risk ({risk_data['risk_pct']}%). Standard position sizing."
        else:
            return f"Low risk ({risk_data['risk_pct']}%). Favorable risk/reward setup."
    
    @staticmethod
    def _probability_tier(score: float, max_score: float) -> str:
        """Assign probability tier"""
        ratio = score / max_score
        
        if ratio >= 0.8:
            return "High Probability (70%+ historical)"
        elif ratio >= 0.6:
            return "Medium Probability (55-70%)"
        else:
            return "Tactical Setup (40-55%)"


class ActiveTradeEvaluator:
    """Daily evaluation of active trades"""
    
    @staticmethod
    def evaluate_trade(ticker: str, entry_price: float, stop_loss: float, target: float, 
                      setup_type: SetupType, entry_date: datetime) -> Dict:
        """
        Comprehensive daily evaluation of active trade
        """
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
            
            # Calculate P&L
            pnl_pct = ((curr_price - entry_price) / entry_price) * 100
            
            # Health check factors
            health_factors = {
                'trend': ActiveTradeEvaluator._check_trend(df, setup_type),
                'volume': ActiveTradeEvaluator._check_volume(volume),
                'regime': ActiveTradeEvaluator._check_regime(),
                'candle': ActiveTradeEvaluator._check_candle(df),
                'structure': ActiveTradeEvaluator._check_structure(df, setup_type)
            }
            
            # Calculate health score (0-100)
            health_score = sum(v for v in health_factors.values()) / len(health_factors)
            
            # Determine status
            if health_score >= 70:
                status = "Strong"
                status_color = "green"
            elif health_score >= 50:
                status = "Warning"
                status_color = "orange"
            else:
                status = "Weak"
                status_color = "red"
            
            # Action recommendation
            action = ActiveTradeEvaluator._generate_action(
                health_score, pnl_pct, days_held, setup_type, curr_price, stop_loss, target
            )
            
            # Suggested trailing SL
            atr = ta.atr(high, low, close, 14).iloc[-1]
            new_sl = ActiveTradeEvaluator._calculate_trailing_sl(
                entry_price, curr_price, stop_loss, pnl_pct, atr, setup_type
            )
            
            return {
                'status': status,
                'status_color': status_color,
                'health_score': round(health_score, 1),
                'health_factors': health_factors,
                'pnl_pct': round(pnl_pct, 2),
                'days_held': days_held,
                'current_price': round(curr_price, 2),
                'suggested_sl': round(new_sl, 2),
                'action': action,
                'phase': ActiveTradeEvaluator._determine_phase(days_held, pnl_pct)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def _check_trend(df: pd.DataFrame, setup_type: SetupType) -> float:
        """Check trend health (0-100)"""
        close = df['Close'].squeeze()
        ema20 = ta.ema(close, 20)
        ema50 = ta.ema(close, 50)
        
        curr_price = close.iloc[-1]
        
        if setup_type == SetupType.MOMENTUM:
            # Momentum needs strong trend
            if curr_price > ema20.iloc[-1] > ema50.iloc[-1]:
                return 100
            elif curr_price > ema20.iloc[-1]:
                return 70
            else:
                return 30
        else:
            # Others just need price above EMA20
            if curr_price > ema20.iloc[-1]:
                return 100
            else:
                return 40
    
    @staticmethod
    def _check_volume(volume: pd.Series) -> float:
        """Check volume health (0-100)"""
        avg_volume = volume.tail(20).mean()
        curr_volume = volume.iloc[-1]
        
        ratio = curr_volume / avg_volume if avg_volume > 0 else 1
        
        if ratio > 1.5:
            return 100
        elif ratio > 1.0:
            return 80
        elif ratio > 0.7:
            return 60
        else:
            return 30
    
    @staticmethod
    def _check_regime() -> float:
        """Check market regime (0-100)"""
        regime_info = MarketRegimeEngine.detect_regime()
        return regime_info['score']
    
    @staticmethod
    def _check_candle(df: pd.DataFrame) -> float:
        """Check latest candle quality (0-100)"""
        close = df['Close'].squeeze()
        open_price = df['Open'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        
        # Latest candle
        body = close.iloc[-1] - open_price.iloc[-1]
        total_range = high.iloc[-1] - low.iloc[-1]
        
        # Bullish candle
        if body > 0:
            body_ratio = body / total_range if total_range > 0 else 0
            return min(100, body_ratio * 120)
        else:
            # Bearish candle
            return 30
    
    @staticmethod
    def _check_structure(df: pd.DataFrame, setup_type: SetupType) -> float:
        """Check price structure (0-100)"""
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        
        # Check for higher highs / higher lows
        recent_highs = high.tail(10)
        recent_lows = low.tail(10)
        
        hh = recent_highs.iloc[-1] > recent_highs.iloc[-5]
        hl = recent_lows.iloc[-1] > recent_lows.iloc[-5]
        
        if hh and hl:
            return 100
        elif hh or hl:
            return 70
        else:
            return 40
    
    @staticmethod
    def _generate_action(health_score: float, pnl_pct: float, days_held: int, 
                        setup_type: SetupType, curr_price: float, sl: float, target: float) -> str:
        """Generate action recommendation"""
        
        # Critical checks first
        if curr_price <= sl:
            return "🚨 EXIT IMMEDIATELY - Stop loss hit"
        
        if health_score < 40:
            return "🔴 EXIT - Structure deteriorating"
        
        # Time-based exits
        config = SETUP_CONFIGS.get(setup_type)
        max_days = config.time_decay_days if config else 5
        
        if days_held > max_days and pnl_pct < 1:
            return f"⏰ EXIT - {days_held} days with no progress"
        
        # Profit-based actions
        if pnl_pct > 10:
            return "✅ TRAIL STOP - Excellent profit, protect gains"
        elif pnl_pct > 5:
            return "🟢 MOVE SL TO BE - Lock in breakeven"
        elif pnl_pct > 3 and health_score > 70:
            return "✓ HOLD - Setup developing well"
        elif pnl_pct < -2:
            return "⚠️ CAUTION - Near stop loss, monitor closely"
        elif health_score < 50:
            return "⚠️ TIGHTEN SL - Weakening structure"
        else:
            return "⚪ HOLD - Monitor development"
    
    @staticmethod
    def _calculate_trailing_sl(entry: float, current: float, original_sl: float, 
                               pnl_pct: float, atr: float, setup_type: SetupType) -> float:
        """Calculate dynamic trailing stop"""
        
        if pnl_pct < 3:
            return original_sl
        
        # Trail based on setup type
        if setup_type == SetupType.MOMENTUM:
            # Aggressive trailing for momentum
            return current - (atr * 1.0)
        elif setup_type == SetupType.BREAKOUT:
            # Medium trailing
            return current - (atr * 1.5)
        else:  # PULLBACK
            # Conservative trailing
            return max(entry, current - (atr * 1.2))
    
    @staticmethod
    def _determine_phase(days_held: int, pnl_pct: float) -> str:
        """Determine trade lifecycle phase"""
        if days_held <= 2:
            return "Entry Phase"
        elif days_held <= 4 and pnl_pct > 0:
            return "Confirmation Phase"
        elif pnl_pct > 5:
            return "Expansion Phase"
        elif pnl_pct > 8:
            return "Resistance Phase"
        elif pnl_pct < 0:
            return "Risk Phase"
        else:
            return "Development Phase"


# Export main classes
__all__ = [
    'SetupType', 'RegimeType', 'SetupConfig', 'SETUP_CONFIGS',
    'MarketRegimeEngine', 'SetupClassifier', 'SetupScorer',
    'RiskCalculator', 'PlaybookGenerator', 'ActiveTradeEvaluator'
]
