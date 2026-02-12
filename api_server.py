"""
Flask API Backend for NSE Trading Dashboard
Wraps trading_engine.py and provides REST endpoints
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
import sys
import os

# Import trading engine
sys.path.append(os.path.dirname(__file__))
from trading_engine import (
    SetupType, MarketRegimeEngine, SetupClassifier,
    SetupScorer, RiskCalculator, PlaybookGenerator,
    ActiveTradeEvaluator
)

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# NSE stocks universe
NSE_STOCKS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
    'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'SUNPHARMA.NS',
    'TITAN.NS', 'ULTRACEMCO.NS', 'BAJFINANCE.NS', 'NESTLEIND.NS', 'HCLTECH.NS',
    'WIPRO.NS', 'ADANIENT.NS', 'ONGC.NS', 'TATAMOTORS.NS', 'POWERGRID.NS',
    'NTPC.NS', 'TATASTEEL.NS', 'M&M.NS', 'JSWSTEEL.NS', 'INDUSINDBK.NS'
]


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.route('/api/market-regime', methods=['GET'])
def get_market_regime():
    """Get current market regime"""
    try:
        regime = MarketRegimeEngine.detect_regime(ticker="^NSEI")
        
        return jsonify({
            'success': True,
            'data': {
                'type': regime['type'].value,
                'score': regime['score'],
                'color': regime['color'],
                'breadth': regime['breadth'],
                'message': regime['message'],
                'details': regime['details']
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/scan', methods=['POST'])
def scan_opportunities():
    """
    Scan for trading opportunities
    Body: { "setup_type": "momentum"|"breakout"|"pullback", "limit": 15 }
    """
    try:
        data = request.get_json()
        setup_type_str = data.get('setup_type', 'momentum').upper()
        limit = data.get('limit', 15)
        
        # Map string to enum
        setup_type = getattr(SetupType, setup_type_str, SetupType.MOMENTUM)
        
        opportunities = []
        
        for ticker in NSE_STOCKS[:20]:  # Limit to first 20 for speed
            try:
                # Classify setup
                classification = SetupClassifier.classify(ticker)
                
                # Filter by requested setup type
                if classification['setup_type'] != setup_type:
                    continue
                
                # Score the opportunity
                score = SetupScorer.score_setup(ticker, setup_type)
                
                if score['total_score'] < 60:  # Minimum quality threshold
                    continue
                
                # Calculate risk/reward
                risk_calc = RiskCalculator.calculate(ticker, setup_type, capital=100000)
                
                opportunities.append({
                    'symbol': ticker.replace('.NS', ''),
                    'setup_type': setup_type.value,
                    'setup_score': round(score['total_score'], 1),
                    'price': round(score.get('current_price', 0), 2),
                    'entry': round(risk_calc.get('entry', 0), 2),
                    'stop_loss': round(risk_calc.get('stop_loss', 0), 2),
                    'target': round(risk_calc.get('target', 0), 2),
                    'risk_reward': risk_calc.get('risk_reward_ratio', 'N/A'),
                    'position_size': risk_calc.get('shares', 0),
                    'rsi': round(score.get('rsi', 50), 1),
                    'volume_ratio': round(score.get('volume_ratio', 1), 2),
                    'atr': round(score.get('atr', 0), 2),
                    'breakdown': score.get('breakdown', {})
                })
                
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                continue
        
        # Sort by score
        opportunities.sort(key=lambda x: x['setup_score'], reverse=True)
        
        return jsonify({
            'success': True,
            'data': opportunities[:limit],
            'count': len(opportunities),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/playbook/<ticker>', methods=['GET'])
def get_playbook(ticker):
    """
    Get detailed trade playbook for a ticker
    """
    try:
        # Add .NS suffix if not present
        if not ticker.endswith('.NS'):
            ticker = f"{ticker}.NS"
        
        # Classify
        classification = SetupClassifier.classify(ticker)
        setup_type = classification['setup_type']
        
        # Generate playbook
        playbook = PlaybookGenerator.generate(ticker, setup_type)
        
        if 'error' in playbook:
            return jsonify({
                'success': False,
                'error': playbook['error']
            }), 400
        
        return jsonify({
            'success': True,
            'data': {
                'symbol': ticker.replace('.NS', ''),
                'setup_type': setup_type.value,
                'playbook': playbook
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/evaluate-trade', methods=['POST'])
def evaluate_trade():
    """
    Evaluate an active trade
    Body: {
        "ticker": "RELIANCE",
        "entry_price": 2500,
        "stop_loss": 2450,
        "target": 2650,
        "setup_type": "momentum",
        "entry_date": "2024-02-01"
    }
    """
    try:
        data = request.get_json()
        
        ticker = data['ticker']
        if not ticker.endswith('.NS'):
            ticker = f"{ticker}.NS"
        
        entry_price = float(data['entry_price'])
        stop_loss = float(data['stop_loss'])
        target = float(data['target'])
        setup_type_str = data['setup_type'].upper()
        entry_date = datetime.fromisoformat(data['entry_date'])
        
        setup_type = getattr(SetupType, setup_type_str, SetupType.MOMENTUM)
        
        evaluation = ActiveTradeEvaluator.evaluate_trade(
            ticker=ticker,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            setup_type=setup_type,
            entry_date=entry_date
        )
        
        if 'error' in evaluation:
            return jsonify({
                'success': False,
                'error': evaluation['error']
            }), 400
        
        return jsonify({
            'success': True,
            'data': {
                'symbol': ticker.replace('.NS', ''),
                'evaluation': evaluation
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/batch-evaluate', methods=['POST'])
def batch_evaluate():
    """
    Evaluate multiple active trades
    Body: {
        "trades": [
            {
                "ticker": "RELIANCE",
                "entry_price": 2500,
                "stop_loss": 2450,
                "target": 2650,
                "setup_type": "momentum",
                "entry_date": "2024-02-01"
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        trades = data.get('trades', [])
        
        results = []
        
        for trade in trades:
            try:
                ticker = trade['ticker']
                if not ticker.endswith('.NS'):
                    ticker = f"{ticker}.NS"
                
                entry_price = float(trade['entry_price'])
                stop_loss = float(trade['stop_loss'])
                target = float(trade['target'])
                setup_type_str = trade['setup_type'].upper()
                entry_date = datetime.fromisoformat(trade['entry_date'])
                
                setup_type = getattr(SetupType, setup_type_str, SetupType.MOMENTUM)
                
                evaluation = ActiveTradeEvaluator.evaluate_trade(
                    ticker=ticker,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target=target,
                    setup_type=setup_type,
                    entry_date=entry_date
                )
                
                results.append({
                    'symbol': ticker.replace('.NS', ''),
                    'evaluation': evaluation,
                    'success': 'error' not in evaluation
                })
                
            except Exception as e:
                results.append({
                    'symbol': trade.get('ticker', 'UNKNOWN'),
                    'error': str(e),
                    'success': False
                })
        
        return jsonify({
            'success': True,
            'data': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/quick-scan', methods=['GET'])
def quick_scan():
    """
    Quick scan of all three setups - returns top 5 for each
    """
    try:
        all_opportunities = {
            'MOMENTUM': [],
            'BREAKOUT': [],
            'PULLBACK': []
        }
        
        for ticker in NSE_STOCKS[:15]:  # Quick scan on subset
            try:
                classification = SetupClassifier.classify(ticker)
                setup_type = classification['setup_type']
                
                if setup_type == SetupType.UNKNOWN:
                    continue
                
                score = SetupScorer.score_setup(ticker, setup_type)
                
                if score['total_score'] >= 65:
                    all_opportunities[setup_type.value].append({
                        'symbol': ticker.replace('.NS', ''),
                        'score': round(score['total_score'], 1),
                        'price': round(score.get('current_price', 0), 2),
                        'rsi': round(score.get('rsi', 50), 1)
                    })
                    
            except Exception as e:
                continue
        
        # Sort and limit each category
        for setup in all_opportunities:
            all_opportunities[setup].sort(key=lambda x: x['score'], reverse=True)
            all_opportunities[setup] = all_opportunities[setup][:5]
        
        return jsonify({
            'success': True,
            'data': all_opportunities,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("🚀 NSE Trading API Server Starting...")
    print("📊 Trading Engine: Loaded")
    print("🌐 Endpoints Available:")
    print("   GET  /health")
    print("   GET  /api/market-regime")
    print("   POST /api/scan")
    print("   GET  /api/playbook/<ticker>")
    print("   POST /api/evaluate-trade")
    print("   POST /api/batch-evaluate")
    print("   GET  /api/quick-scan")
    print("\n✅ Server running on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
