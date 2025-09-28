#!/usr/bin/env python3

"""
Treasury Risk Dashboard - Demo Data Generator
For DEGA Hackathon - AI for DAO Treasury Management

This script generates realistic demo data for the dashboard
including portfolio balances, historical prices, and generates
sample forecasts to showcase the platform capabilities.
"""

import asyncio
import random
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List

# API Configuration
API_BASE = "http://localhost:8000"
SUPPORTED_CRYPTO = ["bitcoin", "ethereum", "usd-coin", "tether"]
SUPPORTED_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN"]

class DemoDataGenerator:
    """Generate realistic demo data for the Treasury Risk Dashboard."""

    def __init__(self):
        self.session = requests.Session()

    def check_health(self) -> bool:
        """Check if the API is healthy."""
        try:
            response = self.session.get(f"{API_BASE}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def create_demo_portfolio(self) -> Dict:
        """Create a diverse demo portfolio."""
        portfolio = {
            "bitcoin": {
                "balance": random.uniform(0.5, 2.0),
                "symbol": "bitcoin"
            },
            "ethereum": {
                "balance": random.uniform(5.0, 15.0),
                "symbol": "ethereum"
            },
            "usd-coin": {
                "balance": random.uniform(10000, 50000),
                "symbol": "usd-coin"
            },
            "AAPL": {
                "balance": random.uniform(10, 50),
                "symbol": "AAPL"
            },
            "MSFT": {
                "balance": random.uniform(5, 25),
                "symbol": "MSFT"
            }
        }
        return portfolio

    def add_portfolio_balances(self, portfolio: Dict) -> bool:
        """Add portfolio balances via API."""
        try:
            balances = []
            for asset, data in portfolio.items():
                balances.append({
                    "symbol": data["symbol"],
                    "balance": data["balance"],
                    "wallet_address": "demo_wallet"
                })

            response = self.session.post(
                f"{API_BASE}/api/v1/balances/update-balances",
                json={"balances": balances},
                timeout=30
            )

            return response.status_code == 200
        except Exception as e:
            print(f"❌ Failed to add portfolio balances: {e}")
            return False

    def generate_forecast(self, symbol: str) -> bool:
        """Generate a forecast for a given symbol."""
        try:
            payload = {
                "symbol": symbol,
                "horizon_days": 30,
                "models": ["arima", "prophet"],
                "include_explanation": True
            }

            response = self.session.post(
                f"{API_BASE}/api/v1/forecasts/generate",
                json=payload,
                timeout=120  # Forecasting can take time
            )

            if response.status_code == 200:
                print(f"✅ Generated forecast for {symbol}")
                return True
            else:
                print(f"⚠️  Forecast generation for {symbol} returned status {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Failed to generate forecast for {symbol}: {e}")
            return False

    def generate_hedge_suggestion(self, portfolio_weights: Dict) -> bool:
        """Generate hedge suggestions for the portfolio."""
        try:
            payload = {
                "portfolio_weights": portfolio_weights,
                "risk_tolerance": "medium",
                "constraints": {
                    "max_single_asset_weight": 0.4,
                    "min_single_asset_weight": 0.01
                }
            }

            response = self.session.post(
                f"{API_BASE}/api/v1/hedge/suggest",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                print("✅ Generated hedge suggestions")
                return True
            else:
                print(f"⚠️  Hedge suggestion returned status {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Failed to generate hedge suggestions: {e}")
            return False

    def trigger_data_ingestion(self) -> bool:
        """Trigger market data ingestion."""
        try:
            response = self.session.post(f"{API_BASE}/admin/ingest-data", timeout=60)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Failed to trigger data ingestion: {e}")
            return False

    def get_portfolio_weights(self, portfolio: Dict) -> Dict[str, float]:
        """Calculate portfolio weights from balances (simplified)."""
        # Simplified weight calculation for demo
        weights = {}
        total_weight = 0

        for asset, data in portfolio.items():
            # Assign reasonable weights based on typical portfolio
            if asset == "bitcoin":
                weight = 0.3
            elif asset == "ethereum":
                weight = 0.25
            elif asset == "usd-coin":
                weight = 0.2
            elif asset == "AAPL":
                weight = 0.15
            elif asset == "MSFT":
                weight = 0.1
            else:
                weight = 0.05

            weights[data["symbol"]] = weight
            total_weight += weight

        # Normalize weights
        for symbol in weights:
            weights[symbol] = weights[symbol] / total_weight

        return weights

def main():
    """Main demo script."""
    print("🎬 Treasury Risk Dashboard - Demo Data Generator")
    print("==============================================")
    print("🏆 Built for DEGA Hackathon: AI for DAO Treasury Management")
    print("")

    generator = DemoDataGenerator()

    # Check if API is healthy
    print("🔍 Checking API health...")
    if not generator.check_health():
        print("❌ API is not healthy. Please ensure the services are running.")
        print("💡 Run './start.sh' to start all services.")
        return

    print("✅ API is healthy!")
    print("")

    # Step 1: Trigger initial data ingestion
    print("📊 Step 1: Triggering market data ingestion...")
    if generator.trigger_data_ingestion():
        print("✅ Market data ingestion started")
    else:
        print("⚠️  Market data ingestion may have failed")

    # Wait for data ingestion
    print("⏳ Waiting for data ingestion to complete...")
    time.sleep(10)

    # Step 2: Create and add demo portfolio
    print("\n💼 Step 2: Creating demo portfolio...")
    portfolio = generator.create_demo_portfolio()

    print("📋 Demo Portfolio:")
    for asset, data in portfolio.items():
        print(f"   • {data['symbol']}: {data['balance']:.6f}")

    if generator.add_portfolio_balances(portfolio):
        print("✅ Portfolio balances added successfully")
    else:
        print("❌ Failed to add portfolio balances")
        return

    # Step 3: Generate forecasts for key assets
    print("\n🔮 Step 3: Generating AI forecasts...")
    forecast_symbols = ["bitcoin", "ethereum", "AAPL"]

    for symbol in forecast_symbols:
        print(f"🧠 Generating forecast for {symbol}...")
        success = generator.generate_forecast(symbol)
        if success:
            print(f"   ✅ Forecast completed for {symbol}")
        else:
            print(f"   ⚠️  Forecast may have failed for {symbol}")

        # Small delay between forecasts
        time.sleep(2)

    # Step 4: Generate hedge suggestions
    print("\n🛡️  Step 4: Generating hedge suggestions...")
    portfolio_weights = generator.get_portfolio_weights(portfolio)

    if generator.generate_hedge_suggestion(portfolio_weights):
        print("✅ Hedge suggestions generated")
    else:
        print("⚠️  Hedge suggestions may have failed")

    print("")
    print("🎉 Demo data generation complete!")
    print("===============================================")
    print("🌐 Visit http://localhost:3000 to explore the dashboard")
    print("")
    print("📊 What you can now demo:")
    print("1. 🔮 View AI-generated forecasts in the 'Forecasts' tab")
    print("2. 💼 Explore portfolio allocation in the 'Portfolio' tab")
    print("3. ⚠️  Analyze risk metrics and performance")
    print("4. 🛡️  Review hedge suggestions and simulations")
    print("5. 🧠 Read AI explanations for each analysis")
    print("")
    print("🏆 Perfect for demonstrating DAO Treasury Management!")
    print("💡 This showcases real ML models, not mocks!")

if __name__ == "__main__":
    main()