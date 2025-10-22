import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class SignalGenerator:

    def __init__(self, copula_model, lookback_days=21):
        self.copula_model = copula_model
        self.lookback_days = lookback_days
        self.current_signals ={}
        self.signal_history = []

    def calculate_normalised_returns(self, current_price, historical_prices):
        
        if len(historical_prices) < 5:
            return 0.0
        
        recent_prices = historical_prices[-self.lookback_days:]
        recent_returns = np.diff(np.log(recent_prices))

        if len(recent_returns) == 0:
            return 0.0
        
        recent_mean = np.mean(recent_returns)
        recent_std = np.std(recent_returns)

        if recent_std == 0:
            return 0.0
        
        current_return = (current_price - recent_prices[-1])/recent_prices[-1]
        normalized_return = (current_return - recent_mean)/recent_std

        return normalized_return
    
    def generate_signal(self, pair_data, current_prices_df):
        
        pair_name = f"{pair_data['ETF1']}_{pair_data['ETF2']}"
        model = pair_data.get("copula_model")
        
        if not model:
            print(f"No copula model found for {pair_name}")
            return None
        
        try:
            # Getting current and historical prices
            etf1_prices = current_prices_df[pair_data["ETF1"]].dropna()
            etf2_prices = current_prices_df[pair_data["ETF2"]].dropna()

            if len(etf1_prices) < self.lookback_days or len(etf2_prices) < self.lookback_days:
                return None
            
            current_prices_1 = etf1_prices.iloc[-1]
            current_prices_2 = etf2_prices.iloc[-1]

            # Calculating normalised returns
            norm_return_1 = self.calculate_normalised_returns(current_prices_1, etf1_prices.values)
            norm_return_2 = self.calculate_normalised_returns(current_prices_2, etf2_prices.values)

            # Transforming to uniform space using fitted marginals
            u = model["marginal_1"]["cdf"](norm_return_1)
            v = model["marginal_2"]["cdf"](norm_return_2)

            u = np.clip(u, 1e-6, 1-1e-6)
            v = np.clip(u, 1e-6, 1-1e-6)

            # Calculating conditional probabilities in both directions
            cond_prob_1_given_2 = model["copula"].partial_derivative(u,v)
            cond_prob_2_given_1 = model["copula"].partial_derivative(v,u)

            signal_info = self.interpret_signals(
                cond_prob_1_given_2, cond_prob_2_given_1,
                pair_data, model, norm_return_1, norm_return_2
            )

            if signal_info:
                signal_info.update({
                    "piar_name": pair_name,
                    "timestamp": datetime.now(),
                    "current_price_1": current_prices_1,
                    "current_price_2": current_prices_2,
                    "norm_return_1": norm_return_1,
                    "cond_prob_1_given_2": cond_prob_1_given_2,
                    "cond_prob_2_given_1": cond_prob_2_given_1
                })

                self.current_signals[pair_name] = signal_info
                self.signal_history.append(signal_info)

            return signal_info

        except Exception as e:
            print(f"Error generating signal for {pair_name}: {e}") 
            return None


    def _interpret_signals(self, prob1g2, prob2g1, pair_data, model, ret_1, ret_2):
        
        etf1, etf2 = pair_data["ETF1"], pair_data["ETF2"]
        hedge_ratio = pair_data.get("hedge_ratio", 1.0)

        if prob1g2 > 0.95 and prob2g1 <0.05:
            # This implies ETF1 is exremely cheap given ETF2's state meaning ETF2 is extremely expensive given ETF1's state
            signal_type = "BUY_1_SELL_2"
            direction = 1
            strength = min(prob1g2 - 0.95, 0.05 - prob2g1) * 20
            action = f"BUY {etf1}, SELL {etf2}"
            rationale = f"{etf1} cheap and (P={prob1g2:.3f}) and {etf2} expensive (P={prob2g1:.3f}) given each other"

        elif prob1g2 < 0.05 and prob2g1 > 0.95:
            # ETF1 is extremely expensive given ETF2's state meaning ETF2 is extremely cheap given ETF1's state
            signal_type = "SELL_1_BUY_2"
            direction = -1
            strength = min(0.05 - prob1g2, prob2g1 - 0.95) * 20
            action = f"SELL {etf1}, BUY {etf2}"
            rationale = f"{etf1} expensive (P={prob1g2:.3f}) and {etf2} cheap (P={prob2g1:.3f}) given each other"

        else:    
            signal_type = "HOLD"
            direction = 0
            strength = 0.0
            action = "HOLD"
            rationale = f"Mixed signals: {etf1} P={prob1g2:.3f}, {etf2} P={prob2g1:.3f}"

        # Adjusting for tail risk
        tail_risk_adjustment = 1 - model["lower_tail_deependence"]    
        adjusted_strength = strength * tail_risk_adjustment

        return {
            "signal_type": signal_type,
            "direction": direction,
            "strength": adjusted_strength,
            "action": action,
            "rationale": rationale,
            "tail_dependence": model["lower_tail_dependece"],
            "copula_type": model["copula_type"],
            "hedge_ratio": hedge_ratio,
            "raw_strength": strength
        }
    
    def generate_batch_signals(self, selected_pairs, current_prices_df):

        signals = {}

        print("Generating trading signals...")
        print(selected_pairs)
        for i, pair_data in enumerate(selected_pairs):
            if i % 5 == 0:
                print(f"Processed {i}/{len(selected_pairs)} pairs...")
            
            # pair_data_dict = pair_data.to_dict()
            signal = self.generate_signal(pair_data, current_prices_df)
            if signal:
                signals[pair_data["pair_name"]] = signal

        self.current_signals = signals
        print(f"Generated {len(signals)} signals")        

    def get_active_signals(self, min_strength=0.3):

        return {k:v for k,v in self.current_signals.items()
                if v["strength"]>=min_strength and v["direction"] !=0}
    
    def calculate_position_size(self, signal, portfolio_value, risk_per_trade=0.01):

        if signal["direction"] ==0:
            return 0,0
        
        base_position = portfolio_value * risk_per_trade

        strength_multiplier = signal["strength"]
        adjusted_position = base_position * strength_multiplier

        hedge_ratio = signal.get("hedge_ratio", 1.0)

        if signal["signal_type"] == "BUY_1_SELL_2":
            position_1 = adjusted_position
            position_2 = -adjusted_position * hedge_ratio

        else:
            position_1 = -adjusted_position
            position_2 = adjusted_position * hedge_ratio

        return position_1, position_2

    def generate_trading_instruction(self, min_strength=0.3, portfolio_value=100000):
        
        active_signals = self.get_active_signals(min_strength)
        instructions = []

        for pair_name, signal in active_signals.items():
            etf1, etf2 = pair_name.split("_")
            position_1, position_2 = self.calculate_position_size(signal, portfolio_value)

            instruction = {
                "pair_name": pair_name,
                "signal_type": signal["signal_type"],
                "action": signal["action"],
                "strength": signal["strength"],
                "position_etf1": position_1,
                "position_etf2": position_2,
                "hedge_ratio": signal["hedge_ratio"],
                "rationale": signal["rationale"],
                "timestamp": signal["timestamp"]
            }
            instructions.append(instruction)

        return instructions

    def print_signals(self, min_strength=0.1):
        print("\n" + "="*60)
        print("Current Trading Signals")
        print("="*60)    

        active_count = 0
        for pair_name, signal in self.current_signals.items():
            if signal["strength"] >= min_strength and signal["direction"] != 0:
                active_count +=1
                print(f"\n{pair_name}:")
                print(f"Action: {signal["action"]}")
                print(f"Strength: {signal["strength"]:.2f} (Raw: {signal["raw_strength"]:.2f})")
                print(f"Signal: {signal["signal_type"]}")
                print(f"Rationale: {signal["rationale"]}")
                print(f"Tail Risk: {signal["tail_dependence"]:.3f}")
                print(f"Copula: {signal["copula_type"]}")

            if active_count == 0:
                print("\nNo active signals above strength threshold")    
            else:
                print(f"\nTotal active signals: {active_count}")

    def get_signal_summary(self):

        if not self.current_signals:
            return {}
        
        signals_df = pd.DataFrame(self.current_sginals.values())
        summary = {
            "total_signals": len(self.current_signals),
            "active_signals": len(self.get_active_signals(min_strength=0.3)),
            "avg_strength": signals_df["strength"].mean(),
            "bullish_signals": len(signals_df[signals_df["direction"] == 1]),
            "bearish_signals": len(signals_df[signals_df["direction"] == -1]),
            "avg_tail_risk": signals_df["tail_dependence"].mean()
        }
        return summary
    

def main():
    from copula_model import CopulaModel
    from pair_selection import PairSelector

    prices = pd.read_csv("data/processed/cleaned_prices.csv", index_col=0, parse_dates=True)
    returns = pd.read_csv("data/processed/log_returns.csv", index_col=0, parse_dates=True)
    selector = PairSelector(prices, returns)
    selected_pairs = selector.run_selection()
    # Check if copula models are present
    print(f"Selected {len(selected_pairs)} pairs")
    
    if len(selected_pairs) > 0:
        print(f"First pair keys: {selected_pairs[0].keys()}")
        print(f"Has copula_model: {'copula_model' in selected_pairs[0]}")
    
    copula_model = selector.get_copula_model()

    signal_generator = SignalGenerator(copula_model, lookback_days=20)

    recent_prices = prices.tail(50)
    signals = signal_generator.generate_batch_signals(selected_pairs, recent_prices)

    signal_generator.print_signals(min_strength=0.3)

    instructions = signal_generator.generate_trading_instruction(
        min_strength=0.3,
        portfolio_value=1000000
    )

    print(f"\nGenerated {len(instructions)} trading instructions")

if __name__ == "__main__":
    main()