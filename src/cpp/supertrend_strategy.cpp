#include "supertrend_strategy.h"
#include <vector>
#include <cmath>
#include <algorithm> // For std::max
#include <numeric>   // For std::accumulate (if needed, not directly used in your ATR calc)
#include "data_types.h" // Added: Include for Candle and TradeResult structs

using namespace std;

// Calculate ATR (Average True Range)
double calculate_atr(const vector<Candle>& candles, int current_index, int period) {
    if (current_index < period) return 0.0; // Not enough data for full ATR period

    double sum_tr = 0.0;
    // Calculate sum of True Ranges for the current period
    for (int i = current_index - period + 1; i <= current_index; ++i) {
        double high = candles[i].high;
        double low = candles[i].low;
        double prev_close = (i > 0) ? candles[i - 1].close : candles[0].close; // Handle first candle edge case

        double tr1 = high - low;
        double tr2 = abs(high - prev_close);
        double tr3 = abs(low - prev_close);

        double true_range = max({tr1, tr2, tr3});
        sum_tr += true_range;
    }
    return sum_tr / period; // Simple Average True Range for this window
}

TradeResult run_supertrend_strategy(const vector<Candle>& candles, double profit_threshold, int period, double multiplier) {
    if (candles.size() < period) {
        return {0.0, 0.0, 0, vector<int>(candles.size(), 0)}; // Not enough data
    }

    vector<int> supertrend_positions(candles.size(), 0); // Renamed from 'signals' for consistency
    vector<double> basic_upper_band(candles.size(), 0.0);
    vector<double> basic_lower_band(candles.size(), 0.0);
    vector<double> final_upper_band(candles.size(), 0.0); // Renamed for clarity
    vector<double> final_lower_band(candles.size(), 0.0); // Renamed for clarity
    vector<bool> trend_up(candles.size(), true); // True if trend is up, false if down

    int profitable_trades = 0;
    double total_return = 0.0;
    int total_trades = 0;

    bool in_position = false; // True if currently holding a long position
    double entry_price = 0.0;

    // Initialize the first 'period' values for bands and trend
    for (int i = 0; i < period; ++i) {
        supertrend_positions[i] = 0; // No signals for initial period
        basic_upper_band[i] = (candles[i].high + candles[i].low) / 2.0 + multiplier * calculate_atr(candles, i, period);
        basic_lower_band[i] = (candles[i].high + candles[i].low) / 2.0 - multiplier * calculate_atr(candles, i, period);
        final_upper_band[i] = basic_upper_band[i];
        final_lower_band[i] = basic_lower_band[i];
        trend_up[i] = true; // Assume initial trend is up
    }


    for (size_t i = period; i < candles.size(); ++i) {
        double atr = calculate_atr(candles, i, period);
        double hl2 = (candles[i].high + candles[i].low) / 2.0;

        basic_upper_band[i] = hl2 + multiplier * atr;
        basic_lower_band[i] = hl2 - multiplier * atr;

        // Calculate final upper and lower bands, enforcing continuity
        if (basic_upper_band[i] < final_upper_band[i - 1] || candles[i - 1].close > final_upper_band[i - 1]) {
            final_upper_band[i] = basic_upper_band[i];
        } else {
            final_upper_band[i] = final_upper_band[i - 1];
        }

        if (basic_lower_band[i] > final_lower_band[i - 1] || candles[i - 1].close < final_lower_band[i - 1]) {
            final_lower_band[i] = basic_lower_band[i];
        } else {
            final_lower_band[i] = final_lower_band[i - 1];
        }

        // Determine trend direction
        if (trend_up[i - 1]) { // If previous trend was up
            if (candles[i].close < final_upper_band[i]) { // Close below upper band, trend might reverse down
                trend_up[i] = false;
            } else { // Close above or equal to upper band, trend remains up
                trend_up[i] = true;
            }
        } else { // If previous trend was down
            if (candles[i].close > final_lower_band[i]) { // Close above lower band, trend might reverse up
                trend_up[i] = true;
            } else { // Close below or equal to lower band, trend remains down
                trend_up[i] = false;
            }
        }


        // Generate signals based on Supertrend flips
        if (!in_position && trend_up[i] && !trend_up[i - 1]) {
            // Buy signal: Trend flipped from down to up
            entry_price = candles[i].close;
            in_position = true;
            supertrend_positions[i] = 1; // Mark as buy
        } else if (in_position && !trend_up[i] && trend_up[i - 1]) {
            // Sell signal: Trend flipped from up to down
            double exit_price = candles[i].close;
            double ret = (exit_price - entry_price) / entry_price;
            total_return += ret;
            if (ret > profit_threshold)
                profitable_trades++;
            total_trades++;
            in_position = false;
            supertrend_positions[i] = -1; // Mark as sell
            entry_price = 0.0; // Reset entry price
        } else {
            supertrend_positions[i] = 0; // No action
        }
    }

    // Force-close any open position at the end of the data
    if (in_position) {
        double exit_price = candles.back().close;
        double ret = (exit_price - entry_price) / entry_price;
        total_return += ret;
        if (ret > profit_threshold)
            profitable_trades++;
        total_trades++;
    }

    double success_rate = total_trades > 0 ? (double)profitable_trades / total_trades * 100 : 0;
    double avg_return = total_trades > 0 ? (total_return / total_trades) * 100 : 0;

    return {success_rate, avg_return, total_trades, supertrend_positions};
}
// ... (existing run_supertrend_strategy code) ...

void calculate_supertrend_series(const std::vector<Candle>& candles, std::vector<double>& supertrend_line_out, std::vector<bool>& trend_up_series_out, int period, double multiplier) {
    supertrend_line_out.assign(candles.size(), 0.0);
    trend_up_series_out.assign(candles.size(), true); // Initialize as true

    if (candles.size() < period) {
        return;
    }

    std::vector<double> basic_upper_band(candles.size(), 0.0);
    std::vector<double> basic_lower_band(candles.size(), 0.0);
    std::vector<double> final_upper_band(candles.size(), 0.0);
    std::vector<double> final_lower_band(candles.size(), 0.0);

    // Calculate for all points
    for (size_t i = 0; i < candles.size(); ++i) { // Loop through all candles to populate bands
        double atr_val;
        if (i < period) { // Handle warm-up period for ATR
            atr_val = 0.0; // Or a very small default
        } else {
            atr_val = calculate_atr(candles, i, period);
        }

        double hl2 = (candles[i].high + candles[i].low) / 2.0;

        basic_upper_band[i] = hl2 + multiplier * atr_val;
        basic_lower_band[i] = hl2 - multiplier * atr_val;

        if (i == 0) {
            final_upper_band[i] = basic_upper_band[i];
            final_lower_band[i] = basic_lower_band[i];
        } else {
            final_upper_band[i] = basic_upper_band[i];
            if (basic_upper_band[i] > final_upper_band[i-1] && candles[i-1].close < final_upper_band[i-1]) {
                // Current basic upper band is higher, and previous close was below previous upper band (trend not confirmed up yet)
                final_upper_band[i] = basic_upper_band[i];
            } else if (basic_upper_band[i] < final_upper_band[i-1] && candles[i-1].close > final_upper_band[i-1]) {
                 // Current basic upper band is lower, but price broke above previous upper band (trend might be up)
                 final_upper_band[i] = basic_upper_band[i]; // Use current basic upper
            } else {
                final_upper_band[i] = final_upper_band[i-1]; // Retain previous high point
            }
            if (trend_up_series_out[i-1] && candles[i].close <= final_upper_band[i-1]) { // if previous trend was up but current close drops
                final_upper_band[i] = std::min(basic_upper_band[i], final_upper_band[i-1]);
            }

            final_lower_band[i] = basic_lower_band[i];
            if (basic_lower_band[i] < final_lower_band[i-1] && candles[i-1].close > final_lower_band[i-1]) {
                final_lower_band[i] = basic_lower_band[i];
            } else if (basic_lower_band[i] > final_lower_band[i-1] && candles[i-1].close < final_lower_band[i-1]) {
                final_lower_band[i] = basic_lower_band[i];
            } else {
                final_lower_band[i] = final_lower_band[i-1];
            }
            if (!trend_up_series_out[i-1] && candles[i].close >= final_lower_band[i-1]) { // if previous trend was down but current close rises
                final_lower_band[i] = std::max(basic_lower_band[i], final_lower_band[i-1]);
            }
        }

        // Determine current trend based on previous trend and current close
        if (i > 0) {
            if (trend_up_series_out[i - 1]) { // If previous trend was up
                if (candles[i].close < final_upper_band[i]) { // Close below upper band, trend might reverse down
                    trend_up_series_out[i] = false;
                } else { // Close above or equal to upper band, trend remains up
                    trend_up_series_out[i] = true;
                }
            } else { // If previous trend was down
                if (candles[i].close > final_lower_band[i]) { // Close above lower band, trend might reverse up
                    trend_up_series_out[i] = true;
                } else { // Close below or equal to lower band, trend remains down
                    trend_up_series_out[i] = false;
                }
            }
        }

        // Calculate Supertrend line
        if (trend_up_series_out[i]) {
            supertrend_line_out[i] = final_lower_band[i];
        } else {
            supertrend_line_out[i] = final_upper_band[i];
        }
    }
}