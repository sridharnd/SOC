#include "rsi_strategy.h"
#include <vector>
#include <cmath>
#include "data_types.h" // Added: Include for Candle and TradeResult structs

using namespace std;

double calculate_rsi(const vector<double> &closes, int current_index, int period)
{
    if (current_index < period)
        return 50.0; // not enough data

    double gain = 0.0, loss = 0.0;

    // Calculate initial gain and loss over the first 'period' candles
    // The loop should start from current_index - period + 1 to include 'period' candles
    // The change is from closes[i] to closes[i-1], so for 'period' candles, we need 'period + 1' close prices.
    // However, RSI calculation typically uses the *average* gain/loss over the period.
    // For simplicity and to match the provided logic, we'll calculate sum of changes for the first period.
    for (int i = current_index - period + 1; i <= current_index; ++i)
    {
        double change = closes[i] - closes[i - 1]; // Change from previous close
        if (change > 0)
            gain += change;
        else
            loss -= change; // loss is positive
    }

    // A small modification for robustness: avoid division by zero
    // If loss is zero, RS is infinite, RSI is 100.
    // If gain is zero, RS is zero, RSI is 0.
    double rs;
    if (loss == 0.0) { // No losses in the period
        rs = gain / 0.00000001; // Avoid division by zero, treat as very large
    } else if (gain == 0.0) { // No gains in the period
        rs = 0.0;
    } else {
        rs = gain / loss;
    }

    return 100.0 - (100.0 / (1.0 + rs));
}

TradeResult run_rsi_strategy(const vector<Candle> &candles, double profit_threshold)
{
    vector<double> closes;
    for (const auto &candle : candles)
        closes.push_back(candle.close);

    // Ensure there's enough data for initial RSI calculation (min 15 candles for period=14)
    if (closes.size() < 15) {
        return {0.0, 0.0, 0, vector<int>(closes.size(), 0)}; // Return empty result if not enough data
    }

    vector<int> rsi_positions(closes.size(), 0);

    int profitable_trades = 0;
    double total_return = 0.0;
    int total_trades = 0;

    bool was_above_60 = false;
    bool was_below_40 = false;

    double entry_price = 0.0;
    int entry_index = -1; // Keep track of the entry candle index
    enum Position
    {
        NONE,
        LONG,
        SHORT
    } state = NONE;

    // Start iteration from index 15 as per original code, assuming 14-period RSI needs 14 previous data points (index 0 to 13)
    // to calculate RSI for index 14. So starting loop from 15 means we're calculating RSI for candle at index 15.
    for (size_t i = 15; i < closes.size(); ++i)
    {
        double rsi = calculate_rsi(closes, i);
        // Ensure i-1 is a valid index for price_change calculation
        double price_change = (closes[i] - closes[i - 1]) / closes[i - 1];

        // Original signal encoding logic with 1% movement filter - retaining your logic
        // This part primarily sets the `rsi_positions` vector for visualization or later analysis,
        // but the `state` variable directly drives the trading logic.
        if (rsi > 60 && price_change > 0.05)
        {
            if (!was_above_60)
            {
                rsi_positions[i] = 1; // Buy signal
                was_above_60 = true;
            }
            else
            {
                rsi_positions[i] = 0;
            }
            was_below_40 = false;
        }
        else if (was_above_60 && rsi < 60)
        {
            rsi_positions[i] = -1; // Exit long
            was_above_60 = false;
        }
        else if (rsi < 40 && price_change < -0.05)
        {
            if (!was_below_40)
            {
                rsi_positions[i] = -2; // Short signal
                was_below_40 = true;
            }
            else
            {
                rsi_positions[i] = 0;
            }
            was_above_60 = false;
        }
        else if (was_below_40 && rsi > 40)
        {
            rsi_positions[i] = 2; // Exit short
            was_below_40 = false;
        }
        else
        {
            rsi_positions[i] = 0;
        }

        // Trading logic with updated RSI thresholds and price change filter
        if (state == NONE)
        {
            if (rsi > 60 && price_change > 0.05) // Enter Long
            {
                state = LONG;
                entry_price = closes[i];
                entry_index = i;
            }
            else if (rsi < 40 && price_change < -0.05) // Enter Short
            {
                state = SHORT;
                entry_price = closes[i];
                entry_index = i;
            }
        }
        else if (state == LONG)
        {
            if (rsi < 60) // Exit Long
            {
                double exit_price = closes[i];
                double ret = (exit_price - entry_price) / entry_price;
                total_return += ret;
                if (ret > profit_threshold)
                    profitable_trades++;
                total_trades++;
                state = NONE;
                entry_price = 0.0; // Reset entry price
                entry_index = -1;  // Reset entry index
            }
            // Optionally, add a stop-loss or take-profit logic here
            // else if (ret < -stop_loss_threshold) { ... }
        }
        else if (state == SHORT)
        {
            if (rsi > 40) // Exit Short
            {
                double exit_price = closes[i];
                double ret = (entry_price - exit_price) / entry_price; // For short, profit if price drops
                total_return += ret;
                if (ret > profit_threshold)
                    profitable_trades++;
                total_trades++;
                state = NONE;
                entry_price = 0.0; // Reset entry price
                entry_index = -1;  // Reset entry index
            }
            // Optionally, add a stop-loss or take-profit logic here
        }
    }

    // Force-close any open position at the end of the data
    if (state != NONE)
    {
        double final_price = closes.back();
        double ret = 0.0;
        if (state == LONG)
            ret = (final_price - entry_price) / entry_price;
        else if (state == SHORT)
            ret = (entry_price - final_price) / entry_price;

        total_return += ret;
        if (ret > profit_threshold)
            profitable_trades++;
        total_trades++;
    }

    double success_rate = total_trades > 0 ? (double)profitable_trades / total_trades * 100 : 0;
    double avg_return = total_trades > 0 ? (total_return / total_trades) * 100 : 0;

    return {success_rate, avg_return, total_trades, rsi_positions};
}
// ... (existing run_rsi_strategy code) ...

std::vector<double> calculate_rsi_series(const std::vector<Candle> &candles, int period)
{
    std::vector<double> closes;
    for (const auto &candle : candles)
        closes.push_back(candle.close);

    std::vector<double> rsi_series(closes.size(), 50.0); // Initialize with 50.0 for periods without enough data

    if (closes.size() < period + 1) // Need period + 1 for initial change
        return rsi_series;

    for (size_t i = period; i < closes.size(); ++i) // Start calculating RSI from the first full period
    {
        rsi_series[i] = calculate_rsi(closes, i, period);
    }
    return rsi_series;
}