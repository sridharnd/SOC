#include "macd_strategy.h"
#include <vector>
#include <numeric> // For std::accumulate
#include "data_types.h" // Added: Include for Candle and TradeResult structs

using namespace std;

// Modified calculate_ema for better initial value handling
double calculate_ema(const vector<double>& data, int end_index, int period) {
    if (end_index < 0 || end_index >= data.size()) {
        return 0.0; // Invalid index
    }

    // For the very first EMA calculation, often an SMA is used as the initial EMA.
    // Subsequent EMAs use the previous EMA.
    if (end_index + 1 < period) { // Not enough data for full period, just return current value
        return data[end_index];
    }

    double k = 2.0 / (period + 1);
    double ema_val;

    // Determine the starting point for SMA calculation to get the initial EMA
    // If we are at the very first point where a full 'period' is available,
    // calculate simple moving average (SMA) as the initial EMA.
    if (end_index + 1 == period) {
        double sum = 0.0;
        for (int i = 0; i <= end_index; ++i) {
            sum += data[i];
        }
        ema_val = sum / period;
    } else {
        // Otherwise, we need the EMA from the previous point.
        // This function calculates EMA for *a specific point*.
        // For a full MACD calculation, you'd typically calculate EMAs for each point sequentially.
        // The way this `calculate_ema` is used in `run_macd_strategy` implies it's calculating the EMA
        // *up to* `end_index` using `period` data points.

        // To make this `calculate_ema` function work correctly as a standalone for a single point,
        // it needs access to the *previous* EMA. Since it doesn't, it's calculating a kind of 'rolling' EMA
        // starting from `end_index - period + 1` which is more like an SMA that then "smooths" towards EMA.
        // The current implementation is a bit simplified for a true EMA series.
        // Let's assume the current `calculate_ema` is intended to give an EMA-like value over the window.

        // A more correct EMA calculation (if we were building an EMA series):
        // EMA_current = (Close_current - EMA_previous) * K + EMA_previous
        // Since we don't have EMA_previous easily here, your current loop structure is
        // an iterative calculation over the window.
        // The way your code is structured, `ema = data[end_index - period + 1];` is the starting point
        // and then it iteratively applies the smoothing. This is a common simplification for a single point EMA.

        ema_val = data[end_index - period + 1]; // First value in the window
        for (int i = end_index - period + 2; i <= end_index; ++i) {
            ema_val = data[i] * k + ema_val * (1 - k);
        }
    }
    return ema_val;
}


TradeResult run_macd_strategy(const vector<Candle>& candles, double profit_threshold) {
    vector<double> closes;
    for (const auto& candle : candles)
        closes.push_back(candle.close);

    // MACD needs at least 26 periods for the initial EMA26, plus 9 for signal line,
    // so roughly 34 candles for first valid signal.
    if (closes.size() < 34) {
        return {0.0, 0.0, 0, vector<int>(closes.size(), 0)};
    }

    vector<int> macd_positions(closes.size(), 0);

    vector<double> macd_line_values; // To store MACD values to calculate signal line
    vector<double> signal_line_values; // To store Signal line values

    int profitable_trades = 0;
    double total_return = 0.0;
    int total_trades = 0;
    double entry_price = 0.0;
    bool in_position = false; // True if currently holding a long position

    // Start calculating from index 25 (0-indexed for 26 candles)
    for (size_t i = 25; i < closes.size(); ++i) {
        double ema12 = calculate_ema(closes, i, 12);
        double ema26 = calculate_ema(closes, i, 26);
        double current_macd = ema12 - ema26;
        macd_line_values.push_back(current_macd);

        double current_signal = 0.0;
        // Need at least 9 MACD values to calculate the first signal line EMA
        if (macd_line_values.size() >= 9) {
            current_signal = calculate_ema(macd_line_values, macd_line_values.size() - 1, 9);
            signal_line_values.push_back(current_signal);

            // Ensure we have a previous MACD and Signal value for crossover logic
            if (macd_line_values.size() > 1 && signal_line_values.size() > 1) {
                double prev_macd = macd_line_values[macd_line_values.size() - 2];
                double prev_signal = signal_line_values[signal_line_values.size() - 2];

                // Trading signals
                if (!in_position && prev_macd < prev_signal && current_macd > current_signal) {
                    // Buy signal (MACD crosses above Signal)
                    entry_price = candles[i].close;
                    in_position = true;
                    macd_positions[i] = 1; // Mark as buy
                } else if (in_position && prev_macd > prev_signal && current_macd < current_signal) {
                    // Sell signal (MACD crosses below Signal)
                    double exit_price = candles[i].close;
                    double ret = (exit_price - entry_price) / entry_price;
                    total_return += ret;
                    if (ret > profit_threshold)
                        profitable_trades++;
                    total_trades++;
                    in_position = false;
                    macd_positions[i] = -1; // Mark as sell
                    entry_price = 0.0; // Reset entry price
                } else {
                    macd_positions[i] = 0; // No action
                }
            } else {
                macd_positions[i] = 0; // Not enough history for signals yet
            }
        } else {
            signal_line_values.push_back(current_macd); // Or just 0.0 until enough data
            macd_positions[i] = 0; // Not enough data for signal
        }
    }

    // Force-close any open position at the end of the data
    if (in_position) {
        double exit_price = closes.back();
        double ret = (exit_price - entry_price) / entry_price;
        total_return += ret;
        if (ret > profit_threshold)
            profitable_trades++;
        total_trades++;
    }

    double success_rate = total_trades > 0 ? (double)profitable_trades / total_trades * 100 : 0;
    double avg_return = total_trades > 0 ? (total_return / total_trades) * 100 : 0;

    return {success_rate, avg_return, total_trades, macd_positions};
}
// ... (existing run_macd_strategy code) ...

// Helper to calculate a series of EMA values
std::vector<double> calculate_ema_series(const std::vector<double>& data, int period) {
    std::vector<double> ema_series(data.size(), 0.0);
    if (data.empty()) return ema_series;

    double k = 2.0 / (period + 1);

    // Calculate initial EMA (often SMA for the first 'period' values)
    double sum_first_period = 0.0;
    for (int i = 0; i < std::min((int)data.size(), period); ++i) {
        sum_first_period += data[i];
    }
    if (period > 0) { // Avoid division by zero
        // Initialize the first period-1 elements as 0.0 (or some default like data[0])
        // Then the EMA calculation can start from period-1.
        // The first meaningful EMA value is usually at index `period-1` (0-indexed) or `period` (1-indexed).
        // Let's ensure the array size matches `data.size()` before using indices.
        if (data.size() >= period) {
            ema_series[period - 1] = sum_first_period / period; // SMA for first N values
        } else {
            // Not enough data for even one EMA calculation. Fill with default.
            for (size_t i = 0; i < data.size(); ++i) ema_series[i] = data[i]; // Or 0.0
            return ema_series;
        }
    }


    // Calculate subsequent EMAs
    for (size_t i = period; i < data.size(); ++i) {
        ema_series[i] = data[i] * k + ema_series[i-1] * (1 - k);
    }
    return ema_series;
}

void calculate_macd_series(const std::vector<Candle>& candles, std::vector<double>& macd_line_out, std::vector<double>& signal_line_out, int fast_period, int slow_period, int signal_period) {
    std::vector<double> closes;
    for (const auto& candle : candles)
        closes.push_back(candle.close);

    // Ensure output vectors are correctly sized and initialized
    macd_line_out.assign(closes.size(), 0.0);
    signal_line_out.assign(closes.size(), 0.0);

    if (closes.size() < slow_period) { // Not enough data for slow EMA
        return;
    }

    std::vector<double> ema_fast_series;
    std::vector<double> ema_slow_series;

    // Calculate EMAs for the full series
    ema_fast_series = calculate_ema_series(closes, fast_period);
    ema_slow_series = calculate_ema_series(closes, slow_period);

    // Calculate MACD line
    for (size_t i = 0; i < closes.size(); ++i) {
        if (i >= fast_period - 1 && i >= slow_period - 1) { // Ensure both EMAs are valid at this point
            macd_line_out[i] = ema_fast_series[i] - ema_slow_series[i];
        } else {
            macd_line_out[i] = 0.0; // Fill warmup period with 0
        }
    }

    // Calculate Signal line (EMA of MACD line)
    if (closes.size() >= slow_period + signal_period - 1) { // Need enough MACD values for signal EMA
        signal_line_out = calculate_ema_series(macd_line_out, signal_period);
    } else {
        // Not enough data for signal line, fill with 0s
        signal_line_out.assign(closes.size(), 0.0);
    }
}