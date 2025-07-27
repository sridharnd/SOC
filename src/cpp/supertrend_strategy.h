#ifndef SUPERTREND_STRATEGY_H
#define SUPERTREND_STRATEGY_H

#include "data_types.h"
#include <vector>

// Function to calculate Average True Range (ATR)
double calculate_atr(const std::vector<Candle>& candles, int current_index, int period);

// Function to run the Supertrend trading strategy
TradeResult run_supertrend_strategy(const std::vector<Candle>& candles, double profit_threshold, int period, double multiplier);
// Function to calculate a vector of Supertrend values and its current trend direction
void calculate_supertrend_series(const std::vector<Candle>& candles, std::vector<double>& supertrend_line_out, std::vector<bool>& trend_up_series_out, int period = 10, double multiplier = 3.0);

#endif // SUPERTREND_STRATEGY_H
