#ifndef RSI_STRATEGY_H
#define RSI_STRATEGY_H

#include <vector>
#include "data_types.h" // Include our common data types

// Function to calculate the Relative Strength Index (RSI)
double calculate_rsi(const std::vector<double> &closes, int current_index, int period = 14);

// Function to run the RSI trading strategy
TradeResult run_rsi_strategy(const std::vector<Candle> &candles, double profit_threshold);
// Function to calculate a vector of RSI values over time
std::vector<double> calculate_rsi_series(const std::vector<Candle> &candles, int period = 14);

#endif // RSI_STRATEGY_H