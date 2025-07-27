#ifndef MACD_STRATEGY_H
#define MACD_STRATEGY_H

#include "data_types.h"
#include <vector>

// Function to calculate Exponential Moving Average (EMA)
double calculate_ema(const std::vector<double>& data, int end_index, int period);

// Function to run the MACD trading strategy
TradeResult run_macd_strategy(const std::vector<Candle>& candles, double profit_threshold);
// Function to calculate a vector of EMA values over time (helper, if needed directly)
std::vector<double> calculate_ema_series(const std::vector<double>& data, int period);
// Function to calculate MACD line and Signal line series
void calculate_macd_series(const std::vector<Candle>& candles, std::vector<double>& macd_line_out, std::vector<double>& signal_line_out, int fast_period = 12, int slow_period = 26, int signal_period = 9);

#endif // MACD_STRATEGY_H