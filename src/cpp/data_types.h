#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <vector>
#include <string>

// Structure to represent a single candlestick data point
struct Candle {
    long long timestamp; // Unix timestamp
    double open;
    double high;
    double low;
    double close;
    long long volume;
};

// Structure to store the results of a trading strategy
struct TradeResult {
    double success_rate;        // Percentage of profitable trades
    double per_trade_return;    // Average percentage return per trade
    int total_trades;           // Total number of trades executed
    std::vector<int> positions; // Vector to store positions (e.g., 1 for long, -1 for short, 0 for none)
};

#endif // DATA_TYPES_H