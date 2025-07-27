#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Required for automatic conversion of std::vector, std::map, etc.

#include "data_types.h"
#include "rsi_strategy.h"
#include "macd_strategy.h"
#include "supertrend_strategy.h"

namespace py = pybind11;

PYBIND11_MODULE(trading_strategies, m) {
    m.doc() = "pybind11 example plugin for trading strategies"; // Optional module docstring

    // Bind Candle struct
    py::class_<Candle>(m, "Candle")
        .def(py::init<>()) // Default constructor
        .def_readwrite("timestamp", &Candle::timestamp)
        .def_readwrite("open", &Candle::open)
        .def_readwrite("high", &Candle::high)
        .def_readwrite("low", &Candle::low)
        .def_readwrite("close", &Candle::close)
        .def_readwrite("volume", &Candle::volume);

    // Bind TradeResult struct
    py::class_<TradeResult>(m, "TradeResult")
        .def(py::init<>()) // Default constructor
        .def_readwrite("success_rate", &TradeResult::success_rate)
        .def_readwrite("per_trade_return", &TradeResult::per_trade_return)
        .def_readwrite("total_trades", &TradeResult::total_trades)
        .def_readwrite("positions", &TradeResult::positions); // std::vector<int> will be converted to Python list

    // Bind RSI strategy functions
    m.def("run_rsi_strategy", &run_rsi_strategy,
          "A function to run the RSI trading strategy.",
          py::arg("candles"), py::arg("profit_threshold"));

    // Bind MACD strategy functions
    m.def("run_macd_strategy", &run_macd_strategy,
          "A function to run the MACD trading strategy.",
          py::arg("candles"), py::arg("profit_threshold"));

    // Bind Supertrend strategy functions
    m.def("run_supertrend_strategy", &run_supertrend_strategy,
          "A function to run the Supertrend trading strategy.",
          py::arg("candles"), py::arg("profit_threshold"), py::arg("period") = 10, py::arg("multiplier") = 3.0);

          // Expose indicator series functions for NN features
m.def("calculate_rsi_series", &calculate_rsi_series, "Calculate a series of RSI values.");

// For functions returning multiple values, use py::make_tuple
m.def("calculate_macd_series", [](const std::vector<Candle>& candles, int fast_period, int slow_period, int signal_period){
    std::vector<double> macd_line, signal_line;
    calculate_macd_series(candles, macd_line, signal_line, fast_period, slow_period, signal_period);
    return py::make_tuple(macd_line, signal_line);
}, py::arg("candles"), py::arg("fast_period") = 12, py::arg("slow_period") = 26, py::arg("signal_period") = 9, "Calculate MACD and Signal lines.");

m.def("calculate_supertrend_series", [](const std::vector<Candle>& candles, int period, double multiplier){
    std::vector<double> supertrend_line;
    std::vector<bool> trend_up_series;
    calculate_supertrend_series(candles, supertrend_line, trend_up_series, period, multiplier);
    return py::make_tuple(supertrend_line, trend_up_series);
}, py::arg("candles"), py::arg("period") = 10, py::arg("multiplier") = 3.0, "Calculate Supertrend line and trend series.");
}