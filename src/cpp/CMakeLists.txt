cmake_minimum_required(VERSION 3.15...3.25)

project(trading_strategies LANGUAGES CXX)

project(trading_strategies LANGUAGES CXX)
# Explicitly set the installation prefix to the project source directory
# ... rest of your CMakeLists.txt

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

find_package(pybind11 REQUIRED CONFIG)

pybind11_add_module(trading_strategies bindings.cpp rsi_strategy.cpp macd_strategy.cpp supertrend_strategy.cpp)

target_include_directories(trading_strategies PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}" # For data_types.h, rsi_strategy.h, etc.
)

target_link_libraries(trading_strategies PRIVATE Python3::Python)

# Set the output directory for the compiled module (within the build folder)
# This is where the .so file will be generated BEFORE installation
set_target_properties(trading_strategies PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# Use install() command to copy the generated module to the python directory
# This is more robust for copying build artifacts
# Define a variable for the absolute install path of the Python module
# This resolves to /Users/narrasridhar/24B3906/src/python
set(PYTHON_MODULE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/src/python")

# Use install() command to copy the generated module to the Python directory
install(TARGETS trading_strategies
    DESTINATION "${PYTHON_MODULE_INSTALL_DIR}"
    LIBRARY DESTINATION "${PYTHON_MODULE_INSTALL_DIR}"
    RUNTIME DESTINATION "${PYTHON_MODULE_INSTALL_DIR}"
)