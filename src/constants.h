#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <filesystem>
#include <cstring>
// #define DX 0.2     
// #define DY 0.2
// #define mu 1.0
// #define dt 0.01
// #define T 0.1

namespace fs = std::filesystem; 

/**
 * @brief With ::currentPath we get folder where binary called,
 * @brief try to reference everything from binary call
*/
std::string toAbsFilePath(const std::string &relpath);

struct Config{
    double DX;
    double DY;
    double mu;
    double dt;
    double T;
    Config(const std::string &filename);
    Config() = default;
    Config& operator=(const Config &other) {
        if (this != &other) {
            DX = other.DX;
            DY = other.DY;
            mu = other.mu;
            dt = other.dt;
            T = other.T;
        }
        return *this;
    }
private:
    // Helper function to trim leading and trailing whitespaces
    std::string trim(const std::string &str);
};
