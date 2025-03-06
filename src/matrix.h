#pragma once
#include<iostream>
#include<algorithm>
#include <iomanip>
#include "nodetype.h"
#include "helper.h"
#include <vector>

template<typename T>
struct Matrix
{
    int nr, nc;
    T *data;
    Matrix(int r_, int c_) : nr(r_),nc(c_), data(new T[nr*nc]) {}
    Matrix(int r_, int c_, T val) : nr(r_), nc(c_), data(new T[nr * nc]) { std::fill_n(data, nr * nc, val); }
    Matrix(int r_, int c_, const std::vector<T> &input) : nr(r_), nc(c_), data(new T[nr * nc]) { std::copy(input.begin(), input.end(), data); }
    ~Matrix() { delete[] data; }
    Matrix(const Matrix& other) : nr(other.nr), nc(other.nc), data(new T[other.nr * other.nc]) {
        std::copy(other.data, other.data + (nr * nc), data);
    }
    Matrix(Matrix&& other) noexcept : nr(other.nr), nc(other.nc), data(other.data) {
        other.nr = 0;
        other.nc = 0;
        other.data = nullptr;
    }
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            T* newData = new T[other.nr * other.nc];
            std::copy(other.data, other.data + (other.nr * other.nc), newData);
            delete[] data;
            data = newData;
            nr = other.nr;
            nc = other.nc;
        }
        return *this;
    }
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            nr = other.nr;
            nc = other.nc;
            other.data = nullptr;
            other.nr = 0;
            other.nc = 0;
        }
        return *this;
    }
    T& operator()(int i, int j) { return data[i * nc + j]; }
    const T& operator()(int i, int j) const { return data[i * nc + j]; }
    void setRow(int r_,T val) { for (int j = 0; j < nc; ++j) (*this)(r_, j) = val; }
    void setCol(int c_,T val) { for (int i = 0; i < nr; ++i) (*this)(i, c_) = val; }
};

template <typename T>
void printM(const Matrix<T>& M);

template <typename T>
T *deviceMatrix(const Matrix<T> &M);

template <typename T>
void deviceMatrixToHost(T *d_buffer, Matrix<T> &M);
