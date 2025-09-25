#pragma once
#include <vector>
#include <iostream>

namespace ml
{
    template <typename T = float>
    class Tensor
    {
    public:
        Tensor(size_t size) : data(size, T(0)) {}
        T &operator[](size_t i) { return data[i]; }
        const T &operator[](size_t i) const { return data[i]; }
        size_t size() const { return data.size(); }

        template<typename U> friend std::ostream &operator<<(std::ostream &os, const Tensor<U> &tensor);

    private:
        std::vector<T> data;
    };
}

#include "tensor_impl.hpp"