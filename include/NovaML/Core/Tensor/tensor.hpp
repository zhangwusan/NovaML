#pragma once
#include <vector>
#include <iostream>

namespace NovaML::Core::TensorModule
{

    template <typename T = float>
    class Tensor
    {
    public:
        Tensor(size_t size);
        T &operator[](size_t i);
        const T &operator[](size_t i) const;
        size_t size() const;

        template <typename U>
        friend std::ostream &operator<<(std::ostream &os, const Tensor<U> &tensor);

    private:
        std::vector<T> data;
    };
}
#include "tensor.tpp"