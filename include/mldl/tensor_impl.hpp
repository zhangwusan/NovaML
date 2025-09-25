// tensor_impl.hpp
#pragma once
#include "tensor.hpp"
#include <iostream>

namespace ml
{

    template <typename T>
    std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor)
    {
        os << "[";
        for (size_t i = 0; i < tensor.size(); ++i)
        {
            os << tensor[i];
            if (i != tensor.size() - 1)
                os << ", ";
        }
        os << "]";
        return os;
    }

} // namespace ml