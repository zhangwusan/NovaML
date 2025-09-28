#pragma once
#include <vector>
#include <functional>
#include "tensor.hpp"

namespace NovaML::Core
{

    template <typename T> class Tensor;

    enum class OperatorType
    {
        Add,       // tensor + tensor
        Sub,       // tensor - tensor
        Mul,       // tensor * tensor (element-wise)
        Pow,       // tensor ^ scalar
        Neg,       // -tensor
        AddScalar, // tensor + scalar
        SubScalar, // tensor - scalar or scalar - tensor
        MulScalar  // tensor * scalar or scalar * tensor
    };

    template <typename T>
    struct Edge
    {
        OperatorType op;
        std::weak_ptr<Tensor<T>> parent;
        std::function<void(const std::vector<T> &)> backward_fn;
    };

}