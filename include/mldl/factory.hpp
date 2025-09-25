#include "activation.hpp"
#include "layer.hpp"

namespace ml
{

    // Factory for Dense
    template <typename T = float, typename... Args>
    std::shared_ptr<Dense<T>> make_dense(Args &&...args)
    {
        return std::make_shared<Dense<T>>(std::forward<Args>(args)...);
    }

    // Factory for ReLU
    template <typename T = float, typename... Args>
    std::shared_ptr<ReLU<T>> make_relu(Args &&...args)
    {
        return std::make_shared<ReLU<T>>(std::forward<Args>(args)...);
    }

    // Factory for Sigmoid
    template <typename T = float, typename... Args>
    std::shared_ptr<Sigmoid<T>> make_sigmoid(Args &&...args)
    {
        return std::make_shared<Sigmoid<T>>(std::forward<Args>(args)...);
    }

} // namespace ml