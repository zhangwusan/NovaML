#pragma once
#include "layer.hpp"
#include <memory>
#include <vector>
#include <iostream>

namespace ml
{
    template <typename T = float>
    class Model
    {
    public:
        Model() = default;

        // Add layer to model
        void add(std::shared_ptr<Layer<T>> layer);

        // Forward pass through all layers
        Tensor<T> forward(const Tensor<T> &input);

        // Backward pass through all layers
        Tensor<T> backward(const Tensor<T> &grad_output);

        // Update trainable layers
        void update(T lr);

        // Access layers
        const std::vector<std::shared_ptr<Layer<T>>> &get_layers() const;

        // Total trainable parameters
        size_t total_params() const;

        // Approximate memory usage in bytes
        size_t total_memory_bytes() const;

    private:
        std::vector<std::shared_ptr<Layer<T>>> layers;
    };

    // Overload << operator for full summary
    template <typename U>
    std::ostream &operator<<(std::ostream &os, const Model<U> &model);
}

#include "model_impl.hpp"