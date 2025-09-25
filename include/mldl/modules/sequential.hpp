#pragma once
#include "../core/module.hpp"
#include <sstream>

namespace ml
{
    template <typename T = float>
    class Sequential : public Module<T>
    {
    public:
        Sequential() = default;

        // Add a layer
        void add(std::shared_ptr<Module<T>> layer);

        // Forward pass
        Tensor<T> forward(const Tensor<T> &input) override;

        // Info
        std::string get_info() const override;

        template <typename U> 
        friend std::ostream &operator<<(std::ostream &os, const Sequential<U> &seq);

    };

}

#include "sequential_impl.tpp"