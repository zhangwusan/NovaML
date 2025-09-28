#pragma once
#include "../Tensor/tensor.hpp"
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <iostream>

namespace TensorNS = NovaML::Core::TensorModule;

namespace NovaML::Core::Module
{
    template <typename T = float>
    class BaseModule
    {
    public:
        virtual ~BaseModule() = default;
        // Pure virtual for module description
        virtual std::string info(std::ostream &os) const = 0;

        virtual TensorNS::Tensor<T> forward(const TensorNS::Tensor<T> &input) = 0;
        virtual TensorNS::Tensor<T> backward(const TensorNS::Tensor<T> &grad_output);
        virtual void update(T lr);
        virtual size_t num_params() const;

    protected:
        std::vector<std::shared_ptr<BaseModule<T>>> submodules;

        template <typename U>
        friend std::ostream &operator<<(std::ostream &os, const BaseModule<U> &module);
    };
}

#include "module.tpp"
