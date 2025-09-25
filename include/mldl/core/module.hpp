#pragma once
#include "tensor.hpp"
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>

namespace ml
{

    template <typename T = float>
    class Module
    {
    public:
        virtual ~Module() = default;

        // Each child must implement forward
        virtual Tensor<T> forward(const Tensor<T> &input) = 0;

        // Default backward (iterate child modules if any)
        virtual Tensor<T> backward(const Tensor<T> &grad_output)
        {
            Tensor<T> grad = grad_output;
            for (auto it = submodules.rbegin(); it != submodules.rend(); ++it)
                grad = (*it)->backward(grad);
            return grad;
        }

        virtual void update(T lr)
        {
            // default: call update on submodules
            for (auto &m : submodules)
                m->update(lr);
        }

        virtual std::string get_info() const
        {
            std::ostringstream oss;
            oss << "Module with " << submodules.size() << " submodules\n";
            for (size_t i = 0; i < submodules.size(); ++i)
                oss << " Submodule " << i + 1 << ": "
                    << submodules[i]->get_info()
                    << " | Params: " << submodules[i]->num_params()
                    << "\n";
            oss << "Total parameters: " << num_params();
            return oss.str();
        }

        virtual size_t num_params() const
        {
            size_t total = 0;
            for (auto &m : submodules)
                total += m->num_params();
            return total;
        }

        template <typename U>
        friend std::ostream &operator<<(std::ostream &os, const Module<U> &module);

    protected:
        std::vector<std::shared_ptr<Module<T>>> submodules;
    };
}

#include "module_impl.tpp"