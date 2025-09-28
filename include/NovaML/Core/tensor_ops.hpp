#pragma once
#include <memory>
#include <cmath>
#include "tensor.hpp"
#include "autograd.hpp"

namespace NovaML::Core
{
    // Arithmetic Operation

    template <typename T>
    std::shared_ptr<Tensor<T>> add(
        const std::shared_ptr<Tensor<T>> &a,
        const std::shared_ptr<Tensor<T>> &b)
    {
        std::vector<T> result(a->size());

        for (size_t i = 0; i < result.size(); i++)
        {
            result[i] = a->at(i) + b->at(i);
        }

        auto out = std::make_shared<Tensor<T>>(result, a->get_requires_grad() || b->get_requires_grad());

        if (out->get_requires_grad())
        {
            out->add_edge({OperatorType::Add, a, [a, b](const std::vector<T> &grad_output)
                           {
                               a->backward(grad_output);
                               b->backward(grad_output);
                           }});
            out->set_grad_fn_name("<AddBackward>");
        }

        return out;
    }
}