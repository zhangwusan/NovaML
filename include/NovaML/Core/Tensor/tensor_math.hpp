#pragma once
#include <memory>
#include <vector>
#include <cmath>
#include "tensor.hpp"
#include "autograd.hpp"
#include "utils.hpp"

namespace NovaML::Core
{
    // -------------------------
    // Sum: reduces a tensor to a scalar
    // -------------------------
    template <typename T>
    std::shared_ptr<Tensor<T>> sum(const std::shared_ptr<Tensor<T>> &a)
    {
        T result = 0;
        for (const auto &v : a->get_data())
            result += v;

        auto out = std::make_shared<Tensor<T>>(std::vector<T>{result}, a->get_requires_grad());

        if (out->get_requires_grad())
        {
            out->add_edge({OperatorType::Add, a, [a](const std::vector<T> &grad_output)
                           {
                               std::vector<T> grad_vec(a->size(), grad_output[0]);
                               a->backward(grad_vec);
                           }});
            out->set_grad_fn_name("<SumBackward>");
        }

        return out;
    }

    // -------------------------
    // Mean: average of tensor
    // -------------------------
    template <typename T>
    std::shared_ptr<Tensor<T>> mean(const std::shared_ptr<Tensor<T>> &a)
    {
        T result = 0;
        for (const auto &v : a->get_data())
            result += v;
        result /= a->size();

        auto out = std::make_shared<Tensor<T>>(std::vector<T>{result}, a->get_requires_grad());

        if (out->get_requires_grad())
        {
            out->add_edge({OperatorType::MulScalar, a, [a](const std::vector<T> &grad_output)
                           {
                               std::vector<T> grad_vec(a->size(), grad_output[0] / a->size());
                               a->backward(grad_vec);
                           }});
            out->set_grad_fn_name("<MeanBackward>");
        }

        return out;
    }

    // -------------------------
    // Element-wise exponential
    // -------------------------
    template <typename T>
    std::shared_ptr<Tensor<T>> exp(const std::shared_ptr<Tensor<T>> &a)
    {
        std::vector<T> result(a->size());
        for (size_t i = 0; i < a->size(); i++)
            result[i] = std::exp(a->at(i));

        auto out = std::make_shared<Tensor<T>>(result, a->get_requires_grad());

        if (out->get_requires_grad())
        {
            out->add_edge({OperatorType::Pow, a, [a, out](const std::vector<T> &grad_output)
                           {
                               std::vector<T> grad_vec(a->size());
                               for (size_t i = 0; i < a->size(); i++)
                                   grad_vec[i] = grad_output[i] * out->at(i); // d/dx e^x = e^x
                               a->backward(grad_vec);
                           }});
            out->set_grad_fn_name("<ExpBackward>");
        }

        return out;
    }

    // -------------------------
    // Element-wise logarithm
    // -------------------------
    template <typename T>
    std::shared_ptr<Tensor<T>> log(const std::shared_ptr<Tensor<T>> &a)
    {
        std::vector<T> result(a->size());
        for (size_t i = 0; i < a->size(); i++)
        {
            if (a->at(i) <= 0)
                throw std::runtime_error("log: input must be positive");
            result[i] = std::log(a->at(i));
        }

        auto out = std::make_shared<Tensor<T>>(result, a->get_requires_grad());

        if (out->get_requires_grad())
        {
            out->add_edge({OperatorType::Pow, a, [a](const std::vector<T> &grad_output)
                           {
                               std::vector<T> grad_vec(a->size());
                               for (size_t i = 0; i < a->size(); i++)
                                   grad_vec[i] = grad_output[i] / a->at(i); // d/dx log(x) = 1/x
                               a->backward(grad_vec);
                           }});
            out->set_grad_fn_name("<LogBackward>");
        }

        return out;
    }
}