#pragma once
#include <memory>
#include <cmath>
#include "tensor.hpp"
#include "autograd.hpp"

namespace NovaML::Core
{
    template <typename T>
    std::shared_ptr<Tensor<T>> add(const std::shared_ptr<Tensor<T>> &a, const std::shared_ptr<Tensor<T>> &b)
    {
        std::vector<T> result(a->size());
        for (size_t i = 0; i < result.size(); i++)
            result[i] = a->at(i) + b->at(i);

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

    template <typename T>
    std::shared_ptr<Tensor<T>> sub(const std::shared_ptr<Tensor<T>> &a, const std::shared_ptr<Tensor<T>> &b)
    {
        std::vector<T> result(a->size());
        for (size_t i = 0; i < result.size(); i++)
            result[i] = a->at(i) - b->at(i);

        auto out = std::make_shared<Tensor<T>>(result, a->get_requires_grad() || b->get_requires_grad());

        if (out->get_requires_grad())
        {
            out->add_edge({OperatorType::Sub, a, [a, b](const std::vector<T> &grad_output)
                           {
                               a->backward(grad_output);
                               std::vector<T> neg_grad = grad_output;
                               for (auto &g : neg_grad)
                                   g = -g;
                               b->backward(neg_grad);
                           }});
            out->set_grad_fn_name("<SubBackward>");
        }
        return out;
    }

    template <typename T>
    std::shared_ptr<Tensor<T>> mul(const std::shared_ptr<Tensor<T>> &a, const std::shared_ptr<Tensor<T>> &b)
    {
        std::vector<T> result(a->size());
        for (size_t i = 0; i < result.size(); i++)
            result[i] = a->at(i) * b->at(i);

        auto out = std::make_shared<Tensor<T>>(result, a->get_requires_grad() || b->get_requires_grad());

        if (out->get_requires_grad())
        {
            out->add_edge({OperatorType::Mul, a, [a, b](const std::vector<T> &grad_output)
                           {
                               std::vector<T> grad_a(grad_output.size()), grad_b(grad_output.size());
                               for (size_t i = 0; i < grad_output.size(); i++)
                               {
                                   grad_a[i] = grad_output[i] * b->at(i);
                                   grad_b[i] = grad_output[i] * a->at(i);
                               }
                               a->backward(grad_a);
                               b->backward(grad_b);
                           }});
            out->set_grad_fn_name("<MulBackward>");
        }
        return out;
    }

    template <typename T>
    std::shared_ptr<Tensor<T>> pow(const std::shared_ptr<Tensor<T>> &a, T exponent)
    {
        std::vector<T> result(a->size());
        for (size_t i = 0; i < result.size(); i++)
            result[i] = std::pow(a->at(i), exponent);

        auto out = std::make_shared<Tensor<T>>(result, a->get_requires_grad());
        if (out->get_requires_grad())
        {
            out->add_edge({OperatorType::Pow, a, [a, exponent](const std::vector<T> &grad_output)
                           {
                               std::vector<T> grad_vec(grad_output.size());
                               for (size_t i = 0; i < grad_output.size(); i++)
                                   grad_vec[i] = grad_output[i] * exponent * std::pow(a->at(i), exponent - 1);
                               a->backward(grad_vec);
                           }});
            out->set_grad_fn_name("<PowBackward>");
        }
        return out;
    }

    template <typename T>
    std::shared_ptr<Tensor<T>> neg(const std::shared_ptr<Tensor<T>> &a)
    {
        std::vector<T> result(a->size());
        for (size_t i = 0; i < result.size(); i++)
            result[i] = -a->at(i);

        auto out = std::make_shared<Tensor<T>>(result, a->get_requires_grad());
        if (out->get_requires_grad())
        {
            out->add_edge({OperatorType::Neg, a, [a](const std::vector<T> &grad_output)
                           {
                               std::vector<T> neg_grad = grad_output;
                               for (auto &g : neg_grad)
                                   g = -g;
                               a->backward(neg_grad);
                           }});
            out->set_grad_fn_name("<NegBackward>");
        }
        return out;
    }

    template <typename T>
    std::shared_ptr<Tensor<T>> add_scalar(
        const std::shared_ptr<Tensor<T>> &a,
        const T &scalar)
    {
        std::vector<T> result(a->size());
        for (size_t i = 0; i < result.size(); i++)
            result[i] = a->at(i) + scalar;

        auto out = std::make_shared<Tensor<T>>(result, a->get_requires_grad());
        if (out->get_requires_grad())
        {
            out->add_edge({OperatorType::AddScalar, a, [a](const std::vector<T> &grad_output)
                           {
                               a->backward(grad_output); // gradient w.r.t tensor is 1
                           }});
            out->set_grad_fn_name("<AddScalarBackward>");
        }
        return out;
    }

    // Tensor - scalar
    template <typename T>
    std::shared_ptr<Tensor<T>> sub_scalar(
        const std::shared_ptr<Tensor<T>> &a,
        const T &scalar)
    {
        std::vector<T> result(a->size());
        for (size_t i = 0; i < result.size(); i++)
            result[i] = a->at(i) - scalar;

        auto out = std::make_shared<Tensor<T>>(result, a->get_requires_grad());
        if (out->get_requires_grad())
        {
            out->add_edge({OperatorType::SubScalar, a, [a](const std::vector<T> &grad_output)
                           {
                               a->backward(grad_output); // gradient w.r.t tensor is 1
                           }});
            out->set_grad_fn_name("<SubScalarBackward>");
        }
        return out;
    }

    // Scalar - tensor
    template <typename T>
    std::shared_ptr<Tensor<T>> rsub_scalar(
        const T &scalar,
        const std::shared_ptr<Tensor<T>> &a)
    {
        std::vector<T> result(a->size());
        for (size_t i = 0; i < result.size(); i++)
            result[i] = scalar - a->at(i);

        auto out = std::make_shared<Tensor<T>>(result, a->get_requires_grad());
        if (out->get_requires_grad())
        {
            out->add_edge({OperatorType::SubScalar, a, [a](const std::vector<T> &grad_output)
                           {
                               std::vector<T> neg_grad = grad_output;
                               for (auto &g : neg_grad)
                                   g = -g;
                               a->backward(neg_grad);
                           }});
            out->set_grad_fn_name("<RSubScalarBackward>");
        }
        return out;
    }

    // Tensor * scalar
    template <typename T>
    std::shared_ptr<Tensor<T>> mul_scalar(
        const std::shared_ptr<Tensor<T>> &a,
        const T &scalar)
    {
        std::vector<T> result(a->size());
        for (size_t i = 0; i < result.size(); i++)
            result[i] = a->at(i) * scalar;

        auto out = std::make_shared<Tensor<T>>(result, a->get_requires_grad());
        if (out->get_requires_grad())
        {
            out->add_edge({OperatorType::MulScalar, a, [a, scalar](const std::vector<T> &grad_output)
                           {
                               std::vector<T> grad_vec = grad_output;
                               for (size_t i = 0; i < grad_vec.size(); i++)
                                   grad_vec[i] *= scalar;
                               a->backward(grad_vec);
                           }});
            out->set_grad_fn_name("<MulScalarBackward>");
        }
        return out;
    }

    template <typename T>
    std::shared_ptr<Tensor<T>> rmul_scalar(
        const T &scalar,
        const std::shared_ptr<Tensor<T>> &a)
    {
        return mul_scalar(a, scalar);
    }

    template <typename T>
    std::shared_ptr<Tensor<T>> div(
        const std::shared_ptr<Tensor<T>> &a,
        const std::shared_ptr<Tensor<T>> &b)
    {
        check_size_match(a->get_data(), b->get_data(), "div: size mismatch");

        std::vector<T> result(a->size());
        for (size_t i = 0; i < result.size(); i++)
            result[i] = a->at(i) / b->at(i);

        auto out = std::make_shared<Tensor<T>>(result, a->get_requires_grad() || b->get_requires_grad());

        if (out->get_requires_grad())
        {
            out->add_edge({OperatorType::Mul, a, [a, b](const std::vector<T> &grad_output)
                           {
                               // da = grad_output / b
                               std::vector<T> grad_a(grad_output.size());
                               for (size_t i = 0; i < grad_output.size(); i++)
                                   grad_a[i] = grad_output[i] / b->at(i);
                               a->backward(grad_a);
                           }});
            out->add_edge({OperatorType::Mul, b, [a, b](const std::vector<T> &grad_output)
                           {
                               // db = -grad_output * a / (b^2)
                               std::vector<T> grad_b(grad_output.size());
                               for (size_t i = 0; i < grad_output.size(); i++)
                                   grad_b[i] = -grad_output[i] * a->at(i) / (b->at(i) * b->at(i));
                               b->backward(grad_b);
                           }});
            out->set_grad_fn_name("<DivBackward>");
        }
        return out;
    }

    // Tensor / scalar
    template <typename T>
    std::shared_ptr<Tensor<T>> div_scalar(
        const std::shared_ptr<Tensor<T>> &a,
        const T &scalar)
    {
        std::vector<T> result(a->size());
        for (size_t i = 0; i < result.size(); i++)
            result[i] = a->at(i) / scalar;

        auto out = std::make_shared<Tensor<T>>(result, a->get_requires_grad());

        if (out->get_requires_grad())
        {
            out->add_edge({OperatorType::MulScalar, a, [a, scalar](const std::vector<T> &grad_output)
                           {
                               std::vector<T> grad_vec = grad_output;
                               for (auto &g : grad_vec)
                                   g /= scalar; // derivative w.r.t tensor
                               a->backward(grad_vec);
                           }});
            out->set_grad_fn_name("<DivScalarBackward>");
        }
        return out;
    }

    // Scalar / tensor
    template <typename T>
    std::shared_ptr<Tensor<T>> rdiv_scalar(
        const T &scalar,
        const std::shared_ptr<Tensor<T>> &a)
    {
        std::vector<T> result(a->size());
        for (size_t i = 0; i < result.size(); i++)
            result[i] = scalar / a->at(i);

        auto out = std::make_shared<Tensor<T>>(result, a->get_requires_grad());

        if (out->get_requires_grad())
        {
            out->add_edge({OperatorType::MulScalar, a, [a, scalar](const std::vector<T> &grad_output)
                           {
                               std::vector<T> grad_vec(grad_output.size());
                               for (size_t i = 0; i < grad_output.size(); i++)
                                   grad_vec[i] = -grad_output[i] * scalar / (a->at(i) * a->at(i));
                               a->backward(grad_vec);
                           }});
            out->set_grad_fn_name("<RDivScalarBackward>");
        }
        return out;
    }
}