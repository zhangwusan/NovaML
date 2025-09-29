#pragma once
#include "../module/module.hpp"
#include "../tensor/tensor.hpp"
#include <random>

namespace NovaML::Core
{
    template <typename T = float>
    class Dense : public BaseModule<T>
    {
    public:
        Dense(size_t in_features, size_t out_features, bool use_bias = true)
            : in_features(in_features), out_features(out_features), use_bias(use_bias)
        {
            std::vector<T> w_data(in_features * out_features);
            std::default_random_engine eng;
            std::uniform_real_distribution<T> dist(-0.1, 0.1);
            for (auto &v : w_data) v = dist(eng);

            weights = std::make_shared<Tensor<T>>(w_data, true);

            if (use_bias)
            {
                std::vector<T> b_data(out_features, 0);
                bias = std::make_shared<Tensor<T>>(b_data, true);
            }
        }

        std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>> &input) override
        {
            last_input = input;
            std::vector<T> out_data(out_features, 0);

            for (size_t i = 0; i < out_features; i++)
            {
                T sum = 0;
                for (size_t j = 0; j < in_features; j++)
                    sum += input->at(j) * weights->at(i * in_features + j);
                if (use_bias) sum += bias->at(i);
                out_data[i] = sum;
            }

            auto out = std::make_shared<Tensor<T>>(out_data, input->get_requires_grad());

            // Add autograd edge
            if (out->get_requires_grad())
            {
                out->add_edge({OperatorType::Mul, input, [this](const std::vector<T> &grad_output) {
                    // Grad w.r.t. input
                    std::vector<T> grad_in(in_features, 0);
                    for (size_t i = 0; i < out_features; i++)
                        for (size_t j = 0; j < in_features; j++)
                            grad_in[j] += grad_output[i] * weights->at(i * in_features + j);
                    last_input->backward(grad_in);

                    // Grad w.r.t. weights
                    std::vector<T> grad_w(in_features * out_features, 0);
                    for (size_t i = 0; i < out_features; i++)
                        for (size_t j = 0; j < in_features; j++)
                            grad_w[i * in_features + j] = grad_output[i] * last_input->at(j);
                    weights->backward(grad_w);

                    // Grad w.r.t. bias
                    if (use_bias)
                    {
                        std::vector<T> grad_b(out_features);
                        for (size_t i = 0; i < out_features; i++)
                            grad_b[i] = grad_output[i];
                        bias->backward(grad_b);
                    }
                }});
            }

            return out;
        }

        std::string info(int level = 0) const override
        {
            std::ostringstream oss;
            oss << "Dense(in=" << in_features << ", out=" << out_features
                << ", bias=" << (use_bias ? "true" : "false") << ")";
            return oss.str();
        }

        std::shared_ptr<Tensor<T>> get_weights() const { return weights; }
        std::shared_ptr<Tensor<T>> get_bias() const { return bias; }

    private:
        size_t in_features, out_features;
        bool use_bias;
        std::shared_ptr<Tensor<T>> weights;
        std::shared_ptr<Tensor<T>> bias;
        std::shared_ptr<Tensor<T>> last_input;
    };
}