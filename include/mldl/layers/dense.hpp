#pragma once
#include "../core/module.hpp"
#include <vector>
#include <random>

namespace ml {

template <typename T = float>
class Dense : public Module<T> {
public:
    Dense(size_t in_features, size_t out_features);

    Tensor<T> forward(const Tensor<T>& input) override;
    Tensor<T> backward(const Tensor<T>& grad_output) override;
    void update(T lr) override;
    std::string get_info() const override;
    size_t num_params() const override;

private:
    std::vector<std::vector<T>> weights;
    std::vector<T> bias;
    std::vector<std::vector<T>> grad_weights;
    std::vector<T> grad_bias;
    Tensor<T> last_input;
};

} // namespace ml

#include "dense_impl.tpp"