#pragma once
#include "tensor.hpp"
#include <vector>
#include <string>

namespace ml {

template <typename T = float>
class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor<T> forward(const Tensor<T>& input) = 0;
    virtual Tensor<T> backward(const Tensor<T>& grad_output) = 0;
    virtual std::string get_info() const = 0;
    virtual size_t num_params() const = 0;
};

template <typename T = float>
class Dense : public Layer<T> {
public:
    Dense(size_t in_features, size_t out_features);
    Tensor<T> forward(const Tensor<T>& input) override;
    Tensor<T> backward(const Tensor<T>& grad_output) override;
    std::string get_info() const override;
    size_t num_params() const override;

    void update(T lr);

private:
    std::vector<std::vector<T>> weights;
    std::vector<T> bias;

    std::vector<std::vector<T>> grad_weights;
    std::vector<T> grad_bias;

    Tensor<T> last_input;
};

} // namespace ml

#include "layer_impl.hpp"