#pragma once
#include <vector>
#include "utils.hpp"
#include "autograd.hpp"
#include "tensor_ops.hpp"

namespace NovaML::Core
{
    /**
     * @brief A simple Tensor class that supports automatic differentiation.
     *
     * @tparam T Data type of elements (default: float).
     */
    template <typename T = float>
    class Tensor : public std::enable_shared_from_this<Tensor<T>>
    {
    public:
        /**
         * @brief Construct a new Tensor object.
         *
         * Initializes the tensor with a given size and fills
         * all elements with zero. Optionally, gradients can be tracked.
         *
         * @param size Number of elements in the tensor.
         * @param requires_grad Whether this tensor should track gradients (default: false).
         */
        Tensor(size_t size, bool requires_grad = false)
            : data(size, T(0)), // tensor data initialized to 0
              grad(size, T(0)), // gradient vector initialized to 0
              requires_grad(requires_grad)
        {
        } ///< Initialize vector with zeros

        Tensor(const std::vector<T> &vec, bool requires_grad = false)
            : data(vec), grad(vec.size(), T(0)), requires_grad(requires_grad) {}
        /**
         * @brief Access
         */
        T &operator[](size_t i) { return data[i]; }
        const T &operator[](size_t i) const { return data[i]; }

        const std::vector<T> &get_data() const { return data; }
        const std::vector<T> &get_grad() const { return grad; }
        const bool get_requires_grad() const { return requires_grad; }

        size_t size() const { return data.size(); }
        T at(size_t i) const { return data[i]; }
        void set_grad_fn_name(const std::string &name) { grad_fn_name = name; }

        void zero_grad()
        {
            std::fill(grad.begin(), grad.end(), T(0));
            edges.clear();
        }

        void accumulate_grad(const std::vector<T> &g)
        {
            check_size_match(grad, g, "accumulate_grad: gradient size mismatch");
            for (size_t i = 0; i < grad.size(); i++)
            {
                grad[i] += g[i];
            }
        }

        void add_edge(const Edge<T> &edge)
        {
            edges.push_back(edge);
        }

        void backward(const std::vector<T> &grad_output = {})
        {
            if (!requires_grad)
                return;
            std::vector<T> g = grad_output.empty() ? std::vector<T>(data.size(), 1) : grad_output;
            accumulate_grad(g);
            for (auto &edge : edges)
                edge.backward_fn(g);
        }

        friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &t)
        {
            os << "Tensor(data=[";
            for (size_t i = 0; i < t.data.size(); i++)
            {
                os << t.data[i];
                if (i != t.data.size() - 1)
                    os << ", ";
            }
            os << "], grad=[";
            for (size_t i = 0; i < t.grad.size(); i++)
            {
                os << t.grad[i];
                if (i != t.grad.size() - 1)
                    os << ", ";
            }
            os << "], requires_grad=" << (t.requires_grad ? "true" : "false");
            if (!t.grad_fn_name.empty())
                os << ", grad_fn=" << t.grad_fn_name;
            os << ")";
            return os;
        }

    private:
        std::vector<T> data;        ///< Stores tensor values
        std::vector<T> grad;        ///< Gradient values (same size as data)
        std::vector<Edge<T>> edges; ///< Computational graph edges (for autograd)
        bool requires_grad;         ///< Flag to enable/disable gradient tracking
        std::string grad_fn_name = "";
    };

    // Friend operators
    template <typename T>
    std::shared_ptr<Tensor<T>> operator+(const std::shared_ptr<Tensor<T>> &a, const std::shared_ptr<Tensor<T>> &b) { return add(a, b); }
    template <typename T>
    std::shared_ptr<Tensor<T>> operator-(const std::shared_ptr<Tensor<T>> &a, const std::shared_ptr<Tensor<T>> &b) { return sub(a, b); }
    template <typename T>
    std::shared_ptr<Tensor<T>> operator*(const std::shared_ptr<Tensor<T>> &a, const std::shared_ptr<Tensor<T>> &b) { return mul(a, b); }
    template <typename T>
    std::shared_ptr<Tensor<T>> operator^(const std::shared_ptr<Tensor<T>> &a, T exponent) { return pow(a, exponent); }
    template <typename T>
    std::shared_ptr<Tensor<T>> operator-(const std::shared_ptr<Tensor<T>> &a) { return neg(a); }
    template <typename T>
    std::shared_ptr<Tensor<T>> operator/(const std::shared_ptr<Tensor<T>> &a, const std::shared_ptr<Tensor<T>> &b) { return div(a, b); }


    // Tensor-Scalar
    template <typename T, typename U>
    std::shared_ptr<Tensor<T>> operator+(const std::shared_ptr<Tensor<T>> &a, const U &scalar) { return add_scalar(a, static_cast<T>(scalar)); }

    template <typename T, typename U>
    std::shared_ptr<Tensor<T>> operator+(const U &scalar, const std::shared_ptr<Tensor<T>> &a) { return add_scalar(a, static_cast<T>(scalar)); }

    template <typename T, typename U>
    std::shared_ptr<Tensor<T>> operator-(const std::shared_ptr<Tensor<T>> &a, const U &scalar) { return sub_scalar(a, static_cast<T>(scalar)); }

    template <typename T, typename U>
    std::shared_ptr<Tensor<T>> operator-(const U &scalar, const std::shared_ptr<Tensor<T>> &a) { return rsub_scalar(static_cast<T>(scalar), a); }

    template <typename T, typename U>
    std::shared_ptr<Tensor<T>> operator*(const std::shared_ptr<Tensor<T>> &a, const U &scalar) { return mul_scalar(a, static_cast<T>(scalar)); }

    template <typename T, typename U>
    std::shared_ptr<Tensor<T>> operator*(const U &scalar, const std::shared_ptr<Tensor<T>> &a) { return rmul_scalar(static_cast<T>(scalar), a); }

    template <typename T, typename U>
    std::shared_ptr<Tensor<T>> operator/(const std::shared_ptr<Tensor<T>> &a, const U &scalar) { return div_scalar(a, static_cast<T>(scalar)); }

    template <typename T, typename U>
    std::shared_ptr<Tensor<T>> operator/(const U &scalar, const std::shared_ptr<Tensor<T>> &a) { return rdiv_scalar(static_cast<T>(scalar), a); }

}