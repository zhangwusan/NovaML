#include <NovaML/core/tensor/tensor.hpp>
#include <iostream>

using namespace NovaML::Core;

// Utility function to reset gradients
template <typename T>
void zero_all(const std::shared_ptr<Tensor<T>>& t) {
    t->zero_grad();
}

int main() {
    // Create two tensors
    auto a = std::make_shared<Tensor<double>>(std::vector<double>{1.0, 2.0, 3.0}, true);
    auto b = std::make_shared<Tensor<double>>(std::vector<double>{4.0, 5.0, 6.0}, true);

    // ---------- Test addition ----------
    zero_all(a); zero_all(b);
    auto sum = a + b;
    sum->backward();
    std::cout << "Addition a+b: " << *sum << std::endl;
    std::cout << "Gradients a: " << vector_to_string(a->get_grad())
              << ", b: " << vector_to_string(b->get_grad()) << "\n\n";

    // ---------- Test subtraction ----------
    zero_all(a); zero_all(b);
    auto diff = a - b;
    diff->backward();
    std::cout << "Subtraction a-b: " << *diff << std::endl;
    std::cout << "Gradients a: " << vector_to_string(a->get_grad())
              << ", b: " << vector_to_string(b->get_grad()) << "\n\n";

    // ---------- Test multiplication ----------
    zero_all(a); zero_all(b);
    auto prod = a * b;
    prod->backward();
    std::cout << "Multiplication a*b: " << *prod << std::endl;
    std::cout << "Gradients a: " << vector_to_string(a->get_grad())
              << ", b: " << vector_to_string(b->get_grad()) << "\n\n";

    // ---------- Test power ----------
    zero_all(a);
    auto power = a ^ 2.0; // a squared
    power->backward();
    std::cout << "Power a^2: " << *power << std::endl;
    std::cout << "Gradients a: " << vector_to_string(a->get_grad()) << "\n\n";

    // ---------- Test negation ----------
    zero_all(a);
    auto neg_a = -a;
    neg_a->backward();
    std::cout << "Negation -a: " << *neg_a << std::endl;
    std::cout << "Gradients a: " << vector_to_string(a->get_grad()) << "\n\n";

    return 0;
}