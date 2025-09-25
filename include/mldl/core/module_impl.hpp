#pragma once
#include "module.hpp"

namespace ml
{
    template <typename T>
    std::ostream &operator<<(std::ostream &os, const Module<T> &module)
    {
        os << module.get_info();
        return os;
    }
}