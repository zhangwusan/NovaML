#pragma once
#include <vector>
#include <string>
#include <sstream>

namespace NovaML::Core
{
    template <typename T>
    std::string vector_to_string(const std::vector<T> &vec)
    {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < vec.size(); i++)
        {
            oss << vec[i];
            if (i + 1 < vec.size())
                oss << ", ";
        }
        oss << "]";
        return oss.str();
    }

    template <typename T>
    void check_size_match(const std::vector<T> &a, const std::vector<T> &b, const std::string &msg = "")
    {
        if (a.size() != b.size())
        {
            throw std::invalid_argument(
                msg.empty() ? "Vector sizes do not match" : msg);
        }
    }
}