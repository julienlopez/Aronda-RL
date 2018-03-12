#pragma once

#include "utils/strongtype.hpp"

#include <utility>

namespace Aronda::Utils
{

template <class T> struct StrongType
{
    explicit StrongType(const T index)
        : m_index(index)
    {
    }

    ~StrongType() = default;

    StrongType(const StrongType&) = default;
    StrongType(StrongType&&) = default;

    StrongType& operator=(const StrongType&) = default;
    StrongType& operator=(StrongType&&) = default;

    operator T() const
    {
        return m_index;
    }

private:
    T m_index;
};
}
