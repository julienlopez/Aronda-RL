#pragma once

#include "random.hpp"

#include <vector>

#include <gsl/gsl_assert>

namespace Aronda::Trainer
{

    template<class T>
    class Memory
    {
    public:
        using Container_t = std::vector<T>;

        Memory(const std::size_t maximum_size)
            : m_maximum_size(maximum_size)
        {
        }

        ~Memory() = default;

        auto size() const
        {
            return m_container.size();
        }

        void add(T t)
        {
            m_container.push_back(std::move(t));
            if (size() > m_maximum_size)
                m_container.erase(begin(m_container));
        }

        /**
        * @pre size() > 0
        */
        Container_t sample(const std::size_t number_of_samples) const
        {
            Expects(size() > 0);
            const auto count = std::min(number_of_samples, size());
            Container_t res;
            for (std::size_t i = 0; i < count; i++)
            {
                const auto index = std::uniform_int_distribution<std::size_t>(0, size() - 1)(Random::rng());
                res.push_back(m_container[index]);
            }
            return res;
        }

    private:
        const std::size_t m_maximum_size;
        Container_t m_container;
    };

}
