#pragma once

#include <random>

namespace Aronda::Trainer
{

    namespace Random
    {

        static auto& rng()
        {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            return gen;
        }

        template<class T>
        static auto uniform(const T min, const T max)
        {
            return std::uniform_int_distribution<T>(min, max)(rng());
        }

    }

}

