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

    }

}

