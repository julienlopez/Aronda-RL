#pragma once

#include "types.hpp"

#include <string>

namespace Aronda::Trainer
{

namespace Impl
{
    class CntkBrain;
}

class CntkBrain
{
public:
    CntkBrain();
    ~CntkBrain();

    void save(const std::string& path) const;

    Action predict(const State& current_state) const;

    void train(const State& state, const Action& action);

private:
    std::unique_ptr<Impl::CntkBrain> m_pimpl;
};
}
