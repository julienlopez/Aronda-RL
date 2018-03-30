#pragma once

#include "ibrain.hpp"

namespace Aronda::Trainer
{

namespace Impl
{
    class CntkBrain;
}

class CntkBrain : public IBrain
{
public:
    CntkBrain();
    ~CntkBrain();

private:
    std::unique_ptr<Impl::CntkBrain> m_pimpl;

    virtual void impl_save(const std::string& path) const override;

    virtual Action impl_predict(const State& current_state) const override;

    virtual void impl_train(const State& state, const Action& action) override;
};
}
