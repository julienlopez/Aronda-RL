#pragma once

#include "igame.hpp"

namespace Aronda::Trainer
{

class CurlWrapper;

class JAronda : public IGame
{
public:
    JAronda(std::string port);

    virtual ~JAronda();

private:
    std::unique_ptr<CurlWrapper> m_curl;

    virtual MoveResult impl_play(const State& state, const std::size_t action) override;

    virtual State impl_begin() override;
};
}
