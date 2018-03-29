#include "jaronda.hpp"

#include "curlwrapper.hpp"

#include "state/parser.hpp"

#include <iostream>

using Aronda::State::Parser;

namespace Aronda::Trainer
{

JAronda::JAronda(std::string port)
    : m_curl(std::make_unique<CurlWrapper>("http://localhost:" + port + "/jaronda/"))
{
}

JAronda::~JAronda() = default;

auto JAronda::impl_play(const State& state, const std::size_t action) -> MoveResult
{
    return {};
}

State JAronda::impl_begin()
{
    const auto answer = m_curl->get("startNewGame");
    return Parser::parse(answer);
}
}
