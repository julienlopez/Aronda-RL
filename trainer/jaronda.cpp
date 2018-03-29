#include "jaronda.hpp"

#include "curlwrapper.hpp"

#include "state/parser.hpp"

#include <iostream>

using Aronda::State::Parser;

namespace Aronda::Trainer
{

namespace
{
    std::string encodeMove(const std::size_t action)
    {
        auto res = nlohmann::json::object();
        res["row"] = action / 8;
        res["squareNumber"] = action % 8;
        return res.dump();
    }
}

JAronda::JAronda(std::string port)
    : m_curl(std::make_unique<CurlWrapper>("http://localhost:" + port + "/jaronda/"))
{
}

JAronda::~JAronda() = default;

auto JAronda::impl_play(const State& state, const std::size_t action) -> MoveResult
{
    const auto json = encodeMove(action);
    std::cout << "JAronda::play(" << action << ") => " << json << std::endl;
    ;
    const auto answer = m_curl->post("playMove", json);
    std::cout << answer << std::endl;
    // Parser::parse(answer);
    return {{}, 0.};
}

State JAronda::impl_begin()
{
    const auto answer = m_curl->get("startNewGame");
    return Parser::parse(answer);
}
}
