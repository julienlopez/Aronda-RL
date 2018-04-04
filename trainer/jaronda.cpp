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

    bool startsWith(const std::string& str, const std::string& token)
    {
        if(str.size() < token.size()) return false;
        return std::equal(begin(token), end(token), begin(str));
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
    const auto answer = m_curl->post("playMove", json);
    if(answer.empty()) throw std::runtime_error("No answer to post");
    if(answer.front() != '{')
    {
        if(startsWith(answer, "Illegal move")) return {{}, -10.};
    }
    return {Parser::parse(answer), 0.};
}

auto JAronda::impl_begin() -> GameState
{
    const auto answer = m_curl->get("startNewGame");
    return Parser::parse(answer);
}
}
