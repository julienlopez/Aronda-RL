#include "catch.hpp"

#include "state/parser.hpp"

#include <fstream>

namespace
{
std::string readAll(std::istream& is)
{
    return {std::istreambuf_iterator<char>(is), std::istreambuf_iterator<char>()};
}

using SquareState_t = Eigen::Matrix<float, 1, Aronda::State::number_of_state_per_square>;

SquareState_t emptySquareState()
{
    SquareState_t res;
    res << 1.f, Eigen::Matrix<float, 1, Aronda::State::number_of_state_per_square - 1>::Zero();
    return res;
}

SquareState_t oneMoveForCurrentPlayerOnSquare1()
{
    SquareState_t res;
    res << 0.f, 1.f, Eigen::Matrix<float, 1, Aronda::State::number_of_state_per_square - 2>::Zero();
    return res;
}
}

TEST_CASE("Reading state from json string", "[parser]")
{

    SECTION("Reading an empty board")
    {
        auto file = std::ifstream("empty.json");
        // REQUIRE(file);
        const auto state = Aronda::State::Parser::parse(readAll(file));
        for(std::size_t i = 0; i < Aronda::State::number_of_square; i++)
            CHECK(state.row(i) == emptySquareState());
    }

    SECTION("Reading a board with one move for black")
    {
        auto file = std::ifstream("one-move.json");
        // REQUIRE(file);
        const auto file_json_str = readAll(file);
        const auto state = Aronda::State::Parser::parse(file_json_str);
        for(std::size_t i = 1; i < Aronda::State::number_of_square; i++)
        {
            CHECK(state.row(i) == emptySquareState());
        }
        CHECK(state.row(0) == oneMoveForCurrentPlayerOnSquare1());
    }
}
