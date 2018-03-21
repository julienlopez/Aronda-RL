#include "catch.hpp"

#include "state/parser.hpp"

#include <fstream>

namespace
{
std::string readAll(std::istream& is)
{
    return {std::istreambuf_iterator<char>(is), std::istreambuf_iterator<char>()};
}

Aronda::State::Square emptySquareState()
{
    Aronda::State::Square res;
    res << 1.f, Aronda::Utils::Matrix<1, Aronda::State::number_of_state_per_square - 1>::Zero();
    return res;
}

Aronda::State::Square oneMoveForCurrentPlayerOnSquare1()
{
    Aronda::State::Square res;
    res << 0.f, 1.f, Aronda::Utils::Matrix<1, Aronda::State::number_of_state_per_square - 2>::Zero();
    return res;
}

Aronda::State::Square conqueredByCurrentPlayerSquareState()
{
    Aronda::State::Square res;
    res << Aronda::Utils::Matrix<1, Aronda::State::number_of_state_per_square - 2>::Zero(), 1.f, 0.f;
    return res;
}

Aronda::State::Square conqueredByOtherPlayerSquareState()
{
    Aronda::State::Square res;
    res << Aronda::Utils::Matrix<1, Aronda::State::number_of_state_per_square - 2>::Zero(), 0.f, 1.f;
    return res;
}

nlohmann::json generateJsonForSquare(const std::size_t number_of_black_paws, const std::size_t number_of_white_paws,
                                     const std::string conquering_color = "null")
{
    nlohmann::json json;
    json["numberOfBlackPawns"] = number_of_black_paws;
    json["numberOfWhitePawns"] = number_of_white_paws;
    json["conqueringColor"] = conquering_color;
    return json;
}
}

TEST_CASE("Reading state from json string", "[parser]")
{

    SECTION("Reading an empty board")
    {
        auto file = std::ifstream("empty.json");
        REQUIRE(file);
        const auto state = Aronda::State::Parser::parse(readAll(file));
        for(std::size_t i = 0; i < Aronda::State::number_of_square; i++)
            CHECK(state.row(i) == emptySquareState());
    }

    SECTION("Reading a board with one move for black")
    {
        auto file = std::ifstream("one-move.json");
        REQUIRE(file);
        const auto file_json_str = readAll(file);
        const auto state = Aronda::State::Parser::parse(file_json_str);
        for(std::size_t i = 1; i < Aronda::State::number_of_square; i++)
        {
            CHECK(state.row(i) == emptySquareState());
        }
        CHECK(state.row(0) == oneMoveForCurrentPlayerOnSquare1());
    }
}

TEST_CASE("Reading row state from json string", "[parser]")
{
    SECTION("Reading an empty square is the same for both players")
    {
        const auto json = generateJsonForSquare(0, 0);
        const auto state_for_black = Aronda::State::Parser::parseSquare(json, Aronda::State::Player::Black);
        const auto state_for_white = Aronda::State::Parser::parseSquare(json, Aronda::State::Player::White);
        CHECK(state_for_black == state_for_white);
        CHECK(state_for_black == emptySquareState());
    }

    SECTION("Parsing a square conquered by black")
    {
        const auto json = generateJsonForSquare(0, 0, "BLACK");
        const auto state_for_black = Aronda::State::Parser::parseSquare(json, Aronda::State::Player::Black);
        const auto state_for_white = Aronda::State::Parser::parseSquare(json, Aronda::State::Player::White);
        CHECK(state_for_black == conqueredByCurrentPlayerSquareState());
        CHECK(state_for_white == conqueredByOtherPlayerSquareState());
    }

    SECTION("Parsing a square conquered by white")
    {
        const auto json = generateJsonForSquare(0, 0, "WHITE");
        const auto state_for_black = Aronda::State::Parser::parseSquare(json, Aronda::State::Player::Black);
        const auto state_for_white = Aronda::State::Parser::parseSquare(json, Aronda::State::Player::White);
        CHECK(state_for_black == conqueredByOtherPlayerSquareState());
        CHECK(state_for_white == conqueredByCurrentPlayerSquareState());
    }
}
