#include "parser.hpp"

#include <eigen3/unsupported/Eigen/KroneckerProduct>

namespace Aronda::State
{

namespace
{

    auto stringToPlayer(const std::string& str)
    {
        Expects(str == "WHITE" || str == "BLACK");
        return str == "WHITE" ? Player::White : Player::Black;
    }

    template <class T> auto getOrThrow(const nlohmann::json& json, const std::string& key)
    {
        const auto it = json.find(key);
        if(it == end(json)) throw std::runtime_error(("Unable to find " + key + " in " + json.dump()).c_str());
        return it->get<T>();
    }

    Utils::Matrix<1, 5> encodeSquareStonesForPlayer(const nlohmann::json& square, const Player player)
    {
        const auto number_of_black_stones = getOrThrow<int>(square, "numberOfBlackPawns");
        const auto number_of_white_stones = getOrThrow<int>(square, "numberOfWhitePawns");
        return Utils::oneHotRow<5>(player == Player::Black ? number_of_black_stones : number_of_white_stones);
    }

    Utils::Matrix<1, number_of_state_per_square - 2> encodeSquareStones(const nlohmann::json& square,
                                                                        const Player current_player)
    {
        const auto other_player = current_player == Player::Black ? Player::White : Player::Black;
        return Eigen::kroneckerProduct(encodeSquareStonesForPlayer(square, current_player),
                                       encodeSquareStonesForPlayer(square, other_player));
    }

    Player currentPlayer(const nlohmann::json& json)
    {
        return stringToPlayer(getOrThrow<std::string>(json, "currentPlayer"));
    }

    boost::optional<Player> winner(const nlohmann::json& json)
    {
        const auto winner_str = getOrThrow<std::string>(json, "winner");
        if(winner_str == "BLACK") return Player::Black;
        if(winner_str == "WHITE") return Player::White;
        return boost::none;
    }
}

GameState Parser::parse(const std::string& json_string)
{
    Board res = Board::Zero();
    const auto json = nlohmann::json::parse(json_string);
    if(!json.is_object()) throw std::runtime_error("Unable to parse json string");
    const auto current_player = currentPlayer(json);
    const auto squares = getOrThrow<nlohmann::json::array_t>(json, "squares");
    if(squares.size() < res.rows()) throw std::runtime_error("Unable to parse json string");
    for(int i = 0; i < res.rows(); i++) // TODO better handle multiple center
    {
        res.row(i) = parseSquare(squares[i], current_player);
    }
    return {current_player, res, winner(json)};
}

Square Parser::parseSquare(const nlohmann::json& square, const Player current_player)
{
    if(!square.is_object()) throw std::runtime_error("Unable to parse json string");
    const auto conquering_color = getOrThrow<std::string>(square, "conqueringColor");
    if(conquering_color != "null")
    {
        if(stringToPlayer(conquering_color) == current_player)
            return Utils::oneHotRow<number_of_state_per_square>(number_of_state_per_square - 2);
        else
            return Utils::oneHotRow<number_of_state_per_square>(number_of_state_per_square - 1);
    }
    else
    {
        Utils::Matrix<1, number_of_state_per_square> res;
        res << encodeSquareStones(square, current_player), Utils::Matrix<1, 2>::Zero();
        return res;
    }
}
}
