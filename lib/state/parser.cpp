#include "parser.hpp"

#include <iostream>

#include <nlohmann_json/json.hpp>

#include <gsl/gsl_assert>

#include <unsupported/Eigen/KroneckerProduct>

namespace Aronda::State
{

	namespace
	{
		using Row_t = Eigen::Matrix<float, 1, number_of_state_per_square>;

		enum class Player {Black, White};

		/**
		* @brief returns a 1-hot vector (zero everywhere except for on index set to one) of size 'state_size'
		* @pre index < state_size
		*/
		template<std::size_t state_size>
		Eigen::Matrix<float, 1, state_size> oneHot(const std::size_t index)
		{
			Expects(index < state_size);
			Eigen::Array<float, 1, state_size> res;
			for (std::size_t i = 0; i < state_size; i++)
				res(i) = ((i == index) ? 1.f : 0.f);
			return res.matrix();
		}

		auto stringToPlayer(const std::string& str)
		{
			Expects(str == "WHITE" || str == "BLACK");
			return str == "WHITE" ?  Player::White : Player::Black;
		}

		Eigen::Matrix<float, 1, 5> encodeSquareStonesForPlayer(const nlohmann::json& square, const Player player)
		{
			const auto number_of_black_stones = square["numberOfBlackPawns"].get<int>();
			const auto number_of_white_stones = square["numberOfWhitePawns"].get<int>();
			return oneHot<5>(player == Player::Black ? number_of_black_stones : number_of_white_stones);
		}

		Eigen::Matrix<float, 1, number_of_state_per_square - 2> encodeSquareStones(const nlohmann::json& square, const Player current_player)
		{
			const auto other_player = current_player == Player::Black ? Player::White : Player::Black;
			return Eigen::kroneckerProduct(encodeSquareStonesForPlayer(square, current_player), encodeSquareStonesForPlayer(square, other_player));
		}

		Row_t parseSquare(const nlohmann::json& square, const Player current_player)
		{
			if (!square.is_object())
				throw std::runtime_error("Unable to parse json string");
			std::cout << square.dump() << std::endl;
			std::cout << square["conqueringColor"].dump() << std::endl;
			if (!square["conqueringColor"].is_string())
				throw std::runtime_error("Unable to parse json string");
			const auto conquering_color = square["conqueringColor"];
			if (conquering_color != "null")
			{
				if (stringToPlayer(conquering_color) == current_player)
					return oneHot<number_of_state_per_square>(number_of_state_per_square - 2);
				else
					return oneHot<number_of_state_per_square>(number_of_state_per_square - 1);
			}
			else
			{
				Eigen::Matrix<float, 1, number_of_state_per_square> res;
				res << encodeSquareStones(square, current_player), Eigen::Matrix<float, 1, 2>::Zero();
				return res;
			}
		}

		Player currentPlayer(const nlohmann::json& json)
		{
			return stringToPlayer(json["currentPlayer"]);
		}

	}

	State Parser::parse(const std::string& json_string)
	{
		State res = State::Zero();
		const auto json = nlohmann::json::parse(json_string);
		if (!json.is_object())
			throw std::runtime_error("Unable to parse json string");
		const auto current_player = currentPlayer(json);
		const auto it = json.find("squares");
		if(it == end(json) || !it->is_array())
			throw std::runtime_error("Unable to parse json string");
		const auto squares = it->get<nlohmann::json::array_t>();
		if(squares.size() < res.rows())
			throw std::runtime_error("Unable to parse json string");
		for (int i = 0; i < res.rows(); i++) // TODO better handle multiple center 
		{
			res.row(i) = parseSquare((*it)[i], current_player);
		}
		return res;
	}

}
