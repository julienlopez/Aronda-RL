#include "parser.hpp"

#include <iostream>

#include <nlohmann_json/json.hpp>

namespace Aronda::State
{

	namespace
	{
		using Row_t = Eigen::Matrix<float, 1, number_of_state_per_square>;

		enum class Player {Black, White};

		Row_t parseSquare(const nlohmann::json& square, const Player current_player)
		{
			if (!square.is_object())
				throw std::runtime_error("Unable to parse json string");
			std::cout << square.dump() << std::endl;
			std::cout << square["conqueringColor"].dump() << std::endl;
			if (!square["conqueringColor"].is_string())
				throw std::runtime_error("Unable to parse json string");
			return {};
		}

		Player currentPlayer(const nlohmann::json& json)
		{
			const std::string current_player = json["currentPlayer"];
			std::cout << current_player << std::endl;
			if (current_player == "WHITE")
				return Player::White;
			return Player::Black;
		}

	}

	State Parser::parse(const std::string& json_string)
	{
		State res;
		const auto json = nlohmann::json::parse(json_string);
		if (!json.is_object())
			throw std::runtime_error("Unable to parse json string");
		const auto current_player = currentPlayer(json);
		const auto it = json.find("squares");
		if(it == end(json) || !it->is_array())
			throw std::runtime_error("Unable to parse json string");
		if(it->size() < res.cols())
			throw std::runtime_error("Unable to parse json string");
		for (int i = 0; i < res.cols(); i++) // TODO better handle multiple center 
		{
			res.row(i) = parseSquare((*it)[i], current_player);
		}
		return res;
	}

}
