#include "agent.hpp"

#include "jaronda.hpp"

#include "state/parser.hpp"

#include <iostream>

#include <numeric_range.hpp>

using Aronda::State::Player;
using Aronda::Trainer::Agent;
using Aronda::Trainer::JAronda;

using AgentContainer_t = std::map<Player, std::unique_ptr<Agent>>;

const std::size_t TOTAL_EPISODES = 1000000;

namespace
{

struct GameResult
{
    std::size_t number_of_moves;
    boost::optional<Player> winner;
    double epsilon;
    std::string moves;
};

std::string printWinner(boost::optional<Player> winner)
{
    if(!winner) return " ";
    return *winner == Player::Black ? "B" : "W";
}

GameResult playGame(AgentContainer_t& agents)
{
    double R = 0.;
    JAronda game{"11815"};

    std::size_t move_number = 0;
    auto s = game.begin();
    std::string moves;
    while(true)
    {
        auto& agent = agents[s.current_player];
        const auto a = agent->act(s.board);

        // std::cout << printWinner(s.current_player) << " plays " << a << std::endl;
        moves += printWinner(s.current_player) + std::to_string(a) + ';';

        auto res = game.play(s.board, a);

        if(res.new_state)
            agent->observe({s.board, a, res.reward, res.new_state->board});
        else
            agent->observe({s.board, a, res.reward, boost::none});
        agent->replay();

        if(!res.new_state) return {move_number, s.winner, agent->epsilon(), moves};

        s = *res.new_state;
        move_number++;
    }
}
}

int main()
{
    try
    {
        AgentContainer_t agents;
        agents[Player::Black] = std::make_unique<Agent>();
        agents[Player::White] = std::make_unique<Agent>();

        std::size_t episode_number = 0;
        std::size_t max_number_of_moves = 0;
        for(const auto episode_number : range(TOTAL_EPISODES))
        {
            const auto res = playGame(agents);
            std::cout << episode_number << "," << res.number_of_moves << "," << printWinner(res.winner) << ","
                      << res.moves << std::endl;
            if(res.number_of_moves > max_number_of_moves)
            {
                if(res.number_of_moves > 30)
                {
                    agents.at(Player::Black)->saveModel("black-dqn-" + std::to_string(res.number_of_moves) + ".mod");
                    agents.at(Player::White)->saveModel("white-dqn-" + std::to_string(res.number_of_moves) + ".mod");
                }
                max_number_of_moves = res.number_of_moves;
            }
        }
        agents.at(Player::Black)->saveModel("black-dqn.mod");
        agents.at(Player::White)->saveModel("white-dqn.mod");
    }
    catch(const std::exception& ex)
    {
        std::wcerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
