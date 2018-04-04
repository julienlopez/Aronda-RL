#include "agent.hpp"

#include "jaronda.hpp"

#include "state/parser.hpp"

#include <iostream>

#include <numeric_range.hpp>

using Aronda::Trainer::Agent;
using Aronda::State::Player;
using Aronda::Trainer::JAronda;

using AgentContainer_t = std::map<Player, std::unique_ptr<Agent>>;

const std::size_t BATCH_SIZE_BASELINE = 20; 
const std::size_t TOTAL_EPISODES = 1000;

struct GameResult
{
    std::size_t number_of_moves;
};

GameResult playGame(AgentContainer_t& agents)
{
    double R = 0.;
    JAronda game{ "11815" };
    
    std::size_t move_number = 0;
    auto s = game.begin();
    while (true)
    {
        auto& agent = agents[Player::Black];
        const auto a = agent->act(s);
        auto res = game.play(s, a);

        agent->observe({ s, a, res.reward, res.new_state });
        agent->replay();

        if (!res.new_state)
            return { move_number };

        s = *res.new_state;
        move_number++;
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
        for(const auto episode_number : range(TOTAL_EPISODES))
        {
            const auto res = playGame(agents);
            if(episode_number % BATCH_SIZE_BASELINE == 0)
            {
                std::cout << episode_number << ", " << res.number_of_moves << std::endl;
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
