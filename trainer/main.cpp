#include "agent.hpp"

#include <iostream>

using Aronda::Trainer::Agent;

const std::size_t TOTAL_EPISODES = 50;

const std::size_t REWARD_TARGET = 10;
const std::size_t BATCH_SIZE_BASELINE = 20; // Averaged over these these many episodes

const std::size_t MEMORY_CAPACITY = 100000;

int main()
{
    Agent agent;

    std::size_t episode_number = 0;
    double reward_sum = 0.;
    while(episode_number < TOTAL_EPISODES)
    {
        reward_sum += agent.run();
        episode_number += 1;
        if(episode_number % BATCH_SIZE_BASELINE == 0)
        {
            std::cout << "Episode: " << episode_number << ", Average reward for episode "
                      << reward_sum / BATCH_SIZE_BASELINE << "." << std::endl;
            if(reward_sum / BATCH_SIZE_BASELINE > REWARD_TARGET)
            {
                std::cout << "Task solved in " << episode_number << " episodes" << std::endl;
                break;
            }
            reward_sum = 0;
        }
    }
    agent.saveModel("dqn.mod");

    return EXIT_SUCCESS;
}
