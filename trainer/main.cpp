#include "agent.hpp"

#include <iostream>

using Aronda::Trainer::Agent;

const std::size_t TOTAL_EPISODES = 1000;

const std::size_t REWARD_TARGET = 10;
const std::size_t BATCH_SIZE_BASELINE = 20; // Averaged over these these many episodes

const std::size_t MEMORY_CAPACITY = 100000;

int main()
{
    try
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
        std::cout << "final qmap for beginstate = " << agent.test().transpose() << std::endl;
        agent.saveModel("dqn.mod");
    }
    catch(const std::exception& ex)
    {
        std::wcerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
