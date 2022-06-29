import gym
import gym_miniworld
import math


def init_env(string):
    """
    Initializes the environment with seed 0 and inf. steps
    """
    env = gym.make(string)
    env.seed(0)
    env.max_episode_steps = math.inf
    env.reset()

    return env


def print_env_parameters(env):
    """
    Prints several environment variables to the command line.
    """
    print("Agent variables")
    print(
        "• current-pos: (%.2f, %.2f, %.2f)\n• current-angle: %d\n• steps: %d"
        % (*env.agent.pos, int(env.agent.dir * 180 / math.pi) % 360, env.step_count)
    )
    print("• observation-space:", env.observation_space)
    print("• action-space:", env.action_space)
