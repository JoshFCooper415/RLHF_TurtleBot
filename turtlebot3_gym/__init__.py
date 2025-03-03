from gymnasium.envs.registration import register

register(
    id='TurtleBot3-v0',
    entry_point='turtlebot3_gym.envs:TurtleBot3Env',
    max_episode_steps=500,
)
