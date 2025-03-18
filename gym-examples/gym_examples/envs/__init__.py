from gym.envs.registration import register

register(
    id='gym_examples/GridWorld-v0',
    entry_point='gym_examples.envs.grid_world:GridWorldEnv',
    max_episode_steps=300,
)

register(
    id='gym_examples/MRTAWorld-v0',
    entry_point='gym_examples.envs.mrta_world:MRTAWorldEnv',
    max_episode_steps=300,
)
