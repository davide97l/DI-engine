from easydict import EasyDict
from ding.entry import serial_pipeline

minigrid_ppo_config = dict(
    # exp_name="minigrid_empty8_offppo",
    exp_name="minigrid_fourrooms_offppo",
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        # env_id='MiniGrid-Empty-8x8-v0',
        env_id='MiniGrid-FourRooms-v0',
        n_evaluator_episode=5,
        stop_value=0.96,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=2739,
            action_shape=7,
            encoder_hidden_size_list=[256, 128, 64, 64],
        ),
        learn=dict(
            update_per_collect=4,
            batch_size=64,
            learning_rate=0.0003,
            value_weight=0.5,
            entropy_weight=0.001,
            clip_ratio=0.2,
            adv_norm=False,
        ),
        collect=dict(
            n_sample=128,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
    ),
)
minigrid_ppo_config = EasyDict(minigrid_ppo_config)
main_config = minigrid_ppo_config
minigrid_ppo_create_config = dict(
    env=dict(
        type='minigrid',
        import_names=['dizoo.minigrid.envs.minigrid_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo_offpolicy'),
)
minigrid_ppo_create_config = EasyDict(minigrid_ppo_create_config)
create_config = minigrid_ppo_create_config

if __name__ == "__main__":
    serial_pipeline([main_config, create_config], seed=0)
