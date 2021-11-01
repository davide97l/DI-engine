from easydict import EasyDict

hopper_ppo_default_config = dict(
    env=dict(
        env_id='HopperMuJoCoEnv-v0',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=8,
        evaluator_env_num=10,
        use_act_scale=True,
        n_evaluator_episode=10,
        stop_value=3000,
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        model=dict(
            obs_shape=11,
            action_shape=3,
            continuous=True,
        ),
        continuous=True,
        learn=dict(
            epoch_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.0,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
        ),
        collect=dict(
            n_sample=2048,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.97,
        ),
        eval=dict(evaluator=dict(eval_freq=5000, )),
    ),
)
hopper_ppo_default_config = EasyDict(hopper_ppo_default_config)
main_config = hopper_ppo_default_config

hopper_ppo_create_default_config = dict(
    env=dict(
        type='pybullet',
        import_names=['dizoo.pybullet.envs.pybullet_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='ppo',
    ),
)
hopper_ppo_create_default_config = EasyDict(hopper_ppo_create_default_config)
create_config = hopper_ppo_create_default_config
