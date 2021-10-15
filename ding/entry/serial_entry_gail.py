import copy

from ding.policy.base_policy import Policy
from typing import Union, Optional, List, Any, Tuple
import os
import torch
import logging
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from ding.config import read_config, compile_config
from ding.policy import create_policy, PolicyFactory
from ding.utils import set_pkg_seed
from ding.entry import collect_demo_data, serial_pipeline, serial_pipeline_reward_model

from easydict import EasyDict
expert_config = dict(
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
        replay_path='cartpole_dqn/video',
    ),
    policy=dict(
        load_path='cartpole_dqn/ckpt/ckpt_best.pth.tar',
        cuda=False,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        nstep=1,
        discount_factor=0.97,
        learn=dict(
            batch_size=64,
            learning_rate=0.001,
        ),
        collect=dict(n_sample=8),
        eval=dict(evaluator=dict(eval_freq=40, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
                collect=0.1
            ),
            replay_buffer=dict(replay_buffer_size=20000, ),
        ),
    ),
)
expert_config = EasyDict(expert_config)

cartpole_dqn_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
)
expert_create_config = EasyDict(cartpole_dqn_create_config)

expert_config = (expert_config, expert_create_config)


def serial_pipeline_gail(
        input_cfg: Union[str, Tuple[dict, dict]],
        expert_cfg: Union[str, Tuple[dict, dict]] = expert_config,
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_iterations: Optional[int] = int(1e10),
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline sqil entry: we create this serial pipeline in order to\
            implement SQIL in DI-engine. For now, we support the following envs\
            Cartpole, Lunarlander, Pong, Spaceinvader, Qbert. The demonstration\
            data come from the expert model. We use a well-trained model to \
            generate demonstration data online
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - expert_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_iterations (:obj:`Optional[torch.nn.Module]`): Learner's max iteration. Pipeline will stop \
            when reaching this iteration.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    if isinstance(expert_cfg, str):
        expert_cfg, expert_create_cfg = read_config(expert_cfg)
    else:
        expert_cfg, expert_create_cfg = expert_cfg
    expert_cfg.expert_data.expert_data_path = os.path.join(expert_cfg.exp_name, expert_cfg.expert_data.expert_data_path)
    collect_demo_data((expert_cfg, expert_create_cfg), seed, state_dict_path=expert_cfg.policy.load_path,
                      expert_data_path=cfg.expert_data.expert_data_path, collect_count=cfg.expert_data.collect_count)
    return serial_pipeline_reward_model((cfg, create_cfg), seed=seed, env_setting=env_setting, model=model,
                                        max_iterations=max_iterations)
