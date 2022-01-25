import pytest
import torch
from easydict import EasyDict
from ding.reward_model.gail_irl_model import GailRewardModel
from ding.utils.data import offline_data_save_type
from torch.utils.tensorboard import SummaryWriter
import os

obs_space_1d, obs_space_3d = 4, [4, 84, 84]
expert_data_path_1d, expert_data_path_3d = './expert_data_1d.pkl', './expert_data_3d.pkl'
device = 'cpu'
action_space = 3

cfg1 = dict(
    input_size=obs_space_1d + 1,
    hidden_size=64,
    batch_size=5,
    learning_rate=1e-3,
    update_per_collect=2,
    expert_data_path=expert_data_path_1d,
    ),

cfg2 = dict(
    input_size=obs_space_3d,
    hidden_size=64,
    batch_size=5,
    learning_rate=1e-3,
    update_per_collect=2,
    expert_data_path=expert_data_path_3d,
    action_size=action_space,
    ),

# create fake expert dataset
data_1d = []
for i in range(20):
    d = {}
    d['obs'] = torch.zeros(obs_space_1d)
    d['action'] = torch.Tensor([1.])
    data_1d.append(d)

data_3d = []
for i in range(20):
    d = {}
    d['obs'] = torch.zeros(obs_space_3d)
    d['action'] = torch.Tensor([1.])
    data_3d.append(d)

offline_data_save_type(exp_data=data_1d, expert_data_path=expert_data_path_1d, data_type='naive')
offline_data_save_type(exp_data=data_3d, expert_data_path=expert_data_path_3d, data_type='naive')


@pytest.mark.parametrize('cfg', cfg1)
@pytest.mark.unittest
def test_dataset_1d(cfg):
    data = data_1d
    cfg = EasyDict(cfg)
    policy = GailRewardModel(cfg, device, tb_logger=SummaryWriter())
    policy.load_expert_data()
    assert len(policy.expert_data) == 20
    state = policy.state_dict()
    policy.load_state_dict(state)
    policy.collect_data(data)
    assert len(policy.train_data) == 20
    for _ in range(5):
        policy.train()
    policy.estimate(data)
    assert 'reward' in data[0].keys()
    policy.clear_data()
    assert len(policy.train_data) == 0
    if os.path.exists(expert_data_path_1d):
        os.remove(expert_data_path_1d)


@pytest.mark.parametrize('cfg', cfg2)
@pytest.mark.unittest
def test_dataset_3d(cfg):
    data = data_3d
    cfg = EasyDict(cfg)
    policy = GailRewardModel(cfg, device, tb_logger=SummaryWriter())
    policy.load_expert_data()
    assert len(policy.expert_data) == 20
    state = policy.state_dict()
    policy.load_state_dict(state)
    policy.collect_data(data)
    assert len(policy.train_data) == 20
    for _ in range(5):
        policy.train()
    policy.estimate(data)
    assert 'reward' in data[0].keys()
    policy.clear_data()
    assert len(policy.train_data) == 0
    if os.path.exists(expert_data_path_3d):
        os.remove(expert_data_path_3d)
