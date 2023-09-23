import os
import uuid
import tempfile
import warnings

import torch
from torch import nn
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.data import LazyMemmapStorage, MultiStep, TensorDictReplayBuffer
from torchrl.envs import (
    EnvCreator,
    ExplorationType,
    ParallelEnv,
    RewardScaling,
    StepCounter,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    CatFrames,
    Compose,
    GrayScale,
    ObservationNorm,
    Resize,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.modules import DuelingCnnDQNet, EGreedyWrapper, QValueActor

from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.record.loggers.csv import CSVLogger
from torchrl.trainers import (
    LogReward,
    Recorder,
    ReplayBufferTrainer,
    Trainer,
    UpdateWeights,
)

num_workers = 2  # 8
num_collectors = 2  # 4
device = "cuda" if not torch.cuda.is_available() else "cpu"

init_bias = 2.0
total_frames = 5_000  # 500000
init_random_frames = 100  # 1000
frames_per_batch = 32  # 128
batch_size = 32  # 256
buffer_size = min(total_frames, 100000)

eps_greedy_val = 0.1
eps_greedy_val_env = 0.005

# the learning rate of the optimizer
lr = 2e-3
# weight decay
wd = 1e-5
# the beta parameters of Adam
betas = (0.9, 0.999)
# Optimization steps per batch collected (aka UPD or updates per data)
n_optim = 8
gamma = 0.99
tau = 0.02

def make_env(
    parallel=False,
    obs_norm_sd=None
):
    if obs_norm_sd is None:
        obs_norm_sd = {"standard_normal": True}
    if parallel:
        base_env = ParallelEnv(
            num_workers,
            EnvCreator(
                lambda: GymEnv(
                    "CartPole-v1",
                    from_pixels=True,
                    pixels_only=True, 
                    device=device,
                )
            )
        )
    else:
        base_env = GymEnv(
            "CartPole-v1", 
            from_pixels=True,
            pixels_only = True, 
            device=device
        )
    env = TransformedEnv(
        base_env, 
        Compose(
            StepCounter(),
            ToTensorImage(),
            RewardScaling(loc=0.0, scale=0.1),
            GrayScale(),
            Resize(64, 64),
            CatFrames(4, in_keys=["pixels"], dim=-3),
            ObservationNorm(in_keys=["pixels"], **obs_norm_sd)
        )
    )
    return env

def get_norm_stats():
    test_env = make_env()
    test_env.transform[-1].init_stats(
        num_iter=1000, cat_dim=0, reduce_dim=[-1, -2, -4], keep_dims=(-1, -2)
    )
    obs_norm_sd = test_env.transform[-1].state_dict()
    # let's check that normalizing constants have a size of ``[C, 1, 1]`` where
    # ``C=4`` (because of :class:`~torchrl.envs.CatFrames`).
    print("state dict of the observation norm:", obs_norm_sd)
    return obs_norm_sd

def make_model(dummy_env):
    cnn_kwargs = {
        "num_cells": [32, 64, 64],
        "kernel_sizes": [5, 4, 3], 
        "strides": [2, 2, 1], 
        "activation_class": nn.ELU
    }

    mlp_kwargs = {
        "depth": 2,
        "num_cells": [
            64, 
            64
        ],
        "activation_class": nn.ELU
    }

    net = DuelingCnnDQNet(
        dummy_env.action_spec.sahpe[-1], 1, cnn_kwargs, mlp_kwargs
    ).to(device)

    net.value[-1].bias.data.fill_(init_bias)

    actor = QValueActor(net, in_keys=["pixels"], 
            spec=dummy_env.action_spec
        ).to(device)
    
    # model composed of lazy layers, we must pass the fake 
    # batch of data through it to instantiate them
    tensordict = dummy_env.fake_tensordict()
    actor(tensordict)

    # wrap our actor in EGreedyWrapper for data collection
    actor_explore = EGreedyWrapper(
        actor,
        annealing_num_steps=total_frames,
        eps_init=eps_greedy_val,
        eps_end=eps_greedy_val_env
    )

# Collecting and storing data
def get_replay_buffer(buffer_size, n_optim, batch_size):
    replay_buffer = TensorDictReplayBuffer(
        batch_size,
        storage = LazyMemmapStorage(buffer_size),
        prefetch=n_optim
    )
    return replay_buffer

def get_collector(
    obs_norm_sd,
    num_collectors,
    actor_explore,
    frames_per_batch,
    total_frames,
    device,
):
    data_collector = MultiaSyncDataCollector(
        [
            make_env(parallel=True, obs_norm_sd=obs_norm_sd),
        ]
        * num_collectors,
        policy=actor_explore,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        # this is the default behaviour: the collector runs in ``"random"`` (or explorative) mode
        exploration_type=ExplorationType.RANDOM,
        # We set the all the devices to be identical. Below is an example of
        # heterogeneous devices
        device=device,
        storing_device=device,
        split_trajs=False,
        postproc=MultiStep(gamma=gamma, n_steps=5),
    )
    return data_collector

def get_loss_module(actor, gamma):
    loss_module = DQNLoss(actor, delay_value=True)
    loss_module.make_value_estimator(gamma=gamma)
    target_updater = SoftUpdate(loss_module, eps=0.995)
    return loss_module, target_updater

stats = get_norm_stats()
test_env = make_env(parallel=False, obs_norm_sd=stats)
# Get model
actor, actor_explore = make_model(test_env)
loss_module, target_net_updater = get_loss_module(actor, gamma)

collector = get_collector(
    stats, num_collectors, actor_explore, frames_per_batch, total_frames, device
)
optimizer = torch.optim.Adam(
    loss_module.parameters(), lr=lr, weight_decay=wd, betas=betas
)
exp_name = f"dqn_exp_{uuid.uuid1()}"
tmpdir = tempfile.TemporaryDirectory()
logger = CSVLogger(exp_name=exp_name, log_dir=tmpdir.name)
warnings.warn(f"log dir: {logger.experiment.log_dir}")

log_interval = 500

trainer = Trainer(
    collector=collector,
    total_frames=total_frames,
    frame_skip=1,
    loss_module=loss_module,
    optimizer=optimizer,
    logger=logger,
    optim_steps_per_batch=n_optim,
    log_interval=log_interval,
)

buffer_hook = ReplayBufferTrainer(
    get_replay_buffer(buffer_size, n_optim, batch_size),
    flatten_tensordicts=True
)

buffer_hook.register(trainer)
weight_updater = UpdateWeights(collector, update_weights_interval=1)
weight_updater.register(trainer)
recorder = Recorder(
    record_interval=100,  # log every 100 optimization steps
    record_frames=1000,  # maximum number of frames in the record
    frame_skip=1,
    policy_exploration=actor_explore,
    environment=test_env,
    exploration_type=ExplorationType.MODE,
    log_keys=[("next", "reward")],
    out_keys={("next", "reward"): "rewards"},
    log_pbar=True,
)
recorder.register(trainer)

trainer.register_op("post_optim", target_net_updater.step)

log_reward = LogReward(log_pbar=True)
log_reward.register(trainer)

trainer.train()

def print_csv_files_in_folder(folder_path):
    """
    Find all CSV files in a folder and prints the first 10 lines of each file.

    Args:
        folder_path (str): The relative path to the folder.

    """
    csv_files = []
    output_str = ""
    for dirpath, _, filenames in os.walk(folder_path):
        for file in filenames:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(dirpath, file))
    for csv_file in csv_files:
        output_str += f"File: {csv_file}\n"
        with open(csv_file, "r") as f:
            for i, line in enumerate(f):
                if i == 10:
                    break
                output_str += line.strip() + "\n"
        output_str += "\n"
    print(output_str)


print_csv_files_in_folder(logger.experiment.log_dir)
