# coding=utf-8
# Copyright 2022 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ravens tasks."""

from ravens.tasks.align_box_corner import AlignBoxCorner
from ravens.tasks.assembling_kits import AssemblingKits
from ravens.tasks.assembling_kits import AssemblingKitsEasy
from ravens.tasks.block_insertion import BlockInsertion
from ravens.tasks.block_insertion import BlockInsertionEasy
from ravens.tasks.block_insertion import BlockInsertionNoFixture
from ravens.tasks.block_insertion import BlockInsertionSixDof
from ravens.tasks.block_insertion import BlockInsertionTranslation
from ravens.tasks.manipulating_rope import ManipulatingRope
from ravens.tasks.packing_boxes import PackingBoxes
from ravens.tasks.palletizing_boxes import PalletizingBoxes
from ravens.tasks.place_red_in_green import PlaceRedInGreen
from ravens.tasks.stack_block_pyramid import StackBlockPyramid
from ravens.tasks.sweeping_piles import SweepingPiles
from ravens.tasks.task import Task
from ravens.tasks.towers_of_hanoi import TowersOfHanoi
from gym import register

names = {
    'align-box-corner': AlignBoxCorner,
    'assembling-kits': AssemblingKits,
    'assembling-kits-easy': AssemblingKitsEasy,
    'block-insertion': BlockInsertion,
    'block-insertion-easy': BlockInsertionEasy,
    'block-insertion-nofixture': BlockInsertionNoFixture,
    'block-insertion-sixdof': BlockInsertionSixDof,
    'block-insertion-translation': BlockInsertionTranslation,
    'manipulating-rope': ManipulatingRope,
    'packing-boxes': PackingBoxes,
    'palletizing-boxes': PalletizingBoxes,
    'place-red-in-green': PlaceRedInGreen,
    'stack-block-pyramid': StackBlockPyramid,
    'sweeping-piles': SweepingPiles,
    'towers-of-hanoi': TowersOfHanoi
}


def make_env(env_name, mode="train", obs_mode="rgb"):
    from ravens.environments.environment import ContinuousEnvironment
    import os.path as osp
    env_cls = ContinuousEnvironment
    
    __this_folder__ = osp.dirname(osp.abspath(__file__))
    assets_root = osp.join(__this_folder__, '..', 'environments', 'assets')
    env = env_cls(assets_root, disp=False, shared_memory=False, hz=480, obs_mode=obs_mode)
    task = names[env_name](continuous=True)
    task.mode = mode
    env.set_task(task)
    return env


def register_all():
    for env_name in names:
        register(
            id=f"{env_name}-v0",
            entry_point=make_env,
            max_episode_steps=20,
            kwargs={
                'env_name': env_name,
            }
        )
