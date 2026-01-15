# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a simple environment with DSTAR robot.
It combines the concepts of scene, action, observation and event managers to create an environment.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/03_envs/create_DSTAR_base_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a DSTAR base environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# with
import sys
sys.path.insert(0, r"c:\IsaacLab\SibonitoIsaacLab\scripts\tutorials\02_scene")
from dstar import DSTAR_CFG


@configclass
class DSTARSceneCfg(InteractiveSceneCfg):
    """Configuration for a DSTAR scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation: DSTAR robot
    dstar: ArticulationCfg = DSTAR_CFG.replace(prim_path="{ENV_REGEX_NS}/DSTAR")


@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    # Wheel velocity control (WheelRightMiddle_j, WheelLeftMiddle_j)
    wheel_effort = mdp.JointVelocityActionCfg(
        asset_name="dstar",
        joint_names=["WheelRightMiddle_j", "WheelLeftMiddle_j"],
        scale=10.0,
    )

    # SPRAWL position control (ConnectorRight_j)
    sprawl_effort = mdp.JointPositionActionCfg(
        asset_name="dstar",
        joint_names=["ConnectorRight_j"],
        scale=5.0,
    )

    # FBEM leg position control (LegRight_j, LegLeft_j)
    fbem_effort = mdp.JointPositionActionCfg(
        asset_name="dstar",
        joint_names=["LegRight_j", "LegLeft_j"],
        scale=5.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Wheel joint velocities
        wheel_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("dstar", joint_names=["WheelRightMiddle_j", "WheelLeftMiddle_j"])},
        )

        # SPRAWL joint position and velocity
        sprawl_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("dstar", joint_names=["ConnectorRight_j"])},
        )
        sprawl_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("dstar", joint_names=["ConnectorRight_j"])},
        )

        # FBEM leg positions and velocities
        fbem_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("dstar", joint_names=["LegRight_j", "LegLeft_j"])},
        )
        fbem_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("dstar", joint_names=["LegRight_j", "LegLeft_j"])},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # on startup: randomize passive joint masses slightly
    add_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("dstar", body_names=["Body"]),
            "mass_distribution_params": (0.9, 1.1),  # ±10% mass variation
            "operation": "scale",
        },
    )

    # on reset: randomize wheel joint positions/velocities
    reset_wheel_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "dstar", joint_names=["WheelRightMiddle_j", "WheelLeftMiddle_j"]
            ),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.01, 0.01),
        },
    )

    # on reset: randomize SPRAWL joint position/velocity
    reset_sprawl_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("dstar", joint_names=["ConnectorRight_j"]),
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )

    # on reset: randomize FBEM leg positions/velocities
    reset_fbem_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("dstar", joint_names=["LegRight_j", "LegLeft_j"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class DSTAREnvCfg(ManagerBasedEnvCfg):
    """Configuration for the DSTAR environment."""

    # Scene settings
    scene = DSTARSceneCfg(num_envs=16, env_spacing=1)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [-3.5, 0.0, 3.2]
        self.viewer.lookat = [0.0, 0.0, 0.5]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz


def main():
    """Main function."""
    # parse the arguments
    env_cfg = DSTAREnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting DSTAR environment...")
            # sample random actions
            # actions: [wheel_right, wheel_left, sprawl, fbem_right, fbem_left]
            joint_efforts = torch.randn_like(env.action_manager.action)
            efforts = env.scene["dstar"].data.applied_torque
            print("efforts :", efforts)

            # step the environment
            obs, _ = env.step(joint_efforts)
            # print current wheel velocities and sprawl position
            print(f"[Env 0]: Wheel vel: {obs['policy'][0][0:2]}, Sprawl pos: {obs['policy'][0][2].item():.3f}")
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
