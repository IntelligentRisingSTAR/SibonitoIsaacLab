# filepath: c:\IsaacLab\SibonitoIsaacLab\scripts\tutorials\01_assets\add_new_dstar.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom DSTAR robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Path to the custom USD robot provided by the user
DSTAR_USD_PATH = r"C:\Users\sibonito\OneDrive - post.bgu.ac.il\Desktop\All Robot New Work\URDF\URDFs\DSTAR2.USD"

# Define the DSTAR robot configuration
DSTAR_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=DSTAR_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=300.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=1
        ),
    ),
    # Provide initial joint positions (assumes numbered joints joint1..joint6 and an optional grip joint)
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "ConnectorLeft_j": 0.0,
            "BarLeftBack_j": 0.0,
            "BarLeftFront_j": 0.0,
            "LegLeft_j": 0.0, # FBEM joint
            "WheelLeftBack_j": 0.0,
            "WheelLeftFront_j": 0.0,
            "WheelLeftMiddle_j": 0.0,
            "ConnectorRight_j": 0.0, # SPRAWL joint
            "BarRightBack_j": 0.0,
            "BarRightFront_j": 0.0,
            "LegRight_j": 0.0, # FBEM joint
            "WheelRightBack_j": 0.0,
            "WheelRightFront_j": 0.0,
            "WheelRightMiddle_j": 0.0, # Wheel joint
        },
        pos=(0.0, 0.0, 0.02),
    ),

    # Actuators configuration inferred from DSTARa.USDA naming conventions (modify expressions to match actual names)
    actuators={
        # SPRAWL actuator   
        "Sprawl_joint": ImplicitActuatorCfg(
            joint_names_expr=["ConnectorRight_j"],
            effort_limit_sim=500.0,
            velocity_limit_sim=2.0,
            stiffness=10.0,
            damping=0.1,
        ),

        # FBEM actuator
        "FBEM_joints": ImplicitActuatorCfg(
            joint_names_expr=["LegRight_j", "LegLeft_j"],
            effort_limit_sim=50.0,
            velocity_limit_sim=2.0,
            stiffness=10.0,
            damping=0.1,
        ),
        # Wheels actuator
        "Wheels_joints": ImplicitActuatorCfg(
            joint_names_expr=["WheelRightMiddle_j", "WheelLeftMiddle_j"],
            effort_limit_sim=50.0,
            velocity_limit_sim=5.0,
            stiffness=0,
            damping=10,
        ),
        # Fallback actuator to match any remaining joints
        "passive_joints": ImplicitActuatorCfg(joint_names_expr=["ConnectorLeft_j","BarLeftBack_j","BarLeftFront_j","WheelLeftBack_j","WheelLeftFront_j","BarRightBack_j","BarRightFront_j","WheelRightBack_j","WheelRightFront_j",],
                                              effort_limit_sim=5.0, stiffness=0.0, damping=0.0),
    },
)

DOFBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Yahboom/Dofbot/dofbot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
        },
        pos=(0.25, -0.25, 0.0),
    ),
    actuators={
        "front_joints": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "joint3_act": ImplicitActuatorCfg(
            joint_names_expr=["joint3"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "joint4_act": ImplicitActuatorCfg(
            joint_names_expr=["joint4"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
    },
)

def apply_joint_command(scene, articulation_name: str, joints, cmd_type: str, value, sim_time: float = None):
    """
    Apply a simple joint command to an articulation.

    - articulation_name: key used in scene (e.g. "DSTAR")
    - joints: single joint name or list/tuple of joint names
    - cmd_type: 'pos' or 'vel'
    - value: scalar or callable(sim_time) -> scalar. If a list/array provided it will be used directly for matched joints.
    - sim_time: current simulation time (optional, used when value is callable)
    """
    if isinstance(joints, (str,)):
        joint_list = [joints]
    else:
        joint_list = list(joints)

    art = scene[articulation_name]
    joint_names = list(art.data.joint_names)

    if cmd_type.lower().startswith("v"):
        arr = art.data.default_joint_vel.clone()
        for j in joint_list:
            if j in joint_names:
                idx = joint_names.index(j)
                v = value(sim_time) if callable(value) else value
                arr[:, idx] = v
            else:
                print(f"[WARN] joint not found for velocity command: '{j}'")
        art.set_joint_velocity_target(arr)
        return arr

    # default -> position
    arr = art.data.default_joint_pos.clone()
    for j in joint_list:
        if j in joint_names:
            idx = joint_names.index(j)
            v = value(sim_time) if callable(value) else value
            arr[:, idx] = v
        else:
            print(f"[WARN] joint not found for position command: '{j}'")
    art.set_joint_position_target(arr)
    return arr


class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Robots: replace Jetbot with DSTAR and keep Dofbot as example
    DSTAR = DSTAR_CONFIG.replace(prim_path="{ENV_REGEX_NS}/DSTAR")
    Dofbot = DOFBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Dofbot")

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # do a clean reset: physics first, then recreate scene data, then write states
            sim.reset()        # reset physics / integrator state
            scene.reset()      # recreate scene and articulation .data

            # recompute default root states (one per env) and push them into the sim
            root_dstar_state = scene["DSTAR"].data.default_root_state.clone()
            root_dstar_state[:, :3] += scene.env_origins
            root_dofbot_state = scene["Dofbot"].data.default_root_state.clone()
            root_dofbot_state[:, :3] += scene.env_origins

            scene["DSTAR"].write_root_pose_to_sim(root_dstar_state[:, :7])
            scene["DSTAR"].write_root_velocity_to_sim(root_dstar_state[:, 7:])
            scene["Dofbot"].write_root_pose_to_sim(root_dofbot_state[:, :7])
            scene["Dofbot"].write_root_velocity_to_sim(root_dofbot_state[:, 7:])

            # joints: write each articulation's default pos/vel with separate arrays
            dstar_pos, dstar_vel = (
                scene["DSTAR"].data.default_joint_pos.clone(),
                scene["DSTAR"].data.default_joint_vel.clone(),
            )
            scene["DSTAR"].write_joint_state_to_sim(dstar_pos, dstar_vel)
            # ensure controllers/targets match the written state so they don't fight the reset
            scene["DSTAR"].set_joint_position_target(dstar_pos)
            scene["DSTAR"].set_joint_velocity_target(dstar_vel * 0.0)  # zero velocities as start

            dofbot_pos, dofbot_vel = (
                scene["Dofbot"].data.default_joint_pos.clone(),
                scene["Dofbot"].data.default_joint_vel.clone(),
            )
            scene["Dofbot"].write_joint_state_to_sim(dofbot_pos, dofbot_vel)
            scene["Dofbot"].set_joint_position_target(dofbot_pos)
            scene["Dofbot"].set_joint_velocity_target(dofbot_vel * 0.0)

            joint_names = list(scene["DSTAR"].data.joint_names)
            print("[INFO]: Resetting DSTAR and Dofbot state...")

            # push written data to sim and step a few frames to settle contacts/penetrations
            scene.write_data_to_sim()
            settle_steps = 8  # increase settle for stability
            for _ in range(settle_steps):
                 sim.step()
                 scene.update(sim_dt)
                 # advance sim_time / count to keep timing consistent
                 sim_time += sim_dt
                 count += 1
            # avoid immediate re-trigger on next loop iteration
            count = 1

        # # DSTAR driving forward by setting wheel velocities
        # apply_joint_command(scene, "DSTAR", ["WheelRightMiddle_j", "WheelLeftMiddle_j"], "vel", 100, sim_time)

        # # apply position to sprawl (ConnectorRight_j) using a sinusoid

        # apply_joint_command(scene, "DSTAR", "ConnectorRight_j", "pos", 1, sim_time)

        # # apply position sinusoid to FBEM legs (two joints)
        # apply_joint_command(scene, "DSTAR", ["LegRight_j", "LegLeft_j"], "pos", lambda t: 10 * np.sin(2 * np.pi * 1.0 * t), sim_time)



        joint_names = list(scene["DSTAR"].data.joint_names)
        joint_pos, joint_vel = (scene["DSTAR"].data.default_joint_pos.clone(),
                                scene["DSTAR"].data.default_joint_vel.clone(),)
                                
        # Set wheel actuators (drives) velocity to 1.50 for all sim
        wheel_joint_names = ["WheelRightMiddle_j", "WheelLeftMiddle_j"]
        for wjname in wheel_joint_names:
            if wjname in joint_names:
                idx = joint_names.index(wjname)
                joint_vel[:, idx] = 5.0  # set wheel velocity
            else:
                print(f"[WARN] wheel joint not found in scene joints: '{wjname}'")
        scene["DSTAR"].set_joint_velocity_target(joint_vel)
        # print("final joint velocities:", joint_vel)

        # Set SPRAWL and FBEM joint positions to do a periodic motion
        sprawl_joint_name = "ConnectorRight_j"
        if sprawl_joint_name in joint_names:
            idx = joint_names.index(sprawl_joint_name)
            joint_pos[:, idx] = 1 * np.sin(2 * np.pi * 1.0 * sim_time)
        else:
            print(f"[WARN] sprawl joint not found in scene joints: '{sprawl_joint_name}'")
        
        fbem_joint_names = ["LegRight_j", "LegLeft_j"]
        for ljname in fbem_joint_names:
            if ljname in joint_names:
                idx = joint_names.index(ljname)
                joint_pos[:, idx] = 0.5 * np.sin(2 * np.pi * (1.0 * sim_time + 1.0))  # FBEM leg motion
            else:
                print(f"[WARN] FBEM leg joint not found in scene joints: '{ljname}'")
        # print("final joint positions:", joint_pos)
        # print("joint names:", joint_names)
        scene["DSTAR"].set_joint_position_target(joint_pos)
        # Dofbot waving as before
        wave_action = scene["Dofbot"].data.default_joint_pos
        wave_action[:, 0:4] = 0.25 * np.sin(2 * np.pi * 1.0 * sim_time)
        scene["Dofbot"].set_joint_position_target(wave_action)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()