# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tutorial: Interactive scene with DSTAR robot (14-DOF wheeled robot with SPRAWL, FBEM, and wheel actuators).

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/02_scene/create_scene_DSTAR.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface with DSTAR robot.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from dstar import DSTAR_CFG  # isort:skip



@configclass
class DSTARSceneCfg(InteractiveSceneCfg):
    """Configuration for a DSTAR scene with ground, lighting, and DSTAR robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation: DSTAR robot
    dstar: ArticulationCfg = DSTAR_CFG.replace(prim_path="{ENV_REGEX_NS}/DSTAR")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop with DSTAR robot control."""
    # Extract scene entities
    robot = scene["dstar"]
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    # Get joint names for reference
    joint_names = list(robot.data.joint_names)
    print(f"[INFO] DSTAR joint names: {joint_names}")
    print(f"[INFO] Total joints: {len(joint_names)}")
    
    # Simulation loop
    while simulation_app.is_running():
        # Reset every 2000 steps
        if count % 2000 == 0:
            count = 0
            
            # Reset root state: position + velocity
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins  # offset by environment origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            
            # Reset joint state: positions + velocities
            joint_pos, joint_vel = (
                robot.data.default_joint_pos.clone(),
                robot.data.default_joint_vel.clone(),
            )
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
            # Sync controllers/targets with written state
            robot.set_joint_position_target(joint_pos)
            robot.set_joint_velocity_target(joint_vel * 0.0)  # zero velocities
            
            # Write data and settle
            scene.write_data_to_sim()
            settle_steps = 8
            for _ in range(settle_steps):
                sim.step()
                scene.update(sim_dt)
                sim_time += sim_dt
                count += 1
            count = 1
            print("[INFO]: Resetting DSTAR robot state...")
            continue
        
        # Apply periodic commands to DSTAR actuators
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        
        # SPRAWL actuator: sinusoidal motion on ConnectorRight_j
        if "ConnectorRight_j" in joint_names:
            idx = joint_names.index("ConnectorRight_j")
            joint_pos[:, idx] = 1.0 * np.sin(2 * np.pi * 1.0 * sim_time)
        
        # FBEM actuators: sinusoidal motion on LegRight_j and LegLeft_j
        for leg_joint in ["LegRight_j", "LegLeft_j"]:
            if leg_joint in joint_names:
                idx = joint_names.index(leg_joint)
                joint_pos[:, idx] =  -0.5 * np.sin(2 * np.pi * 1.0 * sim_time + np.pi)
        
        # Wheel actuators: constant velocity on WheelRightMiddle_j and WheelLeftMiddle_j
        for wheel_joint in ["WheelRightMiddle_j", "WheelLeftMiddle_j"]:
            if wheel_joint in joint_names:
                idx = joint_names.index(wheel_joint)
                joint_vel[:, idx] = 5.0  # wheel velocity target
        
        # Apply targets
        robot.set_joint_position_target(joint_pos)
        robot.set_joint_velocity_target(joint_vel)
        
        # Write data to simulator
        scene.write_data_to_sim()
        
        # Perform step
        sim.step()
        
        # Increment counter and time
        count += 1
        sim_time += sim_dt
        
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Design scene
    scene_cfg = DSTARSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.7)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
