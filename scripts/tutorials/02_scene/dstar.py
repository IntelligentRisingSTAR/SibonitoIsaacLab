# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the DSTAR robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

DSTAR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=r"C:\Users\sibonito\OneDrive - post.bgu.ac.il\Desktop\All Robot New Work\URDF\URDFs\DSTAR2.USD",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=300.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "ConnectorLeft_j": 0.0,
            "BarLeftBack_j": 0.0,
            "BarLeftFront_j": 0.0,
            "LegLeft_j": 0.0,  # FBEM joint
            "WheelLeftBack_j": 0.0,
            "WheelLeftFront_j": 0.0,
            "WheelLeftMiddle_j": 0.0,
            "ConnectorRight_j": 0.0,  # SPRAWL joint
            "BarRightBack_j": 0.0,
            "BarRightFront_j": 0.0,
            "LegRight_j": 0.0,  # FBEM joint
            "WheelRightBack_j": 0.0,
            "WheelRightFront_j": 0.0,
            "WheelRightMiddle_j": 0.0,  # Wheel joint
        },
        pos=(0.0, 0.0, 0.02),
    ),
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
        # Passive joints (bars, connectors, other wheels)
        "passive_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "ConnectorLeft_j",
                "BarLeftBack_j",
                "BarLeftFront_j",
                "WheelLeftBack_j",
                "WheelLeftFront_j",
                "BarRightBack_j",
                "BarRightFront_j",
                "WheelRightBack_j",
                "WheelRightFront_j",
            ],
            effort_limit_sim=5.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for the DSTAR robot with SPRAWL, FBEM, and wheel actuators."""
