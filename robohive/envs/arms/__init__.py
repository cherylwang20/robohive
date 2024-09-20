""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

from gymnasium.envs.registration import register
from robohive.envs.env_variants import register_env_variant
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))

print("RoboHive:> Registering Arms Envs")

# FRANKA REACH =======================================================================
from robohive.envs.arms.reach_base_v0 import ReachBaseV0


# Reach to fixed target
register(
    id='FrankaReachFixed-v0',
    entry_point='robohive.envs.arms.reach_base_v0:ReachBaseV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_reach_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_reach_v0.config',
        'robot_site_name': "end_effector",
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.2, 0.3, 1.2], 'low':[0.2, 0.3, 1.2]}
    }
)

# reach_base_v1 is env with five object but distance based reward
# reach_base_v2 is the env with pick & place
# reach_base_v3 is the env for two objects reaching and grabbing with masking
# reach_base_v4 is the env with five object with masking based reward

register(
    id='UR10eReachFixed-v0',
    entry_point='robohive.envs.arms.reach_base_v1:ReachBaseV0',
    max_episode_steps=200, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_gripper.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'obj_xyz_range': {'high':[0.235, 0.5, 0.86], 'low':[-0.235, 0.4, 0.86]},
        'target_site_name': "obj0",
        'target_xyz_range': {'high':[0.435, 0.5, 0.86], 'low':[-0.435, 0.4, 0.86]}
    }
)

#reach and pick env for block object
register(
    id='UR10eReachFixed-v1',
    entry_point='robohive.envs.arms.reach_base_v2:ReachBaseV0',
    max_episode_steps=150, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_gripper.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "obj0",
        'obj_xyz_range': {'high':[0.235, 0.5, 0.86], 'low':[-0.1, 0.4, 0.86]},
        'goal_site_name': "pick_target",
        'target_xyz_range': {'high':[0.435, 0.5, 0.86], 'low':[-0.435, 0.4, 0.86]}
    }
)


#env for chemical objects, for picking and lifting
register(
    id='UR10eReachFixed-v2',
    entry_point='robohive.envs.arms.reach_base_v2:ReachBaseV0',
    max_episode_steps=500, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_chem_vel.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        'obj_xyz_range': {'high':[0.05, 0.5, 0.895831], 'low':[-0., 0.5 ,0.895831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[0.435, 0.5, 0.9], 'low':[-0.435, 0.4, 0.9]}
    }
)

##env training for reaching and touching for two beakers
register(
    id='UR10eReachFixed-v3',
    entry_point='robohive.envs.arms.reach_base_v3:ReachBaseV0',
    max_episode_steps=200, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_chem_vel.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)

#this env is for testing for two rbf
register(
    id='UR10eReachFixed-v4',
    entry_point='robohive.envs.arms.reach_base_v3:ReachBaseV0',
    max_episode_steps=200, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_chem.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)

#this env is for testing for two squre blocks
register(
    id='UR10eReachFive-v0',
    entry_point='robohive.envs.arms.reach_base_v1:ReachBaseV0',
    max_episode_steps=200, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_five_obj.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        #'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)

#this is for reaching with three channel, no masking (distance based)
register(
    id='UR10eReach3C-v0',
    entry_point='robohive.envs.arms.reach_3d_v0:ReachBaseV0',
    max_episode_steps=150, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_five_obj.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        #'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)

register(
    id='UR10eEvalReach3C-v0',
    entry_point='robohive.envs.arms.reach_3d_v0:ReachBaseV0',
    max_episode_steps=150, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_eval.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        #'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)

register(
    id='UR10eReach4C-v0',
    entry_point='robohive.envs.arms.reach_4d_v0:ReachBaseV0',
    max_episode_steps=150, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_five_obj.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        #'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)

register(
    id='UR10eEvalReach4C-v0',
    entry_point='robohive.envs.arms.reach_4d_v0:ReachBaseV0',
    max_episode_steps=150, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_eval.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        #'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)

register(
    id='UR10eReach4C-v1',
    entry_point='robohive.envs.arms.reach_4d_v1:ReachBaseV0',
    max_episode_steps=150, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_five_obj.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        #'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)

register(
    id='UR10eEvalReach4C-v1',
    entry_point='robohive.envs.arms.reach_4d_v1:ReachBaseV0',
    max_episode_steps=150, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_eval.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        #'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)


register(
    id='UR10eReach7C-v1',
    entry_point='robohive.envs.arms.reach_7d_v1:ReachBaseV0',
    max_episode_steps=150, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_five_obj.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        #'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)

register(
    id='UR10eReach1H-v0',
    entry_point='robohive.envs.arms.reach_1h_v0:ReachBaseV0',
    max_episode_steps=150, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_five_obj.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        #'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)

register(
    id='UR10eEvalReach1H-v0',
    entry_point='robohive.envs.arms.reach_1h_v0:ReachBaseV0',
    max_episode_steps=150, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_eval.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        #'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)

##this is for sparse env
register(
    id='UR10eSparse3C-v0',
    entry_point='robohive.envs.arms.sparse_3d_v0:ReachBaseV0',
    max_episode_steps=200, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_five_obj.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        #'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)

register(
    id='UR10eSparse4C-v0',
    entry_point='robohive.envs.arms.sparse_4d_v0:ReachBaseV0',
    max_episode_steps=200, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_five_obj.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        #'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)


#this is for reaching with three channel, with masking (pixel based)
register(
    id='UR10eMask3C-v0',
    entry_point='robohive.envs.arms.mask_3d_v0:ReachBaseV0',
    max_episode_steps=200, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_five_obj.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        #'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)

register(
    id='UR10eMask4C-v0',
    entry_point='robohive.envs.arms.mask_4d_v0:ReachBaseV0',
    max_episode_steps=200, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_five_obj.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        #'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)

register(
    id='UR10eMask3C-v1',
    entry_point='robohive.envs.arms.mask_3d_v1:ReachBaseV0',
    max_episode_steps=200, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_five_obj.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        #'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)

register(
    id='UR10eMask4C-v1',
    entry_point='robohive.envs.arms.mask_4d_v1:ReachBaseV0',
    max_episode_steps=200, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_five_obj.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        #'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)


#with masking
register(
    id='UR10eReachFive-v1',
    entry_point='robohive.envs.arms.reach_base_v4:ReachBaseV0',
    max_episode_steps=200, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_five_obj.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        #'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)

#this env is for testing for two squre blocks
register(
    id='UR10eReachObject-v0',
    entry_point='robohive.envs.arms.reach_base_v3:ReachBaseV0',
    max_episode_steps=200, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_gripper.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)

#this env is for testing for two apples
register(
    id='UR10eReachObject-v1',
    entry_point='robohive.envs.arms.reach_base_v3:ReachBaseV0',
    max_episode_steps=200, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_apple.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)

#this env is testing for two cylinders
register(
    id='UR10eReachObject-v2',
    entry_point='robohive.envs.arms.reach_base_v3:ReachBaseV0',
    max_episode_steps=200, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_cylinder.xml',
        #'config_path': curr_dir+'/ur10e/ur10e_v0.config',
        'robot_site_name': "pinch",
        'target_site_name': "object_1",
        'obj_xyz_range': {'high':[0.4, 0.55, 0.995831], 'low':[0.35, 0.6 ,0.995831]}, #{'high':[0.2, 0.3, 0.895831], 'low':[-0.3, 0.6 ,0.895831]},
        'goal_site_name': "place_target",
        'target_xyz_range': {'high':[-0.435, 0.5, 0.9], 'low':[-0.435, 0.5, 0.9]}
    }
)



# Reach to random target
register_env_variant(
    env_id='FrankaReachFixed-v0',
    variant_id='FrankaReachRandom-v0',
    variants={
        'target_xyz_range': {'high':[0.3, .5, 1.2], 'low':[-.3, .1, .8]}
        },
    silent=True
)

# Reach to random target using visual inputs
register_env_variant(
    env_id='FrankaReachRandom-v0',
    variant_id='FrankaReachRandom_v2d-v0',
    variants={
            "obs_keys": ['time', 'time'],    # supress state obs
            'visual_keys':[                     # exteroception
                "rgb:left_cam:224x224:2d",
                "rgb:right_cam:224x224:2d",
                "rgb:top_cam:224x224:2d"],
        },
    silent=True
)

# Reach to random target using latent inputs
def register_visual_envs(encoder_type):
    register_env_variant(
        env_id='FrankaReachRandom-v0',
        variant_id='FrankaReachRandom_v{}-v0'.format(encoder_type),
        variants={
                'visual_keys':[
                    "rgb:left_cam:224x224:{}".format(encoder_type),
                    "rgb:right_cam:224x224:{}".format(encoder_type),
                    "rgb:top_cam:224x224:{}".format(encoder_type)]
        },
        silent=True
    )
for enc in ["r3m18", "r3m34", "r3m50", "rrl18", "rrl34", "rrl50"]:
    register_visual_envs(enc)

register_env_variant(
    env_id='UR10eReachFixed-v2',
    variant_id='UR10eReachFixed_v2d-v0',
    variants={
            "obs_keys": ['time', 'time'],    # supress state obs
            'visual_keys':[                     # exteroception
                "rgb:right_cam:224x224:2d",
                "rgb:front_cam:224x224:2d"],
        },
    silent=True
)

# FRANKA PUSH =======================================================================
from robohive.envs.arms.push_base_v0 import PushBaseV0

# Push object to target
register(
    id='FrankaPushFixed-v0',
    entry_point='robohive.envs.arms.push_base_v0:PushBaseV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_ycb_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_ycb_v0.config',
        'robot_ndof': 9,
        'robot_site_name': "end_effector",
        'object_site_name': "sugarbox",
        'target_site_name': "target",
        'target_xyz_range': {'high':[-.4, 0.5, 0.78], 'low':[-.4, 0.5, 0.78]}
    }
)

# Push object to target
register_env_variant(
    env_id='FrankaPushFixed-v0',
    variant_id='FrankaPushRandom-v0',
    variants={
        'target_xyz_range': {'high':[0.4, 0.5, 0.78], 'low':[-.4, .4, 0.78]}
        },
    silent=True
)

# Push to random target using visual inputs
register_env_variant(
    env_id='FrankaPushRandom-v0',
    variant_id='FrankaPushRandom_v2d-v0',
    variants={
            "obs_keys": ['time', 'time'],    # supress state obs
            'visual_keys':[                     # exteroception
                "rgb:left_cam:224x224:2d",
                "rgb:right_cam:224x224:2d",
                "rgb:top_cam:224x224:2d"],
        },
    silent=True
)


# FRANKA PICK-PLACE =======================================================================
from robohive.envs.arms.pick_place_v0 import PickPlaceV0

# Fixed Target
register(
    id='FrankaPickPlaceFixed-v0',
    entry_point='robohive.envs.arms.pick_place_v0:PickPlaceV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/franka/assets/franka_busbin_v0.xml',
        'config_path': curr_dir+'/franka/assets/franka_busbin_v0.config',
        'robot_ndof': 9,
        'robot_site_name': "end_effector",
        'object_site_name': "obj0",
        'target_site_name': "drop_target",
        'target_xyz_range': {'high':[-.235, 0.5, 0.85], 'low':[-.235, 0.5, 0.85]},
    }
)

# Random Targets
register_env_variant(
    env_id='FrankaPickPlaceFixed-v0',
    variant_id='FrankaPickPlaceRandom-v0',
    variants={
        'randomize': True,
        'target_xyz_range': {'high':[-.135, 0.6, 0.85], 'low':[-.335, 0.4, 0.85]},
        'geom_sizes': {'high':[.03, .03, .03], 'low':[.02, 0.02, 0.02]}
        },
    silent=True
)

# PickPlace using visual inputs
register_env_variant(
    env_id='FrankaPickPlaceRandom-v0',
    variant_id='FrankaPickPlaceRandom_v2d-v0',
    variants={
            "obs_keys": ['time', 'time'],    # supress state obs
            'visual_keys':[                     # exteroception
                "rgb:left_cam:224x224:2d",
                "rgb:right_cam:224x224:2d",
                "rgb:top_cam:224x224:2d"],
        },
    silent=True
)


#Register env for UR10e

register(
    id='UR10ePickPlaceFixed-v0',
    entry_point='robohive.envs.arms.pick_place_v1:PickPlaceV0',
    max_episode_steps=200, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/ur10e/scene_gripper.xml',
        'robot_ndof': 14,
        'robot_site_name': "pinch",
        'object_site_name': "obj0",
        'target_site_name': "pick_target",
        'target_xyz_range': {'high':[-.235, 0.5, 0.85], 'low':[-.235, 0.5, 0.85]},
    }
)


# Random Targets
register(
    id='UR10ePickPlaceRandom-v0',
    entry_point='robohive.envs.arms.pick_place_v0:PickPlaceV0',
    max_episode_steps=50,
    kwargs={
        'model_path': curr_dir+'/ur10e/scene.xml',
        #'config_path': curr_dir+'/franka/assets/franka_busbin_v0.config',
        'robot_ndof': 6,
        'robot_site_name': "attachment_site",
        'object_site_name': "obj0",
        'target_site_name': "drop_target",
        'randomize': True,
        'target_xyz_range': {'high':[-.135, 0.6, 0.85], 'low':[-.335, 0.4, 0.85]},
        'geom_sizes': {'high':[.03, .03, .03], 'low':[.02, 0.02, 0.02]},
    }
)

register(
    id='UR10ePickPlaceRandom_v2d-v0',
    entry_point='robohive.envs.arms.pick_place_v0:PickPlaceV0',
    max_episode_steps=50,
    kwargs={
        'model_path': curr_dir+'/ur10e/scene.xml',
        #'config_path': curr_dir+'/franka/assets/franka_busbin_v0.config',
        'robot_ndof': 6,
        'robot_site_name': "attachment_site",
        'object_site_name': "obj0",
        'target_site_name': "drop_target",
        'randomize': True,
        'target_xyz_range': {'high':[-.135, 0.6, 0.85], 'low':[-.335, 0.4, 0.85]},
        'geom_sizes': {'high':[.03, .03, .03], 'low':[.02, 0.02, 0.02]},
        "obs_keys": ['time', 'time'],    # supress state obs
        'visual_keys':[                     # exteroception
        "rgb:left_cam:224x224:2d",
        "rgb:right_cam:224x224:2d",
        "rgb:top_cam:224x224:2d"],
    }
)



# FETCH =======================================================================
from robohive.envs.arms.reach_base_v0 import ReachBaseV0

# Reach to fixed target
register(
    id='FetchReachFixed-v0',
    entry_point='robohive.envs.arms.reach_base_v0:ReachBaseV0',
    max_episode_steps=50, #50steps*40Skip*2ms = 4s
    kwargs={
        'model_path': curr_dir+'/fetch/assets/fetch_reach_v0.xml',
        'config_path': curr_dir+'/fetch/assets/fetch_reach_v0.config',
        # 'robot_ndof': 12,
        'robot_site_name': "grip",
        'target_site_name': "target",
        'target_xyz_range': {'high':[0.2, 0.3, 1.2], 'low':[0.2, 0.3, 1.2]}
    }
)

# Reach to random target
register_env_variant(
    env_id='FetchReachFixed-v0',
    variant_id='FetchReachRandom-v0',
    variants={
        'target_xyz_range': {'high':[0.3, .5, 1.2], 'low':[-.3, .1, .8]}
        },
    silent=True
)

# Reach to random target using visual inputs
register_env_variant(
    env_id='FetchReachRandom-v0',
    variant_id='FetchReachRandom_v2d-v0',
    variants={
            "obs_keys": ['time', 'time'],    # supress state obs
            'visual_keys':[                     # exteroception
                "rgb:left_cam:224x224:2d",
                "rgb:right_cam:224x224:2d",
                "rgb:top_cam:224x224:2d"],
        },
    silent=True
)
