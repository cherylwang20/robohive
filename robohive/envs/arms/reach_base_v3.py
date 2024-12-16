""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import collections
import gym
import numpy as np
import cv2 as cv
import copy

from robohive.envs import env_base_2
from robohive.envs.arms.python_api_2 import BodyIdInfo, arm_control, get_touching_objects, ObjLabels



class ReachBaseV0(env_base_2.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'qp_robot', 'qv_robot'
    ]
    DEFAULT_PROPRIO_KEYS = [
        'qp_robot', 'qv_robot'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": -1.0,
        #"bonus": 4.0,
        'solved': 1, 
        "penalty": -50,
    }


    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)

        self._setup(**kwargs)


    def _setup(self,
               robot_site_name,
               target_site_name,
               target_xyz_range,
               frame_skip = 20,
               image_width = 212,
               image_height= 120,
               reward_mode = "dense",
               obs_keys=DEFAULT_OBS_KEYS,
               proprio_keys=DEFAULT_PROPRIO_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs,
        ):

        # ids
        self.grasp_sid = self.sim.model.site_name2id(robot_site_name)
        self.target_site_name = target_site_name
        self.target_sid = self.sim.model.site_name2id(target_site_name)
        self.target_xyz_range = target_xyz_range
        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height
        self.rgb_out = np.ones((image_width, image_height))
        self.current_image = np.ones((image_width, image_height, 3), dtype=np.uint8)
        self.vel_action = [0]*6
        self.contact = 0 
        super()._setup(obs_keys=obs_keys,
                       proprio_keys=proprio_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       reward_mode=reward_mode,
                       frame_skip=frame_skip,
                       **kwargs)
        self.init_qpos[:] = self.sim.model.key_qpos[0].copy()


    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([self.sim.data.time])
        obs_dict['qp_robot'] = sim.data.qpos[:7].copy()
        obs_dict['qv_robot'] = self.vel_action.copy()
        obs_dict['reach_err'] = sim.data.site_xpos[self.target_sid]-sim.data.site_xpos[self.grasp_sid]
        obs_dict['goal_pos'] = sim.data.site_xpos[self.target_sid]

        self.get_image_data()

        this_model = sim.model
        id_info = BodyIdInfo(this_model)
        this_data = sim.data

        touching_objects = set(get_touching_objects(this_model, this_data, id_info, 'object_1'))
        #print('touching objects', touching_objects)

        obs_vec = self._obj_label_to_obs(touching_objects)
        obs_dict["touching_body"] = obs_vec
        return obs_dict
    
    def get_image_data(self, show=False, camera="end_effector_cam", width= 212, height= 120):
        """
        Returns the RGB and depth images of the provided camera.

        Args:
            show: If True displays the images for five seconds or until a key is pressed.
            camera: String specifying the name of the camera to use.
        """
        #rgb_out =  self.sim.renderer.render_offscreen(width=800, height=800, camera_id='ft_cam')
        #self.sim.renderer.close()

        # Initialize the simulator
        rgb, depth = copy.deepcopy(
            self.sim.renderer.render_offscreen(height=height,width=width,  camera_id=camera, depth = True)
        )
        
        self.rgb_out = rgb

        rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)


        self.current_image = rgb/255

        return np.array(np.fliplr(np.flipud(rgb))), np.array(np.fliplr(np.flipud(depth)))

    def _obj_label_to_obs(self, touching_body):
        # Function to convert touching body set to an binary observation vector
        # order follows the definition in python_api file
        obs_vec = np.array([0, 0, 0])
        for i in touching_body:
            if i == ObjLabels.LEFT_GRIP:
                obs_vec[0] += 1
            elif i == ObjLabels.RIGHT_GRIP:
                obs_vec[1] += 1
            else:
                obs_vec[2] += 1

        return obs_vec


    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict['reach_err'], axis=-1)
        far_th = 2.0
        contact = np.array([[np.sum(obs_dict["touching_body"][0][0][:2])]])
        if contact > 0:
            self.contact += 1
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   reach_dist),
            ('bonus',   (reach_dist<.1) + (reach_dist<.05)),
            ('penalty', (reach_dist>far_th)),
            # Must keys
            ('sparse',  -1.0*reach_dist),
            ('solved',  reach_dist < 0.05),
            ('done',    reach_dist > far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict
    
    def step(self, a, **kwargs):
        """
        Step the simulation forward (t => t+1)
        Uses robot interface to safely step the forward respecting pos/ vel limits
        Accepts a(t) returns obs(t+1), rwd(t+1), done(t+1), info(t+1)
        """
        a = np.clip(a, self.action_space.low, self.action_space.high)
        if self.contact >= 1:
            a[-1] = 1
        self.last_ctrl, self.vel_action = self.robot.step(ctrl_desired=a,
                                        last_qpos = self.sim.data.qpos[:7].copy(),
                                        #ctrl_normalized=self.normalize_act,
                                        dt=self.dt,
                                        #realTimeSim=self.mujoco_render_frames,
                                        render_cbk=self.mj_render if self.mujoco_render_frames else None)
        return self.forward(**kwargs)
    
    def forward(self,**kwargs):
        """
        Forward propagate env to recover env details
        Returns current obs(t), rwd(t), done(t), info(t)
        """

        # render the scene
        if self.mujoco_render_frames:
            self.mj_render()

        # observation
        obs = self.get_obs(**kwargs)

        # rewards
        self.expand_dims(self.obs_dict) # required for vectorized rewards calculations
        self.rwd_dict = self.get_reward_dict(self.obs_dict)
        self.squeeze_dims(self.rwd_dict)
        self.squeeze_dims(self.obs_dict)

        # finalize step
        env_info = self.get_env_infos()

        obs = {'image': self.current_image.reshape((self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3)), 'vector': obs}

        return obs, env_info['rwd_'+self.rwd_mode], bool(env_info['done']), env_info


    def reset(self, reset_qpos=None, reset_qvel=None):
        reset_qpos = self.sim.model.key_qpos[0].copy()
        self.sim.model.site_pos[self.target_sid] = self.np_random.uniform(high=self.target_xyz_range['high'], low=self.target_xyz_range['low'])
        self.sim_obsd.model.site_pos[self.target_sid] = self.sim.model.site_pos[self.target_sid]
        obj_xyz_ranges = {
            'object': {'low': [-0.85, -0.05, 0], 'high': [.85, 0.15, 0]},
        }

        new_x, new_y = np.random.uniform(
                low=[obj_xyz_ranges['object']['low'][0], obj_xyz_ranges['object']['low'][1]],
                high=[obj_xyz_ranges['object']['high'][0], obj_xyz_ranges['object']['high'][1]],
                size=2
        )

        reset_qpos = self.sim.model.key_qpos[0].copy()

        objec_bid = self.sim.model.body_name2id('object_1')  # get body ID using object name
        object_jnt_adr = self.sim.model.body_jntadr[objec_bid]
        object_qpos_adr = self.sim.model.jnt_qposadr[object_jnt_adr]
        initial_pos = reset_qpos[object_qpos_adr:object_qpos_adr + 3]  # copy the initial position
        z_coord = initial_pos[2]  # get the fixed z-coordinate from the initial position

        # Generate new x, y positions within specified ranges, keeping z constant
        new_pos = [initial_pos[0] + new_x, initial_pos[1] + new_y, z_coord]
        reset_qpos[object_qpos_adr:object_qpos_adr + 3] = new_pos

        obs = super().reset(reset_qpos, reset_qvel)
        self.final_image = np.ones((self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3), dtype=np.uint8)
        return {'image': self.final_image, 'vector': obs}
