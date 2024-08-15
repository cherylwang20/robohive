""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """


"""
We are using this as a testing ground for reaching with visual inputs. 
"""


import collections
#import mujoco as mp
import os

# Set environment variables
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import gym
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import copy
from robohive.physics.sim_scene import SimScene
#import dm_control.mujoco as dm_mujoco

from robohive.envs import env_base_1
from robohive.utils.quat_math import mat2euler, euler2quat

from .python_api import BodyIdInfo, arm_control, get_touching_objects, ObjLabels


class ReachBaseV0(env_base_1.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'time', 'qp_robot', 'qv_robot'
    ]
    DEFAULT_PROPRIO_KEYS = [
        'qp_robot', 'qv_robot'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": .01,
        #"bonus": 1.0,
        "contact": 2,
        "claw_ori": 5, 
        "obj_ori": .01,
        "target_dist": -1.0,
        'gripper_height': 10,
        'penalty': 5, #penalty is defined negative
        'sparse': .1,
        'solved': 10,
        "done": 1000,
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
               goal_site_name,
               target_xyz_range,
               image_width=224,
               image_height=224,
               obj_xyz_range = None,
               frame_skip = 12,#40,
               reward_mode = "dense",
               obs_keys=DEFAULT_OBS_KEYS,
               proprio_keys=DEFAULT_PROPRIO_KEYS,
               weighted_reward_keys=DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs,
        ):

        # ids
        self.target_site_name = target_site_name
        self.grasp_sid = self.sim.model.site_name2id(robot_site_name) #robot part name
        self.target_sid = self.sim.model.site_name2id(target_site_name) #object name
        self.goal_sid = self.sim.model.site_name2id(goal_site_name) #final location
        self.obj_xyz_range = obj_xyz_range #random re-initialized object location
        self.object_bid = self.sim.model.body_name2id(target_site_name)
        self.target_xyz_range = target_xyz_range
        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height
        self.grasping_steps_left = 0
        self.grasp_attempt = 0
        self.obj_init_z = self.sim.data.site_xpos[self.grasp_sid][-1]
        print(self.obj_init_z)
        self.fixed_positions = None
        self.cam_init = False
        self.color = np.random.choice(['red'])
        self.current_image = np.ones((image_width, image_height, 4), dtype=np.uint8)
        self.rgb_out = np.ones((image_height, image_width))
        self.pixel_perc = 0
        self.total_pix = 0
        self.touch_success = 0
        self.single_touch = 0
        self.cx, self.cy = 0, 0
        
        #self._last_robot_qpos = self.sim.model.key_qpos[0].copy()

        super()._setup(obs_keys=obs_keys,
                       proprio_keys=proprio_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       reward_mode=reward_mode,
                       frame_skip=frame_skip,
                       **kwargs)
        self.init_qpos[:] = self.sim.model.key_qpos[2].copy()


    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([self.sim.data.time])
        obs_dict['qp_robot'] = sim.data.qpos[:7].copy()
        obs_dict['qv_robot'] = sim.data.qvel[:7].copy()
        obs_dict['xmat_pinch'] = mat2euler(np.reshape(self.sim.data.site_xmat[self.grasp_sid], (3, 3)))
        obs_dict['obj_ori'] = mat2euler(np.reshape(self.sim.data.site_xmat[self.target_sid], (3, 3)))
        obs_dict['obj_ori_err'] =  obs_dict['obj_ori'] - np.array([np.pi/2, 0, 0])
        obs_dict['claw_ori_err'] = obs_dict['xmat_pinch'] - np.array([-np.pi, 0, -np.pi/2])
        obs_dict['reach_err'] = np.abs(np.array([self.total_pix/100])) #sim.data.site_xpos[self.target_sid]-sim.data.site_xpos[self.grasp_sid]
        obs_dict['target_err'] = sim.data.site_xpos[self.goal_sid]-sim.data.site_xpos[self.grasp_sid]
        #obs_dict['pixel'] = np.array([self.pixel_perc])
        obs_dict['power_cost'] = sim.data.qvel.copy()*sim.data.qfrc_actuator.copy()
        obs_dict['total_pix'] = np.array([self.total_pix*10]) #times 10 to incentivzie larger pixel
        self.current_observation = self.get_observation(show=True)

        this_model = sim.model
        id_info = BodyIdInfo(this_model)
        this_data = sim.data

        touching_objects = set(get_touching_objects(this_model, this_data, id_info))

        obs_vec = self._obj_label_to_obs(touching_objects)
        obs_dict["touching_body"] = obs_vec

        #self.check_contact()
        return obs_dict

    def _obj_label_to_obs(self, touching_body):
        # Function to convert touching body set to an binary observation vector
        # order follows the definition in python_api file
        obs_vec = np.array([0, 0, 0, 0])
        for i in touching_body:
            if i == ObjLabels.LEFT_GRIP:
                obs_vec[0] += 1
            elif i == ObjLabels.RIGHT_GRIP:
                obs_vec[1] += 1
            elif i == ObjLabels.GOAL:
                obs_vec[3] += 1
            else:
                obs_vec[2] += 1

        return obs_vec
    
    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict['reach_err'], axis=-1)[0]
        total_pix = np.linalg.norm(obs_dict['total_pix'], axis=-1)[0]
        target_dist = np.linalg.norm(obs_dict['target_err'], axis=-1)[0]
        claw_rot_err = np.linalg.norm(obs_dict['claw_ori_err'], axis=-1)[0]
        obj_ori_err = np.linalg.norm(obs_dict['obj_ori_err'], axis=-1)[0]
        #print(claw_rot_err)
        obj_height = np.array([self.sim.data.site_xpos[self.target_sid][-1]])
        gripper_height = np.array([self.sim.data.site_xpos[self.grasp_sid][-1]])
        pix_perc = np.array([self.pixel_perc - 2.4234])
        #print(pix_perc)
        contact = np.array([np.sum(obs_dict["touching_body"][0][0][:2])])
        if contact == 1:
            self.single_touch += 1
        elif contact == 2:
            if self.touch_success ==1:
                print('grasping')
            self.touch_success +=1
        #power_cost = np.linalg.norm(obs_dict['power_cost'], axis = -1)[0]
        rwd_dict = collections.OrderedDict((
            # Optional Keys[]
            ('reach',  total_pix),
            ('target_dist',   target_dist + np.log(target_dist + 1e-6)),
            ('claw_ori',  np.exp(-claw_rot_err**2)),
            ('obj_ori', np.exp(-obj_ori_err**2)),
            #('obj_ori',   -(obj_rot_err[0])**2), 
            #('bonus',   total_pix > 10),
            ('contact', contact),
            ('penalty', np.array([-1])),
            #('power_cost', power_cost),
            # Must keys
            ('sparse',  pix_perc),
            ('solved',  np.array([self.touch_success]) >= 25 and contact == 2),
            ('gripper_height',  gripper_height > 1.2), #gripper_height - 0.83),
            ('done',  obj_height > 1.5 ), #obj_height  - self.obj_init_z > 0.5), #reach_dist > far_th
        ))
        #print(obj_height)
        #print([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        gripper_width = np.linalg.norm([self.sim.data.site_xpos[self.sim.model.site_name2id('left_silicone_pad')]- 
                                 self.sim.data.site_xpos[self.sim.model.site_name2id('right_silicone_pad')]], axis = -1)
        return rwd_dict
    
    def reset(self, reset_qpos=None, reset_qvel=None, **kwargs):
        #print('resetting')
        self.target_sid = self.sim.model.site_name2id(self.target_site_name)
        self.grasping_steps_left = 0
        self.grasp_attempt = 0
        self.touch_success = 0
        self.single_touch = 0
        self.cx, self.cy = 0, 0
        if self.obj_xyz_range is not None:        
            reset_qpos = self.sim.model.key_qpos[2].copy()
            new_pos = self.np_random.uniform(**self.obj_xyz_range)
            self.sim.model.body_pos[self.object_bid] = new_pos
            object_qpos_adr = self.sim.model.body(self.object_bid).jntadr[0]
            self.sim.data.qpos[object_qpos_adr:object_qpos_adr+3] = new_pos
        

        #self.sim.model.site_pos[self.target_sid] = self.np_random.uniform(high=self.target_xyz_range['high'], low=self.target_xyz_range['low'])
        #self.sim_obsd.model.site_pos[self.target_sid] = self.sim.model.site_pos[self.target_sid]
        obs = super().reset(reset_qpos = reset_qpos, reset_qvel = None, **kwargs)
        #self._last_robot_qpos = self.sim.model.key_qpos[0].copy()
        self.current_image = np.ones((self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 4), dtype=np.uint8)
        self.color = np.random.choice(['red'])
        return {'image': self.current_image, 'vector': obs}
    

    def get_observation(self, show=True):
        """
        Uses the controllers get_image_data method to return an top-down image (as a np-array).

        Args:
            show: If True, displays the observation in a cv2 window.
        """

        rgb, depth = self.get_image_data(
            width=self.IMAGE_WIDTH, height=self.IMAGE_HEIGHT, show=show
        )
        #depth = self.depth_2_meters(depth) #we don't need this, already in meters
        site_pos = self.sim.data.site_xpos[self.target_sid]
        #pixel_x, pixel_y = self.world_2_pixel(site_pos)

        observation = {}
        observation["rgb"] = rgb
        #observation["depth"] =   np.array([depth[pixel_y][pixel_x]]) #np.array([ 2.701]) #
        #print(np.array([depth[pixel_y][pixel_x]]), np.array([depth[pixel_y][224 - pixel_x]]))
        #observation["pixel_coords"] = [pixel_x, pixel_y]
        #print('pixel coords,', pixel_x, pixel_y)

        return observation

    #setting a boundary of virtual box such that the arm will not accidentally
    def check_collision(self):
        """ Check if any joint is out of the defined boundary """
        x_min, x_max = -1.5, 1.5
        y_min, y_max = -1.7, 1.5
        z_min, z_max = 0.85, 2.23
        for i in range(1, 13):
            joint_frame_id = self.sim.model.jnt_bodyid[i]
            joint_pos = self.sim.data.xpos[joint_frame_id]
            if not (x_min <= joint_pos[0] <= x_max and 
                    y_min <= joint_pos[1] <= y_max and 
                    z_min <= joint_pos[2] <= z_max):
                #print(joint_pos)
                #print(f"Collision at joint {i}")
                return True
        return False
    
    def save_state(self):
        """ Save the current simulation state """
        self.previous_state = {
            'qpos': np.copy(self.sim.data.qpos),
            'qvel': np.copy(self.sim.data.qvel),
            'actuator': np.copy(self.sim.data.ctrl) if hasattr(self.sim.data, 'ctrl') else None
        }
    
    def restore_state(self, **kwargs):
        """ Restore the simulation state from self.previous_state """
        if self.previous_state:
            self.sim.data.qpos[:] = self.previous_state['qpos']
            self.sim.data.qvel[:] = self.previous_state['qvel']
            if self.previous_state['actuator'] is not None:
                self.sim.data.ctrl[:] = self.previous_state['actuator']
            obs = super().reset(reset_qpos = self.previous_state['qpos'], reset_qvel = None, **kwargs)
        return obs

    def step(self, a, **kwargs):
        """
        Step the simulation forward (t => t+1)
        Uses robot interface to safely step the forward respecting pos/ vel limits
        Accepts a(t) returns obs(t+1), rwd(t+1), done(t+1), info(t+1)
        change control method here if needed 
        """
        self.save_state()
        #if self.pixel_perc > 50 and self.grasp_attempt <= 1:
        #if self.sim.data.site_xpos[self.grasp_sid][-1] < 0.8 and self.grasp_attempt <= 1:

        #print(self.time) self.sim.data.site_xpos[self.grasp_sid][-1] < 0.53
        #print(self.single_touch) 
        if self.single_touch >= 1000:
            print('hard-coded')
            self.fixed_positions = self.sim.data.qpos[:7].copy()
            self.fixed_positions[-1] = 1
            a[-1] = 1
            self.grasping_steps_left -= 1 # Decrement the counter each step
            self.last_ctrl = self.robot.step(ctrl_desired=a,
                                        last_qpos = self.fixed_positions,
                                        dt = self.dt,
                                        render_cbk=self.mj_render if self.mujoco_render_frames else None)
        else:
            a = np.clip(a, self.action_space.low, self.action_space.high)
            self.fixed_positions = None
            self.last_ctrl = self.robot.step(ctrl_desired=a,
                                        last_qpos = self.sim.data.qpos[:7].copy(),
                                        dt = self.dt,
                                        render_cbk=self.mj_render if self.mujoco_render_frames else None)
        #self.do_simulation(ctrl_feasible, self.frame_skip)
        
        
        if self.check_collision():
            print("Collision detected, reverting action")
            self.restore_state()
        
        
        

        return self.forward(self.current_image, **kwargs)
    
    def set_color(self, color):
            self.color = color
    
    def get_image_data(self, show=False, camera="end_effector_cam", width=224, height=224):
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
            self.sim.renderer.render_offscreen(width=width, height=height, camera_id=camera, depth = True)
        )

        #self.rgb_out = rgb_out

        rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
        blurred = cv.GaussianBlur(rgb, (11, 11), 0)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        # we might want to add a series of color to identify e.g., green, blue, red, yellow. 
        
        if self.touch_success < 1000:
            if self.color == 'red':
                Lower = (0, 50, 50)
                Upper = (7, 255, 245)
            elif self.color == 'green':
                Lower = (29, 86, 56)
                Upper = (64, 255, 255)
            elif self.color == 'blue':
                Lower = (80, 50, 20)
                Upper = (100, 255, 255)
            else:
                raise Warning('please define a valid color (red, gree, blue)')
        else:
            Lower = (0, 0, 0)
            Upper = (0, 0, 0)
        mask = cv.inRange(hsv, Lower, Upper)
        mask = cv.erode(mask, None, iterations=2)
        mask = cv.dilate(mask, None, iterations=2)
        self.current_image = np.concatenate((rgb/255, np.expand_dims(mask/255, axis=-1)), axis=2)
        
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


        if contours:
            cnt = max(contours, key = cv.contourArea)

            # Calculate the centroid of the contour
            M = cv.moments(cnt)
            if M['m00'] != 0:
                self.cx = int(M['m10']/M['m00'])
                self.cy = int(M['m01']/M['m00'])
        #else:
            #self.cx, self.cy = 0, 224
        
        self.pixel_dis = np.abs(np.linalg.norm(np.array([100, 100]) - np.array([self.cx, self.cy])))

        #define the grasping rectangle
        x1, y1 = int(63/200 * self.IMAGE_HEIGHT), self.IMAGE_HEIGHT - int(68/200 * self.IMAGE_HEIGHT)
        x2, y2 = int(136/200 * self.IMAGE_HEIGHT), self.IMAGE_HEIGHT 

        cv.rectangle(rgb, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
        cv.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=1)

        roi = mask[y1:y2, x1:x2]
        white_pixels = np.sum(roi == 255)
        total_pixels = roi.size
        self.pixel_perc = (white_pixels / total_pixels) * 100
        self.total_pix = (np.sum(mask==255)/mask.size) * 100

        #print(f"Percentage of white pixels in the rectangle: {self.pixel_perc:.2f}%")
        if show:
            cv.circle(rgb, (self.cx, self.cy), 1, (0, 0, 255), -1)
            cv.circle(rgb, (100, 100), 1, (0, 255, 0), -1)
            #cv.imshow("rbg", rgb)# cv.cvtColor(rgb, cv.COLOR_BGR2RGB))
            #cv.imshow("mask", mask)
            #cv.imshow('Inverted Colored Depth', depth_normalized)
            #cv.waitKey(1)
            # cv.waitKey(delay=5000)
            # cv.destroyAllWindows()

        return np.array(np.fliplr(np.flipud(rgb))), np.array(np.fliplr(np.flipud(depth)))

    def render(self, mode='rgb_array'):
        # Your implementation here, which should return an RGB array if mode is 'rgb_array'
        mode='rgb_array'
        if mode == 'rgb_array':
            rgb, depth = copy.deepcopy(
            self.sim.renderer.render_offscreen(width=224, height=224, camera_id='end_effector_cam', depth = True)
            )
            return rgb
        else:
            super().render(mode)
