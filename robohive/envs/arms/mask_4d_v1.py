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
import sys
import groundingdino
# from groundingdino.util.inference import load_model, load_image, predict, annotate
# import groundingdino.datasets.transforms as T
from PIL import Image, ImageDraw
from torchvision.ops import box_convert
import torch

# Set environment variables
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

from robohive.envs.arms.python_api_2 import BodyIdInfo, arm_control, get_touching_objects, ObjLabels


class ReachBaseV0(env_base_1.MujocoEnv):

    DEFAULT_OBS_KEYS = [
        'time', 'qp_robot', 'qv_robot'
    ]
    DEFAULT_PROPRIO_KEYS = [
        'qp_robot', 'qv_robot'
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": 1, 
        #"bonus": 1.0,
        "contact": 1,
        #"claw_ori": 1, 
        #"obj_ori": .01,
        #"target_dist": -1.0,
        #'gripper_height': 1,
        #'penalty': 1, #penalty is defined negative
        'sparse': 1,
        'solved': 1,
        "done": 10,
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
        #self.goal_sid = self.sim.model.site_name2id(goal_site_name) #final location
        if obj_xyz_range is not None:
            self.obj_xyz_range = obj_xyz_range #random re-initialized object location
        self.object_bid = self.sim.model.body_name2id(target_site_name)
        self.target_xyz_range = target_xyz_range
        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height
        self.grasping_steps_left = 0
        self.grasp_attempt = 0
        self.obj_init_z = self.sim.data.site_xpos[self.grasp_sid][-1]
        self.fixed_positions = None
        self.cam_init = True
        self._setup_camera()
        self.color = np.random.choice(['green'])
        self.current_image = np.ones((image_width, image_height, 4), dtype=np.uint8)
        self.object_image = np.ones((image_width, image_height, 3), dtype=np.uint8)
        self.rgb_out = np.ones((image_height, image_width))
        self.mask_out = np.ones((image_height, image_width))
        self.pixel_perc = 0
        self.total_pix = 0
        self.touch_success = 0
        self.single_touch = 0
        self.cx, self.cy = 0, 0
        self.r = 0
        self.depth = 0
        
        if 'eval_mode' in kwargs:
            self.eval_mode = kwargs['eval_mode']
        else: 
            self.eval_mode = False
            
        # self.mask_model = load_model( "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "./GroundingDINO/weights/groundingdino_swint_ogc.pth")
        self.BOX_THRESHOLD = 0.4
        self.TEXT_THRESHOLD = 0.25
        self.TEXT_PROMPT = 'apple'

        #self._last_robot_qpos = self.sim.model.key_qpos[0].copy()

        super()._setup(obs_keys=obs_keys,
                       proprio_keys=proprio_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       reward_mode=reward_mode,
                       frame_skip=frame_skip,
                       **kwargs)
        self.init_qpos[:] = self.sim.model.key_qpos[1].copy()


    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([self.sim.data.time])
        obs_dict['qp_robot'] = sim.data.qpos[:7].copy()
        obs_dict['qv_robot'] = sim.data.qvel[:7].copy()
        obs_dict['xmat_pinch'] = mat2euler(np.reshape(self.sim.data.site_xmat[self.grasp_sid], (3, 3)))
        #obs_dict['obj_ori'] = mat2euler(np.reshape(self.sim.data.site_xmat[self.target_sid], (3, 3)))
        #obs_dict['obj_ori_err'] =  obs_dict['obj_ori'] - np.array([np.pi/2, 0, 0])
        obs_dict['claw_ori_err'] = obs_dict['xmat_pinch'] - np.array([-np.pi, 0, -np.pi/2])
        obs_dict['reach_err'] = sim.data.site_xpos[self.target_sid]-sim.data.site_xpos[self.grasp_sid]
        #obs_dict['target_err'] = sim.data.site_xpos[self.goal_sid]-sim.data.site_xpos[self.grasp_sid]
        #obs_dict['pixel'] = np.array([self.pixel_perc])
        obs_dict['power_cost'] = sim.data.qvel.copy()*sim.data.qfrc_actuator.copy()
        obs_dict['total_pix'] = np.array([self.total_pix/100]) 
        self.current_observation = self.get_observation(show=True)

        this_model = sim.model
        id_info = BodyIdInfo(this_model)
        this_data = sim.data

        touching_objects = set(get_touching_objects(this_model, this_data, id_info, self.target_site_name))

        obs_vec = self._obj_label_to_obs(touching_objects)
        obs_dict["touching_body"] = obs_vec

        #self.check_contact()
        return obs_dict

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
        reach_dist = np.linalg.norm(obs_dict['reach_err'], axis=-1)[0]
        self.depth = reach_dist
        total_pix = np.linalg.norm(obs_dict['total_pix'], axis=-1)[0]
        #target_dist = np.linalg.norm(obs_dict['target_err'], axis=-1)[0]
        claw_rot_err = np.linalg.norm(obs_dict['claw_ori_err'], axis=-1)[0]
        #obj_ori_err = np.linalg.norm(obs_dict['obj_ori_err'], axis=-1)[0]
        #print(claw_rot_err)
        obj_height = np.array([self.sim.data.site_xpos[self.target_sid][-1]])
        gripper_height = np.array([self.sim.data.site_xpos[self.grasp_sid][-1]])
        pix_perc = np.array([self.pixel_perc - 2.4234])/10
        contact = np.array([np.sum(obs_dict["touching_body"][0][0][:2])])
        #print(contact)
        if contact == 1:
            self.single_touch += 1
            if self.single_touch == 1:
                print('first touch')
        elif contact == 2:
            if self.touch_success == 1:
                print('grasping')
            self.touch_success +=1
        #print(contact)
        #power_cost = np.linalg.norm(obs_dict['power_cost'], axis = -1)[0]
        rwd_dict = collections.OrderedDict((
            # Optional Keys[]
            ('reach',  total_pix),
            #('target_dist',   target_dist + np.log(target_dist + 1e-6)),
            ('claw_ori',  np.exp(-claw_rot_err**2)),
            #('obj_ori', np.exp(-obj_ori_err**2)),
            #('obj_ori',   -(obj_rot_err[0])**2), 
            #('bonus',   total_pix > 10),
            ('contact', contact == 2),
            ('penalty', np.array([-1])),
            #('power_cost', power_cost),
            # Must keys
            ('sparse',  pix_perc),
            ('solved',  np.array([self.touch_success]) >= 20 and contact == 2),
            ('gripper_height',  gripper_height - 0.83),
            ('done', contact == 2), #    obj_height  - self.obj_init_z > 0.2, #reach_dist > far_th
        ))
        #print(pix_perc, total_pix)
        #print([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])
        if not self.eval_mode:
            rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        else:
            rwd_dict['dense'] = 1.0 if contact == 2 else 0
            rwd_dict['done'] = contact == 2
            
        gripper_width = np.linalg.norm([self.sim.data.site_xpos[self.sim.model.site_name2id('left_silicone_pad')]- 
                                 self.sim.data.site_xpos[self.sim.model.site_name2id('right_silicone_pad')]], axis = -1)
        return rwd_dict
    
    def reset(self, reset_qpos=None, reset_qvel=None, **kwargs):
        #print('resetting')
        #self.target_sid = self.sim.model.site_name2id(self.target_site_name)
        self.grasping_steps_left = 0
        self.grasp_attempt = 0
        self.touch_success = 0
        self.single_touch = 0
        self.cx, self.cy = 0, 0
        '''
        if self.obj_xyz_range is not None:        
            reset_qpos = self.sim.model.key_qpos[1].copy()
            new_pos = self.np_random.uniform(**self.obj_xyz_range)
            self.sim.model.body_pos[self.object_bid] = new_pos
            object_qpos_adr = self.sim.model.body(self.object_bid).jntadr[0]
            self.sim.data.qpos[object_qpos_adr:object_qpos_adr+3] = new_pos
        '''

        #randomly choose between the five objects; color it green, and the rest as white. 
        target_sites = ['object_1', 'object_2', 'object_3', 'object_4', 'object_5']
        target_names = ['apple', 'block', 'beaker', 'donut', 'rubber duck']
        number = np.random.randint(0, 5)
        self.target_site_name = target_sites[number]
        self.TEXT_PROMPT = target_names[number]
        self.target_sid = self.sim.model.site_name2id(self.target_site_name) #object name
        current_directory = os.getcwd()
        self.object_image = cv.imread(current_directory + '/mj_envs/robohive/envs/arms/object_image/' + self.target_site_name + '.png', cv.IMREAD_COLOR)
        self.object_image = cv.cvtColor(self.object_image, cv.COLOR_BGR2RGB)

        
        '''
        for object_name in target_sites:
            object_id = self.sim.model.geom_name2id(object_name)
            if object_name == self.target_site_name:
                # Set the color to green with alpha 0.9
                self.sim.model.geom_rgba[object_id] = [0, 1, 0, 0.9]
            else:
                # Set the color to white with alpha 0.9
                self.sim.model.geom_rgba[object_id] = [1, 1, 1, 0.9]
        '''

        obs = super().reset(reset_qpos = reset_qpos, reset_qvel = None, **kwargs)
        #self._last_robot_qpos = self.sim.model.key_qpos[0].copy()
        self.final_image = np.ones((self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 4), dtype=np.uint8)
        self.color = np.random.choice(['green'])
        return {'image': self.final_image, 'vector': obs}
    

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
        pixel_x, pixel_y, radius = self.world_2_pixel(site_pos)
        self.cx, self.cy = pixel_x, pixel_y
        self.r = radius

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
    
        self.object_image_normalized = self.object_image / 255
        self.final_image = self.current_image

        return self.forward(self.final_image, **kwargs)
    
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

        self.rgb_out = rgb

        rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
        blurred = cv.GaussianBlur(rgb, (11, 11), 0)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
        
        mask = np.zeros(( self.IMAGE_HEIGHT,  self.IMAGE_HEIGHT), dtype=np.uint8)
        x, y = self.cx, self.cy
        if isinstance(self.r, np.ndarray):
            half_side = int(self.r.item())
        else:
            half_side = int(self.r)
        cv.rectangle(mask, (224 - x - half_side, y - half_side), (224- x + half_side, y + half_side), 255, thickness=-1)

        self.mask_out = mask

        #print(self.TEXT_PROMPT, boxes, logits, phrases)

        # Display the mask
        '''
        cv.imshow('Mask', mask)
        cv.imshow("rbg", rgb)
        cv.waitKey(1)
        cv.waitKey(delay=5000)
        cv.destroyAllWindows()
        '''

        self.current_image = np.concatenate((rgb/255, np.expand_dims(mask/255, axis=-1)), axis=2)

        #print(self.current_image.shape)
        
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


        #print('total pixel',self.total_pix)


        return np.array(np.fliplr(np.flipud(rgb))), np.array(np.fliplr(np.flipud(depth)))

    def create_mask(self, image_source: np.ndarray, boxes: torch.Tensor) -> np.ndarray:
        """
        This function creates a mask with white rectangles on a black background,
        where the rectangles are defined by the bounding boxes.

        Parameters:
        image_source (np.ndarray): The source image for determining the size of the mask.
        boxes (torch.Tensor): A tensor containing bounding box coordinates in cxcywh format.

        Returns:
        np.ndarray: The mask image.
        """
        # Get the dimensions of the source image
        h, w, _ = image_source.shape

        # Scale the boxes to the image dimensions
        boxes = boxes * torch.Tensor([w, h, w, h])

        # Convert boxes from cxcywh to xyxy format
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # Create a black mask
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw each box as a white rectangle on the mask
        for box in xyxy:
            top_left = (int(box[0]), int(box[1]))
            bottom_right = (int(box[2]), int(box[3]))
            cv.rectangle(mask, top_left, bottom_right, (255), thickness=-1)  # Fill the rectangle

        return mask

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
    
    def load_image2(self, image_source):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_transformed, _ = transform(image_source, None)
        return image_transformed

    def depth_2_meters(self, depth):
        """
        Converts the depth array delivered by MuJoCo (values between 0 and 1) into actual m values.

        Args:
            depth: The depth array to be converted.
        """
        current_directory = os.getcwd()
        self.model = mp.MjModel.from_xml_path(current_directory + "/mj_envs/robohive/envs/arms/ur10e/scene_chem.xml")
        extend = self.model.stat.extent
        near = self.model.vis.map.znear * extend
        far = self.model.vis.map.zfar * extend
        return near / (1 - depth * (1 - near / far))

    def pixel_2_world(self, pixel_x, pixel_y, depth, width=224, height=224, camera="end_effector_cam"):
        """
        Converts pixel coordinates into world coordinates.

        Args:
            pixel_x: X-coordinate in pixel space.
            pixel_y: Y-coordinate in pixel space.
            depth: Depth value corresponding to the pixel.
            width: Width of the image (pixel).
            height: Height of the image (pixel).
            camera: Name of camera used to obtain the image.
        """

        if not self.cam_init:
            self.create_camera_data(width, height, camera)

        # Create coordinate vector
        pixel_coord = np.array([pixel_x, pixel_y, 1])

        # Apply the intrinsic matrix to get camera space coordinates
        pos_c = np.linalg.inv(self.cam_matrix) @ pixel_coord
        pos_c *= -depth  # Apply depth to scale to the actual position in camera space

        # Convert camera space coordinates to world coordinates
        pos_w = np.linalg.inv(self.cam_rot_mat) @ pos_c + self.cam_pos

        return pos_w

    def _setup_camera(self, height=224, width=224):
        """Sets up the camera to render the scene from the required view."""
        # This assumes you have a fixed camera in your model XML
        self.camera_id = self.sim.model.camera_name2id('end_effector_cam')
        self.get_camera_matrices(self.camera_id, height, width)
    
    def get_camera_matrices(self, camera_id, height, width):
        """Retrieve projection, position, and rotation matrices for the specified camera."""
        fovy = self.sim.model.cam_fovy[camera_id]  # Fetch camera settings
        # Calculate focal length
        f = 0.5 * height / np.tan(fovy * np.pi / 360)
        #construct camera matrix
        self.cam_matrix = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
        self.cam_init = True
    

    def world_2_pixel(self, world_coordinate, width=224, height=224, camera="end_effector_cam"):
        """
        Takes a XYZ world position and transforms it into pixel coordinates.
        Mainly implemented for testing the correctness of the camera matrix, focal length etc.

        Args:
            world_coordinate: XYZ world coordinate to be transformed into pixel space.
            width: Width of the image (pixel).
            height: Height of the image (pixel).
            camera: Name of camera used to obtain the image.
        """

        if not self.cam_init:
            self.create_camera_data(width, height, camera)
        self.cam_pos = self.sim.data.cam_xpos[self.sim.model.camera_name2id(camera)]
        
        self.cam_rot_mat = self.sim.data.cam_xmat[self.sim.model.camera_name2id(camera)].reshape(3, 3)

        
        cam_coord = self.cam_rot_mat.T @ (world_coordinate - self.cam_pos)
    

        # Project to image plane
        hom_pixel = self.cam_matrix @ cam_coord
        # Real image point
        if hom_pixel[2] != 0:
            pixel = hom_pixel[:2] / hom_pixel[2]
        else:
            pixel = hom_pixel[:2]  # Avoid division by zero
        radius = self.calculate_radius(self.depth)
        return np.round(pixel[0]).astype(int), np.round(pixel[1]).astype(int), radius
    
    def calculate_radius(self, d_depth):
        """
        Calculates the radius based on the depth.
        The function can be adjusted based on empirical data or desired visualization effect.
        """
        base_radius = 20  # Maximum radius when object is at depth zero
        if d_depth > 0:
            radius = base_radius*0.2/d_depth
            return radius  # Example function: Decrease radius with depth
        else:
            return 5
