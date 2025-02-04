""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
SCRIPT CREATED TO TRY DIFFERENT REWARDS ON THE WALK_V0 ENVIRONMENT. 
================================================= """

import collections
import random
import gym
import numpy as np
from robohive.envs.myo.base_v0 import BaseV0
import matplotlib.path as mplPath
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import time
from robohive.utils.quat_math import mat2euler, euler2quat
import cv2

class ReachEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['qpos', 'qvel', 'tip_pos', 'reach_err']
    # Weights should be positive, unless the contribution of the components of the reward shuld be changed. 
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "positionError":    1.0,
        "smallErrorBonus":  4.0,
        "highError":        50,
        "metabolicCost":    0
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
        self.cpt = 0
        self.perturbation_time = -1
        self.perturbation_duration = 0
        self.force_range = [0, 1]
        self._setup(**kwargs)

    def _setup(self,
            target_reach_range:dict,
            far_th = .35,
            target_rot = None,
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):
        self.far_th = far_th
        self.target_reach_range = target_reach_range
        self.target_rot = target_rot
        super()._setup(obs_keys=obs_keys,
                weighted_reward_keys=weighted_reward_keys,
                sites=self.target_reach_range.keys(),
                **kwargs,
                )
        key_index = random.randint(3, 5)
        self.init_qpos = self.sim.model.key_qpos[0]
        

    def step(self, a):
        if self.perturbation_time <= self.time < self.perturbation_time + self.perturbation_duration*self.dt : 
            self.sim.data.xfrc_applied[self.sim.model.body_name2id('pelvis'), :] = self.perturbation_magnitude
        else: self.sim.data.xfrc_applied[self.sim.model.body_name2id('pelvis'), :] = np.zeros((1, 6))
        # rest of the code for performing a regular environment step
        a = np.clip(a, self.action_space.low, self.action_space.high)
        self.last_ctrl = self.robot.step(ctrl_desired=a,
                                          ctrl_normalized=self.normalize_act,
                                          step_duration=self.dt,
                                          realTimeSim=self.mujoco_render_frames,
                                          render_cbk=self.mj_render if self.mujoco_render_frames else None)
        return super().forward()

    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time + 1.5])
        self.obs_dict['qpos'] = self.sim.data.qpos[:].copy()
        self.obs_dict['qvel'] = self.sim.data.qvel[:].copy()*self.dt
        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()
        # reach error
        self.obs_dict['tip_pos'] = np.array([])
        self.obs_dict['target_pos'] = np.array([])
        for isite in range(len(self.tip_sids)):
            #self.obs_dict['tip_pos'] = np.append(self.obs_dict['tip_pos'], self.sim.data.site_xpos[self.tip_sids[isite]].copy())
            self.obs_dict['target_pos'] = np.append(self.obs_dict['target_pos'], self.sim.data.site_xpos[self.target_sids[isite]].copy())
        self.obs_dict['tip_pos'] = self.obs_dict['target_pos']
        self.obs_dict['reach_err'] = np.array(self.obs_dict['target_pos'])-np.array(self.obs_dict['tip_pos'])

        # center of mass and base of support
        xpos = {}
        body_names = ['calcn_l', 'calcn_r', 'femur_l', 'femur_r', 'patella_l', 'patella_r', 'pelvis', 
                      'root', 'talus_l', 'talus_r', 'tibia_l', 'tibia_r', 'toes_l', 'toes_r', 'world']
        for names in body_names: xpos[names] = self.sim.data.xipos[self.sim.model.body_name2id(names)].copy() # store x and y position of the com of the bodies
        # Bodies relevant for hte base of support: 
        labels = ['calcn_r', 'calcn_l', 'toes_l', 'toes_r']
        x, y = [], [] # Storing position of the foot
        for label in labels:
            x.append(xpos[label][0]) # storing x position
            y.append(xpos[label][1]) # storing y position
        # CoM is considered to be the center of mass of the pelvis (for now)
        pos = self.sim.data.xipos.copy()
        vel = self.sim.data.cvel.copy()
        mass = self.sim.model.body_mass
        com_v = np.sum(vel *  mass.reshape((-1, 1)), axis=0) / np.sum(mass)
        self.obs_dict['com_v'] = com_v[-3:]
        com = np.sum(pos * mass.reshape((-1, 1)), axis=0) / np.sum(mass)
        self.obs_dict['com'] = com[:2]
        self.obs_dict['com_height'] = com[-1:]
        self.obs_dict['hip_flex_r'] = np.asarray(self.sim.data.joint('hip_flexion_r').qpos.copy())
        self.obs_dict['cal_l'] = np.array(self.sim.data.xipos[self.sim.model.body_name2id('calcn_l')].copy()[1])
        # Storing base of support - x and y position of right and left calcaneus and toes
        self.obs_dict['base_support'] =  [x, y]
        #self.obs_dict['ver_sep'] = np.array(max(y), min(y))
        # print('Ordered keys: {}'.format(self.obs_keys))
        #self.obs_dict['err_cal'] = np.array(0.31 - self.obs_dict['cal_l'] )
        self.obs_dict['knee_angle'] = np.array(np.mean(self.sim.data.qpos[self.sim.model.joint_name2id('knee_angle_l')].copy() + self.sim.data.qpos[self.sim.model.joint_name2id('knee_angle_r')].copy()))
        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}

        obs_dict['time'] = np.array([sim.data.time + 1.5])  
        #print('time', obs_dict['time'] )    
        obs_dict['qpos'] = sim.data.qpos[:].copy()
        obs_dict['qvel'] = sim.data.qvel[:].copy()*self.dt
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        # reach error
        obs_dict['tip_pos'] = np.array([])
        obs_dict['target_pos'] = np.array([])

        ### we append the target position of the two feet
        

        for isite in range(len(self.tip_sids)):
            obs_dict['tip_pos'] = np.append(obs_dict['tip_pos'], sim.data.site_xpos[self.tip_sids[isite]].copy())
            obs_dict['target_pos'] = np.append(obs_dict['target_pos'], sim.data.site_xpos[self.target_sids[isite]].copy())
        obs_dict['reach_err'] = np.array(obs_dict['target_pos'])-np.array(obs_dict['tip_pos'])


        '''
        obs_dict['body_pos'] = np.array([])
        obs_dict['target_feet'] = np.array([])

        target_foot_coordinates = [[-0.1048 , 0.2447, 0],[-0.0989, -0.1317, 0], [0.0845,  0.1449,0],  [0.0796, -0.0315,0]] 
        foot_site = ['calcn_r', 'toes_r', 'calcn_l', 'toes_l', ]
        for site in range(len(foot_site)):
            obs_dict['body_pos'] = np.append(obs_dict['body_pos'], sim.data.xpos[self.sim.model.body_name2id(foot_site[site])])
            obs_dict['target_feet'] = np.append(obs_dict['target_feet'], target_foot_coordinates[site].copy())
        '''
        obs_dict['feet_height_r'] = np.array([(sim.data.body_xpos[sim.model.body_name2id('calcn_r')][2] + sim.data.body_xpos[sim.model.body_name2id('toes_r')][2])/2]) ##self._get_feet_heights().copy()
        obs_dict['feet_heights']= self._get_feet_heights().copy()
        a = (self.sim.data.joint('hip_adduction_r').qpos.copy()+self.sim.data.joint('hip_adduction_l').qpos.copy())/2
        obs_dict['hip_add'] = np.asarray([a])
        b = (self.sim.data.joint('knee_angle_r').qpos.copy()+self.sim.data.joint('knee_angle_l').qpos.copy())/2
        obs_dict['knee_angle'] = np.asarray([b])
        c = (self.sim.data.joint('hip_flexion_r').qpos.copy()+self.sim.data.joint('hip_flexion_l').qpos.copy())/2
        obs_dict['hip_flex'] = np.asarray([c])
        obs_dict['hip_flex_r'] = np.asarray(self.sim.data.joint('hip_flexion_l').qpos.copy())
        obs_dict['hip_rot_r'] = np.asarray(self.sim.data.joint('hip_rotation_l').qpos.copy())
        # center of mass and base of support
        x, y = np.array([]), np.array([])
        for label in ['calcn_r', 'calcn_l', 'toes_l', 'toes_r']:
            xpos = np.array(sim.data.xipos[sim.model.body_name2id(label)].copy())[:2] # select x and y position of the current body
            x = np.append(x, xpos[0])
            y = np.append(y, xpos[1])
        #obs_dict['cal_l'] = np.array(sim.data.xipos[sim.model.body_name2id('calcn_l')].copy())
        obs_dict['base_support'] = np.append(x, y)
        #obs_dict['ver_sep'] = np.array(max(y), min(y))
        # CoM is considered to be the center of mass of the pelvis (for now) 
        pos = sim.data.xipos.copy()
        vel = sim.data.cvel.copy()
        #obs_dict['feet_v'] = sim.data.cvel[sim.model.body_name2id('patella_r')].copy()
        #3*sim.data.cvel[sim.model.body_name2id('pelvis')].copy() - sim.data.cvel[sim.model.body_name2id('toes_r')].copy()
        mass = sim.model.body_mass
        com = np.sum(pos * mass.reshape((-1, 1)), axis=0) / np.sum(mass)
        com_v = np.sum(vel *  mass.reshape((-1, 1)), axis=0) / np.sum(mass)
        obs_dict['com_v'] = com_v[-3:]
        obs_dict['com'] = com[:2]
        obs_dict['com_height'] = com[-1:]# self.sim.data.body('pelvis').xipos.copy()
        baseSupport = obs_dict['base_support'].reshape(2,4)
        #areaofbase = Polygon(zip(baseSupport[0], baseSupport[1])).area
        obs_dict['centroid'] = np.array(Polygon(zip(baseSupport[0], baseSupport[1])).centroid.coords)
        pelvis_com = np.array(sim.data.xipos[sim.model.body_name2id('pelvis')].copy())
        obs_dict['pelvis_com'] = pelvis_com[:2]
        obs_dict['err_com'] = np.array(obs_dict['centroid']- obs_dict['com'])
        #obs_dict['err_com'] = np.array(obs_dict['centroid']- obs_dict['pelvis_com']) #change since 2023/12/08/ 15:52
        return obs_dict


    def get_reward_dict(self, obs_dict):
        #print('hip flexion',self.sim.data.joint('hip_flexion_r').qpos.copy())
        hip_fle = self.obs_dict['hip_flex']
        hip_rot_r = np.linalg.norm(self.obs_dict['hip_rot_r'], axis = -1)
        hip_flex_r = self.obs_dict['hip_flex_r'].reshape(-1)[0]
        hip_add = self.obs_dict['hip_add']
        knee_angle = self.obs_dict['knee_angle']
        self.obs_dict['pelvis_target_rot'] = [-0.35, 0, -1.65]#[np.pi/2, -np.pi/2 , 0]
        self.obs_dict['pelvis_rot'] = mat2euler(np.reshape(self.sim.data.site_xmat[self.sim.model.site_name2id("pelvis")], (3, 3)))
        pelvis_rot_err = np.abs(np.linalg.norm(self.obs_dict['pelvis_rot'] - self.obs_dict['pelvis_target_rot'] , axis=-1))
        #print(pelvis_rot_err)
        #print(self.obs_dict['pelvis_rot'])
        #print(-self.obs_dict['pelvis_rot'][0]+self.sim.data.joint('hip_flexion_r').qpos.copy() )
        positionError = np.linalg.norm(obs_dict['reach_err'], axis=-1)
        #feet_v = np.linalg.norm(obs_dict['feet_v'][-3:], axis = -1) 
        com_vel = np.linalg.norm(obs_dict['com_v'], axis = -1) # want to minimize translational velocity
        comError = np.linalg.norm(obs_dict['err_com'], axis=-1)
        timeStanding = np.linalg.norm(obs_dict['time'], axis=-1)
        metabolicCost = np.sum(np.square(obs_dict['act']))/self.sim.model.na
        # act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        # Within: center of mass in between toes and calcaneous and rihgt foot left foot
        baseSupport = obs_dict['base_support'].reshape(2,4)
        centerMass = np.squeeze(obs_dict['com']) #.reshape(1,2)
        bos = mplPath.Path(baseSupport.T)
        within = bos.contains_point(centerMass)
        feet_height = np.linalg.norm(obs_dict['feet_heights'], axis = -1)
        feet_h_r = np.linalg.norm(obs_dict['feet_height_r'], axis = -1)
        com_height = obs_dict['com_height'][0]
        com_height_error = np.linalg.norm(obs_dict['com_height'][0]-0.6)
        com_bos = 1 if within else -1 # Reward is 100 if com is in bos.
        farThresh = self.far_th*len(self.tip_sids) if np.squeeze(obs_dict['time'])>2*self.dt else np.inf # farThresh = 0.5
        nearThresh = len(self.tip_sids)*.050 # nearThresh = 0.05
        # Rewards are defined ni the dictionary with the appropiate sign
        comError = comError.reshape(-1)[0]
        positionError = positionError.reshape(-1)[0]
        com_height_error = com_height_error.reshape(-1)[0]
        #feet_v = feet_v.reshape(-1)[0]
        feet_height = feet_height.reshape(-1)[0]
        feet_h_r = feet_h_r.reshape(-1)[0]
        hip_r_r = hip_rot_r.reshape(-1)[0]
        timeStanding = timeStanding.reshape(-1)[0]
        com_height = com_height.reshape(-1)[0]
        hip_add = hip_add.reshape(-1)[0]
        hip_fle = hip_fle.reshape(-1)[0]
        knee_angle = knee_angle.reshape(-1)[0]
        com_vel = com_vel.reshape(-1)[0]
        #print('hip flex reward', 5*np.exp(-0.1*(hip_flex_r - 0.6)**2) - 5, hip_flex_r)
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('positionError',        np.exp(-.1*positionError) ),#-10.*vel_dist
            #('smallErrorBonus',     1.*(positionError<2*nearThresh) + 1.*(positionError<nearThresh)),
            #('timeStanding',        1.*timeStanding), 
            ('metabolicCost',        -1.*metabolicCost),
            #('highError',           -1.*(positionError>farThresh)),
            ('centerOfMass',        1.*(com_bos)),
            ('time',                 1),
            ('com_error',             np.exp(-2.*(comError)**2)),
            ('com_height_error',     np.exp(-5*np.abs(com_height_error))),
            ('feet_height_r',          1*np.exp(-5*np.abs(feet_h_r - 0.1))-1),
            ('feet_height',             1*np.exp(-5*np.abs(feet_height))-1),
            #('feet_width',            5*np.clip(feet_width, 0.3, 0.5)),
            ('pelvis_rot_err',        5* np.exp(-pelvis_rot_err)),
            ('com_v',                  3*np.exp(-5*np.abs(com_vel))), #3*(com_bos - np.tanh(feet_v))**2), #penalize when COM_v is high
            ('hip_add',                5*np.exp(-10*(hip_add + 0.25)**2) - 5),
            ('hip_rot_r',              5*np.exp(-10*(hip_r_r)**2) - 5),
            ('knee_angle',             10*np.clip(knee_angle, 1, 1.2)),
            #('hip_flex',              10*np.clip(hip_fle, 0.4, 0.7)),
            ('hip_flex_r',             5*np.exp(-0.1*(hip_flex_r - 0.65)**2) - 5),
            # Must keys
            ('sparse',              -1.*positionError),
            ('solved',              1.*hip_flex_r>1),  # standing task succesful
            ('done',                1.*com_height < 0.3), # model has failed to complete the task 
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict
    
    def _get_feet_heights(self):
        """
        Get the height of both feet.
        """
        foot_id_l = self.sim.model.body_name2id('calcn_l')
        foot_id_r = self.sim.model.body_name2id('calcn_r')
        return np.array([self.sim.data.body_xpos[foot_id_l][2], self.sim.data.body_xpos[foot_id_r][2]])
    
    def allocate_randomly(self, perturbation_magnitude): #allocate the perturbation randomly in one of the six directions
        array = np.zeros(6)
        random_index = 1 #np.random.randint(0, 1) # 0: ML, 1: AP
        array[random_index] = perturbation_magnitude
        return array
    # generate a perturbation
    
    def generate_perturbation(self):
        M = self.sim.model.body_mass.sum()
        g = np.abs(self.sim.model.opt.gravity.sum())
        self.perturbation_time = np.random.uniform(self.dt*(0.001*self.horizon), self.dt*(0.01*self.horizon)) # between 10 and 20 percent
        # perturbation_magnitude = np.random.uniform(0.08*M*g, 0.14*M*g)
        ran = self.force_range
        if np.random.choice([True, False]):
            perturbation_magnitude = np.random.uniform(ran[0], ran[1])
        else:
            perturbation_magnitude = np.random.uniform(ran[0], ran[1])
        self.perturbation_magnitude = self.allocate_randomly(perturbation_magnitude)#[0,0,0, perturbation_magnitude, 0, 0] # front and back
        self.perturbation_duration = 20  # steps
        return
        # generate a valid target

    def generate_targets(self):
        for site, span in self.target_reach_range.items():
            sid = self.sim.model.site_name2id(site)
            sid_target = self.sim.model.site_name2id(site+'_target')
            self.sim.model.site_pos[sid] = self.sim.data.site_xpos[sid].copy() + self.np_random.uniform(low=span[0], high=span[1])
        self.sim.forward()

    def reset(self, policy_a = None, movie = True):
        self.generate_perturbation()
        key_index = random.randint(3, 5)
        qpos= self.sim.model.key_qpos[0]
        #qvel = self.sim.model.key_qvel[key_index]
        #print(key_index)
        self.robot.sync_sims(self.sim, self.sim_obsd)
        self.frames = []
        s_env = gym.make(f'mj_envs.robohive.envs.myo:{"myoLegReachFixed-v2"}')
        s_env.reset()
        self.init_pert = s_env.perturbation_magnitude[1]
        self.init_pert_t = int(100*s_env.perturbation_time)
        obs = s_env.obsdict2obsvec(s_env.obs_dict, s_env.obs_keys)[1]
        for _ in range(int(100*s_env.perturbation_time) + 46):  # Number of steps to run policy_a
            action, _ = policy_a.predict(obs, deterministic=True)
            obs, _, done, _ = s_env.step(action)
            if movie:
                geom_1_indices = np.where(s_env.sim.model.geom_group == 1)
                s_env.sim.model.geom_rgba[geom_1_indices, 3] = 0
                frame = s_env.sim.renderer.render_offscreen(width=680, height=480,camera_id=f'side_view')
                frame = (frame).astype(np.uint8)
                pert_f = self.init_pert

                # Overlay the qpos information on the image
                text = f"Perturbation: {pert_f:.2f}N"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                frame = np.flipud(frame)
        # if slow see https://github.com/facebookresearch/myosuite/blob/main/setup/README.md
                self.frames.append(frame[::-1,:,:])
        qpos = s_env.sim.data.qpos.copy()
        qvel = s_env.sim.data.qvel.copy()
        mass = s_env.sim.model.body_mass
        com_v = np.sum(s_env.sim.data.cvel.copy() *  mass.reshape((-1, 1)), axis=0) / np.sum(mass)
        com_v = com_v[-3:]
        pelvis_a = s_env.sim.data.sensordata[4:7]
        pelvis_a[-1] -= 9.8
        #print(pelvis_a)
        self.com_err = s_env.com_err
        self.sub_obs = [np.sqrt(np.sum(com_v**2)), np.sqrt(sum(x**2 for x in pelvis_a))]
        obs = super().reset(reset_qpos=qpos, reset_qvel= qvel)
        return obs
    
    def _get_ref_rotation_rew(self):
        """
        Incentivize staying close to the initial reference orientation up to a certain threshold.
        """
        #print(self.init_qpos[3:7],self.sim.data.qpos[3:7])
        target_rot = [self.target_rot if self.target_rot is not None else self.init_qpos[3:7]][0]
        return np.exp(-np.linalg.norm(5.0 * (self.sim.data.qpos[3:7] - target_rot)))
