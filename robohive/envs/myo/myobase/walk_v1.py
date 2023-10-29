""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
SCRIPT CREATED TO TRY DIFFERENT REWARDS ON THE WALK_V0 ENVIRONMENT. 
================================================= """

import collections
import gym
import numpy as np
from robohive.envs.myo.base_v0 import BaseV0
import matplotlib.path as mplPath



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
        self._setup(**kwargs)

    def _setup(self,
            target_reach_range:dict,
            far_th = .35,
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):
        self.far_th = far_th
        self.target_reach_range = target_reach_range
        super()._setup(obs_keys=obs_keys,
                weighted_reward_keys=weighted_reward_keys,
                sites=self.target_reach_range.keys(),
                **kwargs,
                )        
        self.init_qpos = self.sim.model.key_qpos[0]

    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        self.obs_dict['qpos'] = self.sim.data.qpos[:].copy()
        self.obs_dict['qvel'] = self.sim.data.qvel[:].copy()*self.dt
        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        # reach error
        self.obs_dict['tip_pos'] = np.array([])
        self.obs_dict['target_pos'] = np.array([])
        for isite in range(len(self.tip_sids)):
            self.obs_dict['tip_pos'] = np.append(self.obs_dict['tip_pos'], self.sim.data.site_xpos[self.tip_sids[isite]].copy())
            self.obs_dict['target_pos'] = np.append(self.obs_dict['target_pos'], self.sim.data.site_xpos[self.target_sids[isite]].copy())
        self.obs_dict['reach_err'] = np.array(self.obs_dict['target_pos'])-np.array(self.obs_dict['tip_pos'])
        
        # center of mass and base of support
        xpos = {}
        for names in self.sim.model.body_names: xpos[names] = self.sim.data.xipos[self.sim.model.body_name2id(names)].copy() # store x and y position of the com of the bodies
        # Bodies relevant for hte base of support: 
        labels = ['calcn_r', 'calcn_l', 'toes_r', 'toes_l']
        x, y = [], [] # Storing position of the foot
        for label in labels:
            x.append(xpos[label][0]) # storing x position
            y.append(xpos[label][1]) # storing y position
        # CoM is considered to be the center of mass of the pelvis (for now)
        pos = self.sim.data.xipos.copy()
        mass = self.sim.model.body_mass
        com = np.sum(pos * mass.reshape((-1, 1)), axis=0) / np.sum(mass)
        self.obs_dict['com'] = com[:2]
        # Storing base of support - x and y position of right and left calcaneus and toes
        self.obs_dict['base_support'] =  [x, y]

        # print('Ordered keys: {}'.format(self.obs_keys))
        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}

        obs_dict['time'] = np.array([sim.data.time])      
        obs_dict['qpos'] = sim.data.qpos[:].copy()
        obs_dict['qvel'] = sim.data.qvel[:].copy()*self.dt
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        # reach error
        obs_dict['tip_pos'] = np.array([])
        obs_dict['target_pos'] = np.array([])
        for isite in range(len(self.tip_sids)):
            obs_dict['tip_pos'] = np.append(obs_dict['tip_pos'], sim.data.site_xpos[self.tip_sids[isite]].copy())
            obs_dict['target_pos'] = np.append(obs_dict['target_pos'], sim.data.site_xpos[self.target_sids[isite]].copy())
        obs_dict['reach_err'] = np.array(obs_dict['target_pos'])-np.array(obs_dict['tip_pos'])
        
        # center of mass and base of support
        x, y = np.array([]), np.array([])
        for label in ['calcn_r', 'calcn_l', 'toes_r', 'toes_l']:
            xpos = np.array(sim.data.xipos[sim.model.body_name2id(label)].copy())[:2] # select x and y position of the current body
            x = np.append(x, xpos[0])
            y = np.append(y, xpos[1])

        obs_dict['base_support'] = np.append(x, y)
        # CoM is considered to be the center of mass of the pelvis (for now) 
        pos = sim.data.xipos.copy()
        mass = sim.model.body_mass
        com = np.sum(pos * mass.reshape((-1, 1)), axis=0) / np.sum(mass)
        obs_dict['com'] = com[:2]

        return obs_dict

    def get_reward_dict(self, obs_dict):
        positionError = np.linalg.norm(obs_dict['reach_err'], axis=-1) # error x y and z
        # positionError = np.linalg.norm(obs_dict['reach_err'][0][0][:2], axis=-1) # error x and y
        # timeStanding = np.linalg.norm(obs_dict['time'], axis=-1)
        # vel_dist = np.linalg.norm(obs_dict['qvel'], axis=-1)
        metabolicCost = np.sum(np.square(obs_dict['act']))/self.sim.model.na
        # act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        # Within: center of mass in between toes and calcaneous and rihgt foot left foot
        baseSupport = obs_dict['base_support'].reshape(2,4)
        centerMass = np.squeeze(obs_dict['com']) #.reshape(1,2)
        bos = mplPath.Path(baseSupport.T)
        within = bos.contains_point(centerMass)

        com_bos = 1 if within else -1 # Reward is 100 if com is in bos.
        farThresh = self.far_th*len(self.tip_sids) if np.squeeze(obs_dict['time'])>2*self.dt else np.inf # farThresh = 0.5
        nearThresh = len(self.tip_sids)*.050 # nearThresh = 0.05
        # Rewards are defined ni the dictionary with the appropiate sign
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('positionError',       -1.*positionError ),#-10.*vel_dist
            ('smallErrorBonus',     1.*(positionError<2*nearThresh) + 1.*(positionError<nearThresh)),
            # ('timeStanding',        1.*timeStanding), 
            ('metabolicCost',       -1.*metabolicCost),
            ('highError',           -1.*(positionError>farThresh)),
            ('centerOfMass',        1.*(com_bos)),
            # Must keys
            ('sparse',              -1.*positionError),
            ('solved',              1.*positionError<nearThresh),  # standing task succesful
            ('done',                1.*positionError > farThresh), # model has failed to complete the task 
        ))

        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    # generate a valid target
    def generate_target_pose(self):
        for site, span in self.target_reach_range.items():
            sid =  self.sim.model.site_name2id(site+'_target')
            self.sim.model.site_pos[sid] = self.np_random.uniform(low=span[0], high=span[1])
        self.sim.forward()


    def reset(self):
        self.generate_target_pose()
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset()
        return obs
