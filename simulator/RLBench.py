from simulator.base import BaseSimulator

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks import Masters
from rlbench.task_environment import InvalidActionError
from pyrep.errors import ConfigurationPathError

import numpy as np

class RLBench(BaseSimulator):
    def __init__(self,h):
        
        #64x64 camera outputs
        cam = CameraConfig(image_size=(96, 96))
        obs_config = ObservationConfig(left_shoulder_camera=cam,right_shoulder_camera=cam,wrist_camera=cam,front_camera=cam)
        obs_config.set_all(True)
        
        # delta EE control with motion planning
        #action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME)
        action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_WORLD_FRAME)
        
        #Inits
        self.env = Environment(action_mode, obs_config=obs_config, headless=h)
        self.env.launch()
        self.task = self.env.get_task(Masters)
        self.gripper_open = 1.0
        


    def reset(self):
        
        d, o = self.task.reset()
        
        return o
    

    def step(self, a, prev_state):
        
        
        # delta orientation
        d_quat = np.array([0, 0, 0, 1])
        
        # delta position
        d_pos = np.zeros(3)
        
        if a == 6:
            # gripper state
            self.gripper_open = abs(self.gripper_open - 1)
        else:
            # For positive magnitude
            if(a%2==0):
                a = int(a/2)
                d_pos[a] = 0.02
                
            # For negative magnitude
            else:
                a = int((a-1)/2)
                d_pos[a] = -0.02
        
        # Forming action as expected by the environment
        action = np.concatenate([d_pos, d_quat, [self.gripper_open]])
    
        try:
            s, r, t = self.task.step(action)
            
        
        # Handling failure in planning
        except ConfigurationPathError:
            s = prev_state
            r = 0
            t = False
        
        # Handling wrong action for inverse Jacobian
        except InvalidActionError:
            s = prev_state
            r = 0
            t = False
            
        return s, r, t
    
    
    
    @staticmethod
    def n_actions():
        return 7
        

    def __del__(self):
        print("Shutdown")
        self.env.shutdown()
    
