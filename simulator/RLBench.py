from simulator.base import BaseSimulator

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks import Masters


import numpy as np

class RLBench(BaseSimulator):
    def __init__(self,h=False):
        
        # Camera params
        cam = CameraConfig(image_size=(96, 96))
        self.obs_config = ObservationConfig(left_shoulder_camera=cam,right_shoulder_camera=cam,wrist_camera=cam,front_camera=cam)
        self.obs_config.set_all(True)
        
        # delta EE control with motion planning
        #self.action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME)
        self.action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_WORLD_FRAME)
        
        # Environment params
        self.env = Environment(self.action_mode, obs_config=self.obs_config, headless=h)
        
        # Shared vars
        self.gripper_open = 1.0
        
        self.launched=False
        

    def launch(self):
        self.launched = True
        self.env.launch()
        self.task = self.env.get_task(Masters)
        

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
    
        
        s, r, t = self.task.step(action)
            
          
        return s, r, t
    
    
    
    @staticmethod
    def n_actions():
        return 7
        

    def __del__(self):
        if self.launched:
            self.env.shutdown()
    
