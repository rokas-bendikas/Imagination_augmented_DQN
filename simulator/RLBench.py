from simulator.base import BaseSimulator

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks import Masters



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
        
        # Flags
        self.launched=False
        self.counts_failed = 0
        

    def launch(self):
        self.launched = True
        self.env.launch()
        self.task = self.env.get_task(Masters)
        

    def reset(self):
        
        d, o = self.task.reset()
        
        return o
    
    def step(self, action):
        
        
        s, r, t = self.task.step(action)
            
      
        return s, r, t
    
    @staticmethod
    def n_actions():
        return 7
        

    def __del__(self):
        if self.launched:
            self.env.shutdown()
    
