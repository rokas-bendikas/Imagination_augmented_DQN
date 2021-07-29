from simulator.base import BaseSimulator

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks import PushBoxOntoBelt



class RLBench(BaseSimulator):
    def __init__(self,h=False):

        # Camera params
        cam = CameraConfig(image_size=(96, 96))
        self.obs_config = ObservationConfig(front_camera=cam,overhead_camera=cam,wrist_camera=cam)
        self.obs_config.set_all(True)

        # delta EE control with motion planning
        #self.action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME)
        self.action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_WORLD_FRAME)

        # Environment params
        self.env = Environment(self.action_mode, obs_config=self.obs_config, headless=h)


    def reset(self):

        d, o = self.task.reset()

        return o

    def step(self, action):

        s, r, t = self.task.step(action)

        return s, r, t

    def launch(self):

        self.env.launch()
        self.task = self.env.get_task(PushBoxOntoBelt)


    @staticmethod
    def n_actions():
        return 4

    def __del__(self):
        self.env.shutdown()
