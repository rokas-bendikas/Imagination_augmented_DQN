from typing import Tuple, List, Union
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, Condition, NothingGrasped
from rlbench.backend.spawn_boundary import SpawnBoundary
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.dummy import Dummy
from rlbench.const import colors
import numpy as np
import math
import time



class OneMetConditionSet(Condition):
        def __init__(self, conditions: List[Condition])->None:

            self._conditions = conditions


        def condition_met(self)->List[bool]:

            for cond in self._conditions:
                ismet, term = cond.condition_met()
                if ismet:
                    return True, False


            return False, False

class BoxEmptyAllDetected(Condition):
        def __init__(self,objs:List[Shape])->None:
            self.objects = objs

        def condition_met(self)->List[bool]:

            if (len(self.objects)==1):
                return True, False

            return False, False




class CollisionCondition(Condition):
    def __init__(self,detector: ProximitySensor, objs:List[Shape]):
        self._detector = detector
        self._objs = objs
        self.objects_detected = list()

    def condition_met(self) -> Tuple[bool,bool]:

        for o in self._objs:

            if (self._detector.is_detected(o) and (o not in self.objects_detected)):
                self.objects_detected.append(o)
                return True,False

        return False, False


class PushBoxOntoBelt(Task):

    #################################################
################ MAIN FUNCTIONS #################
#################################################

    def init_task(self) -> None:
        # Objects used
        self.box = Shape('ConcretBlock')
        self.spawn_boundary0 = [Shape('spawn_boundary0')]
        self.spawn_boundary1 = [Shape('spawn_boundary1')]
        self.success_detector0 = ProximitySensor('success0')
        self.success_detector1 = ProximitySensor('success1')
        self.sucess_detectors = [self.success_detector0,self.success_detector1]
        self.gripper_tip = Dummy('Panda_tip')


        # Waypoint objects
        self.wp2 = Dummy('waypoint2')


        # Waypoint 1 repeats for each object
        self.register_waypoint_ability_start(1, self._move_above_object)
        self.register_waypoints_should_repeat(self._repeat)

        # Additional params
        self.idx = -1
        self.counter = 0


        # Condition flags
        self.success_reward = False
        self.collision_reward = False
        self.completion_reward = False


    def init_episode(self, index: int) -> List[str]:


        self._variation_index = index

        Dummy('BeltObjects').rotate([0,0,np.random.random()*2*math.pi])

        self.robot.arm.set_joint_positions([0.00792551040649414, 0.22508862614631653, 0.02650192379951477, -1.768791913986206, -0.015186183154582977, 2.1108152866363525, 0.8250380754470825],True)


        ################ BOX OBJECTS ###################

        # Registering all the objects in the box
        self.box_obj0 = Shape('obj0')
        self.box_obj1 = Shape('obj1')
        self.box_objects = [self.box_obj0,self.box_obj1]


        # Setting the box object colors
        for obj in self.box_objects:
            color_choice = np.random.choice(len(colors), replace=False)
            _,obj_color = colors[color_choice]
            obj.set_color(obj_color)

        # Initial box positions
        self.obj0_dist = None
        self.obj1_dist = None

        ################ BELT OBJECTS ####################

        # Registering all the belt objects
        self.belt_objects = []
        [self.belt_objects.append(Shape('obj%i'%i)) for i in range(10,46)]



        ################ SUCCESS CONDITIONS ##############


        # Registering the success conditions
        self.success_conditions = []
        for ob in self.box_objects:
            self.success_conditions.append([ob,DetectedCondition(ob, self.success_detector0),DetectedCondition(ob, self.success_detector1)])

        # Registering the success conditions for emptying the whole box successfully
        self.register_success_conditions([BoxEmptyAllDetected(self.box_objects)])



        ############### Collision sensors ############################

        # BELT OBJECT COLLISIONS

        # Conditions for every sensor
        self.collision_conditions = list()
        [self.collision_conditions.append(CollisionCondition(ProximitySensor('collision_sensor%i'%i), self._init_collidables(i))) for i in range(10,46)]

        # Putting all collision conditions to a single OneMet condition set
        self.collided_condition = OneMetConditionSet(self.collision_conditions)


        return ['Not yet implemented']




    def variation_count(self) -> int:
        return 1


#################################################
############ OVERRIDEN FUNCTIONS ################
#################################################


    def step(self) -> None:

        # If placed an object on the belt
        for items in self.success_conditions:

            ob,det0,det1 = items

            if (det0.condition_met()[0] or det1.condition_met()[0]):

                self.success_conditions.remove(items)
                self.box_objects.remove(ob)
                self.counter = 0
                self.success_reward = True


        if not self.box_objects:
            self.completion_reward = True



        if self.collided_condition.condition_met()[0]:
            self.collision_reward = True





    def is_static_workspace(self) -> bool:
        """Specify if the task should'nt be randomly placed in the workspace.

        :return: True if the task pose should not be sampled.
        """
        return True


    def reward(self) -> Union[float, None]:
        """Allows the user to customise the task and add reward shaping."""

        # Initialise award
        reward = 0.0

        # Reward for successfully placing on the belt
        if self.success_reward:
            reward += 5
            self.success_reward = False


        if self.completion_reward:
            reward += 0
            self.completion_reward = False

        # Penalising touching other objects
        if self.collision_reward:
            reward -= 3
            self.collision_reward = False


        """
        # Rewards for moving object 0
        if self.box_obj0 in self.box_objects:

            if self.obj0_dist == None:
                self.obj0_dist = self._distance_2p(self.success_detector0.get_position(),self.box_obj0.get_position())


            try:
                dx = self.obj0_dist - self._distance_2p(self.success_detector0.get_position(),self.box_obj0.get_position())
                if abs(dx) > 1e-5:
                    reward +=  np.clip(dx*25,-0.5,0.5)
                    self.obj0_dist = self._distance_2p(self.success_detector0.get_position(),self.box_obj0.get_position())
                    distance_to_center_reward = False

            except Exception:
                pass

        # Rewards for moving object 1
        if self.box_obj1 in self.box_objects:

            if self.obj1_dist == None:
                self.obj1_dist = self._distance_2p(self.success_detector1.get_position(),self.box_obj1.get_position())


            try:
                dx = self.obj1_dist - self._distance_2p(self.success_detector1.get_position(),self.box_obj1.get_position())
                if abs(dx) > 1e-5:
                    reward += np.clip(dx*25,-0.5,0.5)
                    self.obj1_dist = self._distance_2p(self.success_detector1.get_position(),self.box_obj1.get_position())
                    distance_to_center_reward = False

            except Exception:
                pass

        """



        return reward


#################################################
##################### UTILS #####################
#################################################


    def _move_above_object(self, waypoint: object) -> None:
        if len(self.box_objects) <= 0:
            raise RuntimeError('Should not be here.')

        # Resample the unreachable object
        if self.counter > 2:
            b = SpawnBoundary(self.spawn_boundaries)
            ob = self.box_objects[self.idx]
            ob.set_position(
                [0.2, 0.2, 0.2], relative_to=self.box,
                reset_dynamics=False)
            b.sample(ob, ignore_collisions=True, min_distance=0.05)
            self.counter = 0

        # Find the closest object to the tip
        dist = 1*10**5
        idx_temp = -1
        gripper_pos = self.gripper_tip.get_position()
        for i,obj in enumerate(self.box_objects):
            d = self._distance_2p(obj.get_position(),gripper_pos)
            if (d < dist and (i != self.idx)):
                idx_temp = i
                dist = d
        self.idx = idx_temp

        # Set waypoint to the closest object
        obj = self.box_objects[self.idx]
        x_obj, y_obj, z_obj = obj.get_position()
        x_wp2, y_wp2, z_wp2 = self.wp2.get_position()

        directional_vec = [x_wp2 - x_obj,y_wp2 - y_obj,z_wp2 - z_obj]
        normalised_directional_vec = directional_vec / ((directional_vec[0]**2 + directional_vec[1]**2 + directional_vec[2]**2)**0.5)

        x = x_obj - normalised_directional_vec[0] * 0.07
        y = y_obj - normalised_directional_vec[1] * 0.07
        z = z_obj - normalised_directional_vec[2] * 0.07

        waypoint.get_waypoint_object().set_position([x, y, z])
        waypoint.get_waypoint_object().set_orientation([+3.1415,0,3.1415])

        self.counter += 1



    def _repeat(self) -> bool:
        return len(self.box_objects) > 0


    def _distance_2p(self,x1: List[float], x2: List[float]) -> float:
        return math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2 + (x1[2] - x2[2])**2)


    def _init_collidables(self, index: int) -> List[Shape]:

        collidibles = []

        # Creating all object list
        [collidibles.append(obj) for obj in self.box_objects]

        # Adding robotic arm to the list of detectables
        collidibles.append(Shape('Panda_gripper_visual'))
        collidibles.append(Shape('Panda_leftfinger_visual'))
        collidibles.append(Shape('Panda_rightfinger_visual'))
        collidibles.append(Shape('Panda_link7_visual'))
        collidibles.append(Shape('Panda_link6_visual'))
        collidibles.append(Shape('Panda_link5_visual'))
        collidibles.append(Shape('Panda_link4_visual'))
        collidibles.append(Shape('Panda_link3_visual'))
        collidibles.append(Shape('Panda_link2_visual'))
        collidibles.append(Shape('Panda_link1_visual'))


        return collidibles
