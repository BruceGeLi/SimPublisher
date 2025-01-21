import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import FS10_V1
from simpub.sim.coppelia_sim_publisher import CoppeliaSimPublisher

# from rlbench.tasks.close_microwave import CloseMicrowave
from rlbench.tasks import *

class Agent(object):

    def __init__(self, action_shape):
        self.action_shape = action_shape

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)


obs_config = ObservationConfig()
obs_config.set_all(True)

env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
    obs_config=ObservationConfig(),
    headless=False)
env.launch()

pr = env._pyrep

# visual_keyword_list = ["visual", "visible"]
# visual_keyword_list = ["visual", "visible", "window"]
visual_keyword_list = ["visual", "visible"]
# visual_keyword_list = ["visual", "visible", "cup", "mug"]
# visual_keyword_list = ["visual", "visible", "box"]
# visual_keyword_list = ["visual", "visible", "microwave_frame_vis"]

agent = Agent(env.action_shape)

train_tasks = FS10_V1['train']
test_tasks = FS10_V1['test']

training_cycles_per_task = 3
training_steps_per_task = 30000
episode_length = 30000  # 40

for _ in range(training_cycles_per_task):

    # task_to_train = np.random.choice(train_tasks, 1)[0]
    # task_to_train = CloseMicrowave
    # task_to_train = CloseBox
    # task_to_train = PlaceCups
    # task_to_train = OpenWindow
    task_to_train = MeatOnGrill

    task = env.get_task(task_to_train)
    task.sample_variation()  # random variation
    CoppeliaSimPublisher(pr, host='127.0.0.1',
                         visual_layer_list=None,
                         visual_keyword_list=visual_keyword_list,
                         use_pyrep=True)
    # CoppeliaSimPublisher(pr, host='192.168.0.143',
    #                      visual_layer_list=None,
    #                      visual_keyword_list=visual_keyword_list,
    #                      use_pyrep=True)

    for i in range(training_steps_per_task):
        if i % episode_length == 0:
            print('Reset Episode')
            descriptions, obs = task.reset()
            print(descriptions)
        action = agent.act(obs)
        obs, reward, terminate = task.step(np.zeros_like(action))
    break

print('Done')
env.shutdown()