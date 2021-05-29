from models.RLBenchDQN import DQN
from models.model_accelerator import Accelerator
from simulator.RLBench import RLBench

environments = {
    'RLBench': (RLBench, [DQN,Accelerator])
}
