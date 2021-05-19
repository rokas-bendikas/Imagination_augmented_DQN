from models.RLBenchDQN import DQN
from simulator.RLBench import RLBench

environments = {
    'RLBench': (RLBench, DQN)
}
