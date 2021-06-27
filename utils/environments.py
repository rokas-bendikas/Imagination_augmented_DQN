from models.network import I2A_model
from simulator.RLBench import RLBench

environments = {
    'RLBench': (RLBench, I2A_model)
}
