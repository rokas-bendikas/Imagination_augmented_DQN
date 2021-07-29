import argparse
from utils.environments import environments
from utils.performer import perform
import torch.multiprocessing as mp

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', default='RLBench', help='Environment to use for training [default = RLBench]')
    parser.add_argument('--n_tests', default=10,  type=int, help='How many times to run the simulation [default = 10]')
    parser.add_argument('--load_model', default='./checkpoints/', help='Path to load the model [default = ./checkpoints/]')
    parser.add_argument('--episode_length', default=150, type=int, help='Episode length [default=600]')
    parser.add_argument('--accelerator', default=True, type=str2bool, help='Use model-based accelerator [default=True]')
    parser.add_argument('--eps', default=0.1, type=float, help='Greedy constant for the training [default = 0.1]')
    parser.add_argument('--num_rollouts', default=7, type=int, help='How many rollouts to perform [default=7]')

    args = parser.parse_args()

    SIMULATOR, NETWORK = environments[args.environment]

    simulator = SIMULATOR(False)
    args.n_actions = simulator.n_actions()

    performer = mp.Process(target=perform,args=(NETWORK,simulator,args))

    performer.start()


    try:
        performer.join()
    except KeyboardInterrupt:
        print('<< EXITING >>')
    except Exception as e:
        print(e)
    finally:
        performer.kill()
