import argparse
import os
import shutil
from utils.environments import environments
from utils.processes import train_DQN
from utils.utils import checkpoint as cp
import sys
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



def main():

    shutil.rmtree('tensorboard', ignore_errors=True)
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('plots'):
        os.makedirs('plots')

    parser = argparse.ArgumentParser()

    parser.add_argument('--environment', default='RLBench', help='Environment to use for training [default = RLBench]')

    parser.add_argument('--save_model', default='./checkpoints/', help='Path to save the model [default = "./checkpoints/"]')

    parser.add_argument('--load_model', default='', help='Path to load the model [default = '']')

    parser.add_argument('--checkpoint_frequency', default=45, type=int, help='Frequency for creating checkpoints [default = 45]')

    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for the training [default = 32]')

    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor for the training [default = 0.99]')

    parser.add_argument('--min_eps', default=0.01, type=float, help='Minimum value for greedy constant [default = 0.01]')

    parser.add_argument('--buffer_size', default=300000, type=int, help='Buffer size [default = 300000]')

    parser.add_argument('--episode_length', default=350, type=int, help='Episode length [default=350]')

    parser.add_argument('--headless', default=False, type=str2bool, help='Run simulation headless [default=False]')

    parser.add_argument('--num_episodes', default=1000, type=int, help='How many episodes to plan for (used for decay parameters) [default=1000]')

    parser.add_argument('--warmup', default=25, type=int, help='How many full exploration iterations [default=25]')

    parser.add_argument('--plot', default=False, type=str2bool, help='Plot the accelerator predictions? [default=False]')

    args = parser.parse_args()

    # Setup the task
    SIMULATOR, NETWORK = environments[args.environment]

    sim = SIMULATOR()
    args.n_actions = sim.n_actions()

    model_shared = NETWORK(args)
    model_shared.share_memory()

    # Acquire lock object
    lock = mp.Lock()

    trainer = mp.Process(target=train_DQN,args=(model_shared,NETWORK,SIMULATOR,args,lock))
    checkpoints = mp.Process(target=cp,args=(model_shared,args,lock))

    trainer.start()
    checkpoints.start()

    try:
        trainer.join()
        checkpoints.join()

    except KeyboardInterrupt:
        print("Exiting!")

    finally:

        trainer.kill()
        checkpoints.kill()


        if input('Save model? (y/n): ') in ['y', 'Y', 'yes']:
            print('<< SAVING MODEL >>')
            lock.acquire()
            model_shared.save()
            lock.release()


if __name__ == '__main__':
    main()
