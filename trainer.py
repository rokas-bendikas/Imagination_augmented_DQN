import argparse
import os
import shutil
from utils.environments import environments
from utils.processes import collect, optimise
import torch.multiprocessing as mp
import ctypes
from utils.utils import checkpoint as cp



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
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

    shutil.rmtree('tensorboard', ignore_errors=True)
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('plots'):
        os.makedirs('plots')

    parser = argparse.ArgumentParser()

    parser.add_argument('--environment', default='RLBench', help='Environment to use for training [default = RLBench]')
    parser.add_argument('--save_model', default='./checkpoints', help='Path to save the model [default = "./checkpoints"]')
    parser.add_argument('--load_model', default='', help='Path to load the model [default = '']')
    parser.add_argument('--load_model_acc', default='', help='Path to load the accelerator [default = '']')
    parser.add_argument('--target_update_frequency', default=10, type=int, help='Frequency for syncing target network [default = 10]')
    parser.add_argument('--checkpoint_frequency', default=30, type=int, help='Frequency for creating checkpoints [default = 30]')
    parser.add_argument('--lr', default=1e-6, type=float, help='Learning rate for the training [default = 5e-6]')
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size for the training [default = 128]')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor for the training [default = 0.99]')
    parser.add_argument('--eps', default=0.5, type=float, help='Greedy constant for the training [default = 1]')
    parser.add_argument('--min_eps', default=0.1, type=float, help='Minimum value for greedy constant [default = 0.1]')
    parser.add_argument('--buffer_size', default=1000, type=int, help='Buffer size [default = 180000]')
    parser.add_argument('--episode_length', default=900, type=int, help='Episode length [default=900]')
    parser.add_argument('--headless', default=False, type=str2bool, help='Run simulation headless [default=False]')
    parser.add_argument('--num_episodes', default=750, type=int, help='How many episodes to plan for (used for decay parameters) [default=750]')
    parser.add_argument('--warmup', default=0, type=int, help='How many full exploration iterations [default=10]')
    parser.add_argument('--accelerator', default=True, type=str2bool, help='Use model-based accelerator [default=True]')
    parser.add_argument('--model', default="DQN", type=str, help='What model to use [default="DQN"]')
    parser.add_argument('--plot', default=True, type=str2bool, help='Plot the accelerator predictions? [default=False]')
    args = parser.parse_args()
    
    # Setup the task 
    SIMULATOR, NETWORK = environments[args.environment]
    
    # Determine number of actions available
    sim = SIMULATOR()
    args.n_actions = sim.n_actions()
    del sim

    # Create a shared model
    model_shared = NETWORK(args)
    
    # Acquire lock object
    lock = mp.Lock()
    
    # Queue for data collection
    queue = mp.Queue()
    
    # Flags
    warmup_flag = mp.Value(ctypes.c_bool,(args.warmup > 0))
    flush_flag = mp.Value(ctypes.c_bool,False)
    
    # beta value
    beta = mp.Value(ctypes.c_float,0.0)
    
    
    explorer = mp.Process(target=collect,args=(SIMULATOR,model_shared,queue,lock,args,flush_flag,warmup_flag,beta))
    optimiser = mp.Process(target=optimise,args=(model_shared,queue,lock,args,flush_flag,warmup_flag,beta))
    checkpoint = mp.Process(target=cp, args=(model_shared, args,warmup_flag))
    
    processes = [explorer,optimiser,checkpoint]
    
    [p.start() for p in processes]
    
    try:
        
        [p.join() for p in processes]
        
        
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print('<< EXITING >>')
        
    finally:
        
        queue.close()
        
        [p.kill() for p in processes]
        
        if input('Save model? (y/n): ') in ['y', 'Y', 'yes']:
            print('<< SAVING MODEL >>')
            model_shared.save()
    
            
if __name__ == '__main__':
    main()
