import argparse
import os
import shutil
from environments import environments
from network import collect, optimise
import torch.multiprocessing as mp
from utils import checkpoint as cp




    
    
def main():
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

    shutil.rmtree('runs', ignore_errors=True)
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('trained'):
        os.makedirs('trained')
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    parser = argparse.ArgumentParser()

    parser.add_argument('--environment', default='RLBench', help='Environment to use for training [default = RLBench]')
    parser.add_argument('--save_model', default='./model.pts', help='Path to save the model [default = "/model.pts"]')
    parser.add_argument('--save_accelerator', default='./accelerator.pts', help='Path to save the model [default = "/accelerator.pts"]')
    parser.add_argument('--load_model', default='', help='Path to load the model [default = '']')
    parser.add_argument('--load_model_acc', default='', help='Path to load the model [default = '']')
    parser.add_argument('--target_update_frequency', default=5, type=int, help='Frequency for syncing target network [default = 5]')
    parser.add_argument('--checkpoint_frequency', default=30, type=int, help='Frequency for creating checkpoints [default = 10]')
    parser.add_argument('--lr', default=5e-6, type=float, help='Learning rate for the training [default = 1e-4]')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for the training [default = 64]')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor for the training [default = 0.99]')
    parser.add_argument('--eps', default=0.997, type=float, help='Greedy constant for the training [default = 0.997]')
    parser.add_argument('--min_eps', default=0.1, type=float, help='Minimum value for greedy constant [default = 0.1]')
    parser.add_argument('--buffer_size', default=100000, type=int, help='Buffer size [default = 100000]')
    parser.add_argument('--episode_length', default=500, type=int, help='Episode length [default=900]')
    parser.add_argument('--headless', default=False, type=bool, help='Run simulation headless [default=False]')
    parser.add_argument('--advance_iteration', default=0, type=int, help='By how many iteration extended eps decay [default=0]')
    parser.add_argument('--warmup', default=0, type=int, help='How many full exploration iterations [default=10]')
    args = parser.parse_args()
    
    # Setup the task 
    SIMULATOR, NETWORK = environments[args.environment]
    
    # Create a shared model
    model_shared = NETWORK[0]()
    model_shared.load(args.load_model)
    model_shared.share_memory()
    model_shared.eval()
    
    # Create a shared accelerator
    accelerator_shared = NETWORK[1]()
    accelerator_shared.load(args.load_model_acc)
    accelerator_shared.share_memory()
    accelerator_shared.eval()
    
    # Acquire lock object
    lock = mp.Lock()
    
    # Queue for data collection
    queue = mp.Queue()
    
    
    explorer = mp.Process(target=collect,args=(SIMULATOR,model_shared,accelerator_shared,queue,lock,args))
    optimiser = mp.Process(target=optimise,args=(model_shared,accelerator_shared,queue,lock,args))
    checkpoint = mp.Process(target=cp, args=(model_shared,accelerator_shared, args))
    
    explorer.start()   
    optimiser.start()
    checkpoint.start()
    
    try:
        
        explorer.join()
        optimiser.join()
        checkpoint.join()
        
        
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print('<< EXITING >>')
        
    finally:
        
        queue.close()
        explorer.kill()
        optimiser.kill()
        checkpoint.kill()
        
        if input('Save model? (y/n): ') in ['y', 'Y', 'yes']:
            print('<< SAVING MODEL >>')
            model_shared.save(args.save_model)
            accelerator_shared.save(args.save_accelerator)
    
            
if __name__ == '__main__':
    main()
