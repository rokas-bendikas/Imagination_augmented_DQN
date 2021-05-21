import argparse
import os
import shutil
from environments import environments
from network import DQN


    
    
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
    parser.add_argument('--save_model', default='./model.model', help='Path to save the model [default = "./model.model"]')
    parser.add_argument('--load_model', default='', help='Path to load the model [default = '']')
    parser.add_argument('--target_update_frequency', default=5, type=int, help='Frequency for syncing target network [default = 5]')
    parser.add_argument('--checkpoint_frequency', default=10, type=int, help='Frequency for creating checkpoints [default = 10]')
    parser.add_argument('--lr', default=5e-6, type=float, help='Learning rate for the training [default = 1e-4]')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for the training [default = 64]')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor for the training [default = 0.99]')
    parser.add_argument('--eps', default=0.997, type=float, help='Greedy constant for the training [default = 0.997]')
    parser.add_argument('--min_eps', default=0.1, type=float, help='Minimum value for greedy constant [default = 0.1]')
    parser.add_argument('--buffer_size', default=100000, type=int, help='Buffer size [default = 100000]')
    parser.add_argument('--episode_length', default=500, type=int, help='Episode length [default=900]')
    parser.add_argument('--max_episodes', default=1000, type=int, help='Stop after this many episodes [default=1000]')
    parser.add_argument('--headless', default=False, type=bool, help='Run simulation headless [default=False]')
    parser.add_argument('--advance_iteration', default=0, type=int, help='By how many iteration extended eps decay [default=0]')
    parser.add_argument('--warmup', default=0, type=int, help='How many full exploration iterations [default=10]')
    

    args = parser.parse_args()
    

    SIMULATOR, NETWORK = environments[args.environment]
    model = NETWORK()
    model.load(args.load_model)
    
    net = DQN(model,SIMULATOR,args)
    
    
    
    try:
        net.train()
        
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print('<< EXITING >>')
        
    finally:
        if input('Save model? (y/n): ') in ['y', 'Y', 'yes']:
            print('<< SAVING MODEL >>')
            model.save(args.save_model)
    
            
if __name__ == '__main__':
    main()
