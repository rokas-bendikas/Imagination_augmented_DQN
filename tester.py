import argparse
from environments import environments
from performer import Performer


   
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', default='RLBench', help='Environment to use for training [default = RLBench]')
    parser.add_argument('--load_model', default='./model.model', help='Path to load the model [default = [./model.model]')
    parser.add_argument('--n_tests', default=10,  type=int, help='How many times to run the simulation [default = 10]')
    args = parser.parse_args()

    SIMULATOR, NETWORK = environments[args.environment]
    model = NETWORK()
    model.load(args.load_model)
    
        
    performer = Performer(0,model,SIMULATOR)
        
    performer.perform(args)
            
        
            
        
