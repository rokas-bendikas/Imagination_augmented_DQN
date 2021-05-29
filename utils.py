import torch as t
from device import Device
import time
from datetime import datetime



def copy_gradients(target, source):
    for shared_param, param in zip(target.parameters(), source.parameters()):
        if param.grad is not None:
            shared_param._grad = param.grad.clone().cpu()


def as_tensor(x, dtype=t.float32):
    return t.tensor(x, dtype=dtype, device=Device.get_device(),requires_grad=False)


    
def data_to_queue(state, action, reward, next_state, terminal):
    
    state = as_tensor(state).unsqueeze(3)
    action = as_tensor([action], t.int64).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(96,96,6,1)
    reward = as_tensor([reward]).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(96,96,6,1)
    next_state = as_tensor(next_state).unsqueeze(3)
    terminal = as_tensor([terminal],t.bool).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(96,96,6,1)
    
    data = t.cat((state,action,reward,next_state,terminal),dim=3)
    
    return data

def queue_to_data(data):
    
    state = data[:,:,:,0]
    action = data[0,0,0,1]
    reward = data[0,0,0,2]
    next_state = data[:,:,:,3]
    terminal = data[0,0,0,4]
    
    return (state,action,reward,next_state,terminal)


def checkpoint(shared_model, shared_accelerator, args):
    try:
        while True:
            time.sleep(args.checkpoint_frequency * 60)

            # Save model
            now = datetime.now().strftime("%d_%m_%H_%M")
            shared_model.save('trained/model_{}.pts'.format(now))
            shared_accelerator.save('trained/accelerator_{}.pts'.format(now))

    except KeyboardInterrupt:
        print('exiting checkpoint')