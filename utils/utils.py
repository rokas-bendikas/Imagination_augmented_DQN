import torch as t
from utils.device import Device
import time
from datetime import datetime
from copy import deepcopy as dc
from torchvision.utils import save_image,make_grid


def copy_weights(target, source, deepcopy=False):
    if deepcopy:
        target.load_state_dict(dc(source.state_dict()))
    else:
        target.load_state_dict(source.state_dict())
        


def as_tensor(x, dtype=t.float32,device=Device.get_device()):
    return t.tensor(x, dtype=dtype, device=device,requires_grad=False)


    
def data_to_queue(state, action, reward, next_state, terminal):
    
    state = as_tensor(state,device="cpu").unsqueeze(3)
    action = as_tensor([action], t.long,device="cpu").unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(96,96,6,1)
    reward = as_tensor([reward],device="cpu").unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(96,96,6,1)
    next_state = as_tensor(next_state,device="cpu").unsqueeze(3)
    terminal = as_tensor([terminal],t.bool,device="cpu").unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(96,96,6,1)
    
    data = t.cat((state,action,reward,next_state,terminal),dim=3)
    
    return data

def queue_to_data(data):
    
    state = data[:,:,:,0]
    action = data[0,0,0,1]
    reward = data[0,0,0,2]
    next_state = data[:,:,:,3]
    terminal = data[0,0,0,4]
    
    return (state,action,reward,next_state,terminal)


def checkpoint(shared_model, args,warmup_flag):
    try:
        while True:
            time.sleep(args.checkpoint_frequency * 60)
            
            if not warmup_flag.value:
                # Save model
                now = datetime.now().strftime("%d_%m_%H_%M")
                
                [shared_model.models[key].save(args.save_model+'/{}_{}.pts'.format(key,now)) for key in shared_model.models]

    except KeyboardInterrupt:
        print('exiting checkpoint')
        
        
def plot_data(batch,predicted):
    
    s = batch[0]
    ns = batch[2]
    
    img1 = s[:,0:3,:,:]
    img2 = s[:,3:6,:,:]
    
    img3 = ns[:,0:3,:,:]
    img4 = ns[:,3:6,:,:]
    
   
    save_image(make_grid(img1), './plots/img1.png')
    save_image(make_grid(img2), './plots/img2.png')
    save_image(make_grid(img3), './plots/img3.png')
    save_image(make_grid(img4), './plots/img4.png')
    save_image(make_grid(predicted[:,0:3,:,:]), './plots/model1.png')
    save_image(make_grid(predicted[:,3:6,:,:]), './plots/model2.png')