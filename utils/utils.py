import torch as t
from utils.device import Device
import time
from datetime import datetime
from copy import deepcopy
from torchvision.utils import save_image,make_grid


def copy_weights(target, source):

    target.load_state_dict(deepcopy(source.state_dict()))



def data_to_queue(state, action, reward, next_state, terminal):

    state = t.tensor(state,device="cpu").unsqueeze(0)

    action = t.tensor([action], dtype=t.long,device="cpu").unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(1,9,96,96)

    reward = t.tensor([reward],device="cpu").unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(1,9,96,96)

    next_state = t.tensor(next_state,device="cpu").unsqueeze(0)

    terminal = t.tensor([terminal],dtype=t.bool,device="cpu").unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(1,9,96,96)

    data = t.cat((state,action,reward,next_state,terminal),dim=0)

    return data

def queue_to_data(data):

    state = data[0,:,:,:]
    action = data[1,0,0,0]
    reward = data[2,0,0,0]
    next_state = data[3,:,:,:]
    terminal = data[4,0,0,0]

    return (state,action,reward,next_state,terminal)


def checkpoint(shared_model, args,warmup_flag,lock):
    try:
        while True:
            time.sleep(args.checkpoint_frequency * 60)

            if not warmup_flag.value:
                # Save model
                now = datetime.now().strftime("%d_%m_%H_%M")

                lock.acquire()
                [shared_model.models[key].save(args.save_model+'{}_{}.pts'.format(key,now)) for key in shared_model.models]
                lock.release()

    except KeyboardInterrupt:
        print('exiting checkpoint')


def plot_autoencoder(state,predicted):

    img1 = state[:,0:3,:,:]
    img2 = state[:,3:6,:,:]
    img3 = state[:,6:9,:,:]

    img4 = predicted[:,0:3,:,:]
    img5 = predicted[:,3:6,:,:]
    img6 = predicted[:,6:9,:,:]


    save_image(make_grid(img1), './plots/img1.png')
    save_image(make_grid(img2), './plots/img2.png')
    save_image(make_grid(img3), './plots/img3.png')
    save_image(make_grid(img4), './plots/img4.png')
    save_image(make_grid(img5), './plots/img5.png')
    save_image(make_grid(img6), './plots/img6.png')


def plot_data2(batch,predicted):

    s = batch[0]
    ns = batch[2]

    img1 = s[:,0:3,:,:]
    img2 = s[:,3:6,:,:]

    img3 = ns[:,0:3,:,:]
    img4 = ns[:,3:6,:,:]


    save_image(make_grid(img1), './plots2/img1.png')
    save_image(make_grid(img2), './plots2/img2.png')
    save_image(make_grid(img3), './plots2/img3.png')
    save_image(make_grid(img4), './plots2/img4.png')
    save_image(make_grid(predicted[:,0:3,:,:]), './plots2/model1.png')
    save_image(make_grid(predicted[:,3:6,:,:]), './plots2/model2.png')
