import torch as t
import time
from datetime import datetime
from copy import deepcopy
from torchvision.utils import save_image,make_grid
import numpy as np


def copy_weights(target, source):
    target.load_state_dict(source.state_dict())


def rgb_to_grayscale(image_rgb):

    img1 = image_rgb[0:3,:,:]
    img2 = image_rgb[3:6,:,:]
    img3 = image_rgb[6:9,:,:]

    img1_gray = img1[0,:,:] / 3 + img1[1,:,:] / 3 + img1[2,:,:] / 3
    img2_gray = img2[0,:,:] / 3 + img2[1,:,:] / 3 + img2[2,:,:] / 3
    img3_gray = img3[0,:,:] / 3 + img3[1,:,:] / 3 + img3[2,:,:] / 3

    img_gray = np.stack((img1_gray,img2_gray,img3_gray),axis=0)

    return img_gray


def process_state(state,device):
    state_processed = np.concatenate((state.front_rgb,state.overhead_rgb,state.wrist_rgb),axis=2).transpose(2,0,1) 

    return state_processed

def process_batch(batch):

    state,action,reward,next_state,terminal = batch

    state = t.tensor(state, dtype=t.float32,device="cpu")

    action = t.tensor([action], dtype=t.long,device="cpu")

    reward = t.tensor([reward],device="cpu")

    next_state = t.tensor(next_state, dtype=t.float32,device="cpu")

    terminal = t.tensor([terminal],dtype=t.bool,device="cpu")

    return (state,action,reward,next_state,terminal)


def checkpoint(shared_model, args,lock):
    try:
        while True:
            time.sleep(args.checkpoint_frequency * 60)

            # Save model
            now = datetime.now().strftime("%d_%m_%H_%M")
            lock.acquire()
            [shared_model.models[key].save(args.save_model+'{}_{}.pts'.format(key,now)) for key in shared_model.models]
            lock.release()

    except KeyboardInterrupt:
        print('exiting checkpoint')


def plot_autoencoder(state,predicted):

    img1 = state[:,0,:,:].unsqueeze(1)
    img2 = state[:,1,:,:].unsqueeze(1)
    img3 = state[:,2,:,:].unsqueeze(1)

    img4 = predicted[:,0,:,:].unsqueeze(1)
    img5 = predicted[:,1,:,:].unsqueeze(1)
    img6 = predicted[:,2,:,:].unsqueeze(1)

    save_image(make_grid(img1), './plots/img1.png')
    save_image(make_grid(img2), './plots/img2.png')
    save_image(make_grid(img3), './plots/img3.png')
    save_image(make_grid(img4), './plots/img4.png')
    save_image(make_grid(img5), './plots/img5.png')
    save_image(make_grid(img6), './plots/img6.png')
