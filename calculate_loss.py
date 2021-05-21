import torch as t
import torch.nn.functional as f


def calculate_loss_DQN(q_network, target_network, batch, state_img, next_state_img, hyperparameters,device):
    
    state, action, reward, next_state, terminal = batch
    
    # Moving data to gpu for network pass
    state = state.to(device)
    reward = reward.to(device)
    next_state = next_state.to(device)
    terminal = terminal.to(device)
    action = action.to(device)
    
    # Concatinate state obs and model prediction
    state_augmented = t.cat((state,state_img),1)
    next_state_augmented = t.cat((next_state,next_state_img),1)
    
    
    # Target value
    with t.no_grad():
        target = reward + terminal * hyperparameters.gamma * target_network(next_state_augmented).max()
        
    # Network output    
    predicted = q_network(state_augmented).gather(1,action)
    

    return f.smooth_l1_loss(predicted, target)


def calculate_loss_accelerator(model, batch, hyperparameters, device):
    
    state, action, _, next_state, _ = batch
    
    # Moving data to gpu for network pass
    state = state.to(device)
    action = action.to(device)
    next_state = next_state.to(device)
    
    predicted = model(state,action,hyperparameters,device)
    
    loss = f.mse_loss(predicted,next_state)

    return loss
