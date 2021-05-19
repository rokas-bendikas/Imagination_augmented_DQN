import torch as t
import torch.nn.functional as f


def calculate_loss_DQN(q_network, target_network, batch, hyperparameters,device):
    
    state, action, reward, next_state, terminal = batch
    
    # Moving data to gpu for network pass
    state = state.to(device)
    reward = reward.to(device)
    next_state = next_state.to(device)
    terminal = terminal.to(device)
    action = action.to(device)
    
    
    # Target value
    with t.no_grad():
        target = reward + terminal * hyperparameters.gamma * target_network(next_state).max()
        
    # Network output    
    predicted = q_network(state).gather(1,action)
    

    return f.smooth_l1_loss(predicted, target)


def calculate_loss_accelerator(model, batch, hyperparameters, device):
    
    state, _, _, next_state, _ = batch
    
    # Moving data to gpu for network pass
    state = state.to(device)
    next_state = next_state.to(device)
    
    predicted = model(state)
    
    loss = f.kl_div(predicted,next_state,reduction='batchmean')

    return loss
