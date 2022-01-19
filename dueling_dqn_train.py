"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.deep_q_network import DeepQNetwork, Dueling_DeepQNetwork
from src.tetris import Tetris
from collections import deque

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboardss")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    max_score = 0
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    # if os.path.isdir(opt.log_path):
    #     shutil.rmtree(opt.log_path)
    # os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    model = Dueling_DeepQNetwork()
    
    # model = DeepQNetwork()
    # target_model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    state = env.reset()
    if torch.cuda.is_available():
        # target_model.cuda()
        model.cuda()
        state = state.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0
    while epoch < opt.num_epochs:
        next_steps = env.get_next_states()
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
            # print(predictions.shape)
        model.train()
        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()
            # print("index",index)

        next_state = next_states[index, :]
        action = next_actions[index]

        reward, done = env.step(action, render=False)
        
        
        # next_next_steps = env.get_next_states()
        # next_next_actions, next_next_states = zip(*next_next_steps.items())
        # next_next_states = torch.stack(next_next_states)

        if torch.cuda.is_available():
            next_state = next_state.cuda()
        replay_memory.append([state, reward, next_state, next_states, done])
        if done:
            final_score = env.score
            if final_score > max_score:
                print(final_score)
                max_score = final_score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
            if torch.cuda.is_available():
                state = state.cuda()
        else:
            state = next_state
            continue
        if len(replay_memory) < opt.replay_memory_size / 10:
            continue
        
        
        # target_model.load_state_dict(model.state_dict())
        
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, next_states_batch, done_batch = zip(*batch)
        # state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))
        
        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda() #action
            # next_next_states_batch = next_next_states_batch.cuda()
        q_values = model(state_batch)#eval_total

        model.eval()
        ##DDQN
        with torch.no_grad():
            
            next_prediction_batch = model(next_state_batch)
            
            # print("max",max_action_next)
            # current_prediction_batch = model(next_state_batch)
            # target_prediction_batch = target_model(next_state_batch)
            '''
            max_action_next=[]
            target_prediction_batch = []
            # print(next_states_batch)
            for i in next_states_batch:
                # print(i.size())
                if torch.cuda.is_available():
                    i = i.cuda()
                max_action_next.append(torch.argmax(model(i)).item())
                target_prediction_batch.append(target_model(i))
                
            # print(max_action_next)  
            # print(current_prediction_batch)
            # max_action_next = torch.Tensor(max_action_next)
            # max_action_next = torch.argmax(current_prediction_batches).item()          
            # print(target_prediction_batch.shape)
            '''
        model.train()
       
        y_batch_list = []
        
        
        # print(target_prediction_batch.shape)
        for reward, done, prediction in zip(reward_batch, done_batch, next_prediction_batch):
            # print(done)
            if done:
                 y_batch_list.append(reward)
            else:

                y_batch_list.append(reward + opt.gamma * prediction)
        '''       
        for reward, done, prediction, b_action in zip(reward_batch, done_batch, target_prediction_batch, max_action_next):
            # print(done)
            if done:
                 y_batch_list.append(reward)
            else:
                # print(b_action)
                # target_q_value = prediction[b_action]
                # q_eval_argmax = q_values.max(1)[1].view(opt.batch_size, 1)
                # target_value = target_prediction_batch.gather(1, q_eval_argmax).view(opt.batch_size, 1)#q_max
                
                y_batch_list.append(reward + opt.gamma * prediction)
        '''
        
        y_batch = torch.cat(tuple(y_batch_list))[:, None]
        
                
        # y_batch = torch.cat(
        #     tuple(reward if done else reward + opt.gamma * prediction  print(prediction) for reward, done, prediction in
        #           zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            epoch,
            opt.num_epochs,
            action,
            final_score,
            final_tetrominoes,
            final_cleared_lines))
        writer.add_scalar('Train/New_Dueling_DQN_Score', final_score, epoch - 1)
        writer.add_scalar('Train/New_Dueling_DQN_Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/New_Dueling_DQN_Cleared lines', final_cleared_lines, epoch - 1)

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/New_Dueling_DQN_tetris_{}".format(opt.saved_path, epoch))

    torch.save(model, "{}/New_Dueling_DQN_tetris".format(opt.saved_path))
    torch.save(model.state_dict(),"{}/New_Dueling_DQN_tetris.pkl".format(opt.saved_path))
    return max_score


if __name__ == "__main__":
    opt = get_args()
    max_score = train(opt)

#%%

print(max_score)

