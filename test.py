"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import torch
import cv2
from src.tetris import Tetris
import time
import numpy as np
# from src.deep_q_network import DeepQNetwork, Dueling_DeepQNetwork

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output", type=str, default="Double_DQN/output")

    args = parser.parse_args()
    return args


def test(opt):
    score = []
    times = []
    for i in range(100):
        sum_t= 0.0      
        # time_start = time.time() #開始計時
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)
        if torch.cuda.is_available():
            # model = Dueling_DeepQNetwork()
            # target_model.load_state_dict(model.state_dict())
            # model.load_state_dict(torch.load("trained_models/New_Dueling_DQN_tetris"))
            model = torch.load("{}/Dueling_DQN_tetris".format(opt.saved_path))
            
        else:
            # model = Dueling_DeepQNetwork()
            # model.load_state_dict(torch.load("{}/New_Dueling_DQN_tetris".format(opt.saved_path), map_location=lambda storage, loc: storage))
            model = torch.load("{}/Dueling_DQN_tetris".format(opt.saved_path), map_location=lambda storage, loc: storage)
            
        model.eval()
        env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
        env.reset()
        if torch.cuda.is_available():
            model.cuda()
        opt.output ="New_Dueling_DQN/output" + str(i) + ".mp4"
        print(opt.output)
        out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"MJPG"), opt.fps,
                              (int(1.5*opt.width*opt.block_size), opt.height*opt.block_size))
        while True:
            next_steps = env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)
            if torch.cuda.is_available():
                next_states = next_states.cuda()
            predictions = model(next_states)[:, 0]
            index = torch.argmax(predictions).item()
            action = next_actions[index]
            _, done = env.step(action, render=False, video=out)
    
            if done:
                # time_end = time.time()    #結束計時
                out.release()
                
                end_img = env.render(video=None, done_img=True)
                output_img ="New_Dueling_DQN/outputimg" + str(i) + ".jpg"
                cv2.imwrite(output_img, np.array(end_img))
                
                final_score = env.score
                score.append(final_score)
                print(final_score)
                # sum_t=(time_end - time_start)  #執行所花時間
                # print('time cost', sum_t, 's')
                break
    return score, time

        


if __name__ == "__main__":
    opt = get_args()
    score, time = test(opt)
    
    #%%

print(np.argmax(score),max(score))
print(np.mean(score))
