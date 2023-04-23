# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

from collections import namedtuple

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1
    model.agent.reward_one_epoch = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
                    
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)
        
        # size of loss:[batch_size]
        loss_value = loss.item()

        # train agent
        # classify_results = outputs - targets
        _, outputs_max_index = outputs.max(dim=1)
        _, targets_max_index = targets.max(dim=1)
        # self.buffer = 
        # {
        #     "state":[], -> [block_num, batch_size, token_num, token_dim]
        #     "state_next":[], 
        #     "action":[],
        #     "action_prob":[]
        # }
        Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])
        buffers = model.buffer

        batch_size = buffers["state"][0].shape[0]
        token_num  = buffers["state"][0].shape[1]
        block_num  = len(buffers["state"])

        model.agent.reward_one_batch = 0
        for i in range(batch_size):
            if outputs_max_index[i] == targets_max_index[i]:
                classify_correct = True 
            else:
                classify_correct = False

            for j in range(token_num):
                del model.agent.buffer[:] # clear experience
                for k in range(block_num):
                    mask = buffers["mask"][k][i][j]
                    if mask == 0:
                        break
                    # size of buffers["state"]: [block_num, batch_size, token_num, token_dim]
                    state = buffers["state"][k][i][j]
                    action = buffers["action"][k][i][j]
                    action_prob = buffers["action_prob"][k][i][j]
                    state_next = buffers["state_next"][k][i][j]

                    reward = caculate_reward_per_step(j, classify_correct,
                                                       action)
                    trans = Transition(state.detach().cpu().numpy(), action, action_prob,
                                       reward, state_next.detach().cpu().numpy())
                    model.agent.store_transition(trans)

                    model.agent.reward_one_epoch += reward
                    model.agent.reward_one_batch += reward
                
                a = len(model.agent.buffer)
                b = model.agent.batch_size
                # if len(model.agent.buffer) > model.agent.batch_size:
                #     model.agent.update()
                if len(model.agent.buffer) > 0:
                    model.agent.update()

        print('test') 
        a = model.agent.training_step
        # if utils.is_main_process() and model.agent.training_step > 50000:
        if model.agent.training_step > 5000:
            model.agent.save_param()
            print("save ppo weight")
            # return



            # for j in range(block_num):
            #     # size of buffer["state"]: [block_num, batch_size, token_num, token_dim]
            #     # size of state: [token_num, token_dim]
            #     state = buffers["state"][j][i]
            #     action = buffers["action"][j][i]
            #     action_prob = buffers["action_prob"][j][i]
            #     state_next  = buffers["state_next"][j][i]
            #     # reward = caculate_reward(j, classify_correct, action)

            #     token_num = state.shape[0]
            #     for k in range(token_num):
            #         reward = caculate_reward_per_token(j, classify_correct,
            #                                            action[k])
            #         trans = Transition(state[k], action[k], action_prob[k],
            #                            reward, next_state[k])
            #         model.agent.store_transition(trans)

            #         # one image with 12 block
            #         # which means 12 step/transition for rl
            #         # after 12 transition store into agent.buffer
            #         # mean this trajecotry for rl is done
            #         # so the update should being excuted
            #         if len(model.agent.buffer) > model.agent.batch_size:
            #             model.agent.update()
        

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if args.train_deit:
            optimizer.zero_grad()

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['reward_batch'].update(model.agent.reward_one_batch, n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def caculate_reward_per_step(num_block, classify_correct, action):
    reward_for_classify = 24 
    reward_for_action = 1
    # simplest: split equally
    if classify_correct:
        reward_1 = reward_for_classify/12
    else:
        reward_1 = -reward_for_classify/12

    reward_2 = (1 - action)*reward_for_action

    return reward_1 + reward_2

def caculate_reward(num_block, classify_correct, action):
    # size of action: [token_num] -> 197
    # action for 197 tokens in one image

    # if image classdify correctly, reward = reward_1
    # if the token is discarded,    reward = reward_2

    reward_for_classify = 24 
    # simplest: split equally
    if classify_correct:
        reward_1 = reward_for_classify/12
    else:
        reward_1 = -reward_for_classify/12

    reward_for_action = 1
    reward = torch.empty(action.shape, device=action.device)
    for i in range(len(action)):
        # action: 0:discard token 
        #         1:keep token
        reward_2 = 0
        reward_2 += (1 - action[i])*reward_for_action

        reward_total = reward_1 + reward_2
        reward[i] = reward_total 
        
    return reward

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
