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


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

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
        classify_results = outputs - targets

        Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])
        buffers = model.buffer

        batch_size = buffers["state"].shape[0]
        block_num  = len(buffers)
        for i in range(batch_size):
            classify_result = classify_results[i]
            for j in range(block_num):
                # size of buffer["state"]: [block_num, batch_size, token_num, token_dim]
                state = buffers["state"][j][i]
                action = buffers["action"][j][i]
                action_prob = buffers["action_prob"][j][i]
                state_next  = buffers["state_next"][j][i]
                reward = calulate_reward(j, classify_result, action)
            
                trans = Transition(state, action, action_prob, reward, next_state)
                model.agent.store_transition(trans)

            # one image with 12 block
            # which means 12 step/transition for rl
            # after 12 transition store into agent.buffer
            # mean this trajecotry for rl is done
            # so the update should being excuted
            if len(model.agent.buffer) > model.agent.batch_size:
                model.agent.update()
        

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def caculate_reward(num_block, classify_result, action):
    reward_for_classify = 1 
    # simplest: split equally
    if classify_result = 0:
        reward_1 = reward_for_classify/12
    else:
        reward_1 = -reward_for_classify/12

    reward_for_action = 0.1
    reward_2 = 0
    for i in range(len(action)):
        # action: 0:discard token 
        #         1:keep token
        reward_2 += (1 - action[i])*reward_for_action
        
    return reward_1 + reward_2

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
