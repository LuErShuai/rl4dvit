import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from env_deit import DeitEnv
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# env = gym.make("CartPole-v0").unwrapped

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

class Agent:
    def __init__(self, env, condition):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.lr = 0.0001
        if os.path.exists('model/actor.pkl'):
            self.actor = torch.load('model/actor.pkl')
            print('Actor Model loaded')
        else:
            self.actor = Actor(self.state_size, self.action_size).to(device)
        if os.path.exists('model/critic.pkl'):
            self.critic = torch.load('model/critic.pkl')
            print('Critic Model loaded')
        else:
            self.critic = Critic(self.state_size, self.action_size).to(device)

        self.condition = condition

    def train(n_epoch=100):
    
        optimizerA = optim.Adam(actor.parameters())
        optimizerC = optim.Adam(critic.parameters())
        for epoch in range(n_epoch):
            state = env.reset()
            log_probs = []
            values = []
            rewards = []
            masks = []
            entropy = 0
            env.reset()
    
            for i in count():
                env.render()

                # # get obs from deit
                # with self.condition:
                #     print("RL : Deit, d u need a mask?")
                #     condition.notify()
                #     condition.wait()
                #     # print("Deit : yes!")

                #     print("RL   : pass me the state please.")
                #     conditon.notify()
                #     condition.wait()
                #     # print("Deit : here u go.")

                #     state = self.get_state_from_deit()
                #     action = choose_action(state)
                #     
                #     set_mask_for_deit(action)
                #     print("RL   : got it! here is the mask that you want.")
                #     condition.notify()
                #     condition.wait()


                state = torch.FloatTensor(state).to(device)
                dist, value = actor(state), critic(state)
    
                action = dist.sample()


                # # get obs_next from deit
                # with self.condition:
                #     # print("Deit : step one block with mask done! obs_next give u.")
                #     condition.wait()
                #     state_ = self.get_state_from_deit()
                #     print("RL   : got it! done!")
                #     condition.notify()




                next_state, reward, done, _ = env.step(action.cpu().numpy())
    
                log_prob = dist.log_prob(action).unsqueeze(0)
                entropy += dist.entropy().mean()
    
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
                masks.append(torch.tensor([1-done], dtype=torch.float, device=device))
    
                state = next_state
    
                if done:
                    print('Iteration: {}, Score: {}'.format(epoch, i))
                    break
    
    
            next_state = torch.FloatTensor(next_state).to(device)
            next_value = critic(next_state)
            returns = compute_returns(next_value, rewards, masks)
    
            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns).detach()
            values = torch.cat(values)
    
            advantage = returns - values
    
            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()
    
            optimizerA.zero_grad()
            optimizerC.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            optimizerA.step()
            optimizerC.step()
        torch.save(actor, 'model/actor.pkl')
        torch.save(critic, 'model/critic.pkl')
        env.close()

if __name__ == '__main__':
    condition = threading.Condition()
    train(100, condition)
