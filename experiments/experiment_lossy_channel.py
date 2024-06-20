STATE_SIZE = 28
QUERY_SIZE = 1
VALUE_SIZE = 4
# second config: no information is passed
VALUE_SIZE = 1


JOB_SIZE = 2

# import necessary libraries
import torch
import torch.nn as nn


import numpy as np
import torch
import torch.nn as nn

class SingleAgentNN(nn.Module):
    def __init__(self):
        super(SingleAgentNN, self).__init__()
        self.layer1 = nn.Linear(60, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 256)
        self.layer5 = nn.Linear(256, 4)

    def __str__(self):
        return f'Neural Network with input layer {self.input_layer}, hidden layer 1 {self.hidden_layer_1}, hidden layer 2 {self.hidden_layer_2}, hidden layer 3 {self.hidden_layer_3}, hidden layer 4 {self.hidden_layer_4}, and output layer {self.output_layer}'

    def __repr__(self):
        return self.__str__()
    
    def forward_pass(self, input_data):
        x = self.layer1(input_data)
        x = torch.nn.functional.leaky_relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.layer3(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.layer4(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.layer5(x)
        return x

class DataCenter():
    def __init__(self, device):
        # self.data_center_id = data_center_id
        # self.machine_num = machine_num
        # self.queue_num = queue_num
        self.state = torch.zeros(STATE_SIZE).to(device)
        # self.compressor = StateCompressor(STATE_SIZE, QUERY_SIZE, VALUE_SIZE, device=device)
        # self.dqn = DQN(STATE_SIZE, VALUE_SIZE)
        self.representations = torch.zeros(VALUE_SIZE).to(device)

        self.device = device

        # self.dqn_optimizer = torch.optim.Adam(self.dqn.parameters(), lr=0.001)
        # self.compressor_optimizer = torch.optim.Adam(self.compressor.parameters(), lr=0.001)
    
    def update(self, delta):
        with torch.no_grad():
            # reward = gains from successful job allocation - losses from queueing delay
            reward = torch.tensor(0.0).to(self.device)
            # Separate machine states and queue states
            machines = self.state[:10].view(5, 2).clone()
            queues = self.state[10:].view(6, 3).clone()

            # Update machine states
            machines[:, 1] = torch.maximum(torch.zeros_like(machines[:, 1]), machines[:, 1] - delta)
            machines[machines[:, 1] == 0, 0] = 0

            # Find available machines and assign jobs from the queue
            for i in range(queues.size(0)):
                if queues[i, 0] > 0:
                    # Find first available machine
                    available_machine_index = torch.nonzero(machines[:, 0] == 0, as_tuple=False)
                    if available_machine_index.size(0) > 0:
                        first_available = available_machine_index[0].item()
                        machines[first_available, 0] = 1
                        machines[first_available, 1] = queues[i, 1]
                        reward += queues[i, 2]
                        queues[i, :] = 0
                else:
                    break
            # move remaining jobs to the front
            queues = torch.cat((queues[queues[:, 0] > 0], queues[queues[:, 0] == 0]), 0)
            

            
            # queues[:, 2] = torch.maximum(torch.zeros_like(queues[:, 2]), queues[:, 2] - 0.1)
            queues[:, 2] *= 0.9

            # Merge the updated machine and queue states back into self.state
            self.state = torch.cat((machines.view(-1), queues.view(-1)))

            return reward
    
    def update_rep(self, remote_info):
        new_reps = self.compressor.forward_pass(self.state, remote_info)
        # print(new_reps.size(), self.representations.size())
        assert new_reps.size() == self.representations.size()
        self.representations = new_reps
    
    def get_q_values(self, reps, job):
        batch_input = torch.zeros((2, STATE_SIZE + VALUE_SIZE + JOB_SIZE + 1))
        batch_input[0] = torch.cat((self.state, self.representations, job, torch.ones(1).to(self.device)), 0)
        for i in range(reps.size(0)):
            batch_input[i+1] = torch.cat((self.state, reps[i], job, torch.zeros(1).to(self.device)), 0)
            # expand the concat into several instructions
        
        # batch_input[0, :STATE_SIZE] = self.state
        # # batch_input[0, STATE_SIZE:STATE_SIZE+VALUE_SIZE] = self.representations
        # batch_input[0, STATE_SIZE+VALUE_SIZE:STATE_SIZE+VALUE_SIZE+JOB_SIZE] = job

        batch_input.to(self.device)
        q_values = self.dqn.forward_pass(batch_input)
        return q_values

    # add job to the queue of the data center
    def add_job(self, job):
        reward = 0
        state = self.state.clone()
        for i in range(6):
            if state[10+i*3] == 0:
                state[10+i*3] = 1
                state[10+i*3+1] = job[0]
                state[10+i*3+2] = job[1]
                break
        else:
            reward -= 0.2
        self.state = state
        return reward


    
    # how to do this?
    def backprop(self):
        self.dqn_optimizer.step()
        self.compressor_optimizer.step()

        self.dqn_optimizer.zero_grad()
        self.compressor_optimizer.zero_grad()


class JobGenerator():
    def __init__(self, data_center_num) -> None:
        self.underlying_state = torch.randint(0, 2, (data_center_num,))
        self.data_center_num = data_center_num

    def generate_job(self):
        jobs = []
        for i in range(self.data_center_num):
            if torch.rand(1).item() < 0.02:
                self.underlying_state[i] = 1 - self.underlying_state[i]
                
            seed = torch.rand(1).item()
            if self.underlying_state[i] == 1:
                # choose high workload
                if seed < 0.4:
                    jobs.append((10, 1.0))
                elif seed < 0.7:
                    jobs.append((6, 0.6))
                else:
                    jobs.append((4, 0.4))
            else:
                # choose low workload
                if seed < 0.3:
                    jobs.append((4, 0.4))
                elif seed < 0.7:
                    jobs.append((3, 0.3))
                else:
                    jobs.append((2, 0.2))
        return [torch.tensor(j) for j in jobs]


def epsilon_greedy(q_values, epsilon):
    action = None
    if torch.rand(1).item() < epsilon:
        action = torch.randint(0, q_values.size(0), (1, ))
    else:
        action = torch.argmax(q_values)#.unsqueeze(0)
    q_value = q_values[action]
    return action, q_value


import random


# define IL DQN model
import torch
import torch.nn as nn

class AdvantageMARL(nn.Module):
    def __init__(self):
        super(AdvantageMARL, self).__init__()
        self.layer1 = nn.Linear(30, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, 128)
        self.layer5 = nn.Linear(128, 2)

    def __str__(self):
        return f'Neural Network with input layer {self.input_layer}, hidden layer 1 {self.hidden_layer_1}, hidden layer 2 {self.hidden_layer_2}, hidden layer 3 {self.hidden_layer_3}, hidden layer 4 {self.hidden_layer_4}, and output layer {self.output_layer}'

    def __repr__(self):
        return self.__str__()
    
    def forward_pass(self, input_data):
        x = self.layer1(input_data)
        x = torch.nn.functional.leaky_relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.layer3(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.layer4(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.layer5(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models = [AdvantageMARL().to(device) for _ in range(2)]
advantage = [AdvantageMARL().to(device) for _ in range(2)]

# Uncomment to switch between SingleAgentNN and AdvantageMARL

# load parameters
# model_data = torch.load("saved_models\\advantage_100eps_double_q.pth")
# models[0].load_state_dict(model_data["model_1"])
# models[1].load_state_dict(model_data["model_2"])
# advantage[0].load_state_dict(model_data["advantage_model_1"])
# advantage[1].load_state_dict(model_data["advantage_model_2"])


model = torch.load("saved_models\SingleAgentBaseline.pth")
# model = SingleAgentNN().to(device)
# print(model_data)
# exit()


adv_look_up = {}
N = 2
for p in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
# for p in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
# for p in [0.15, 0.1, 0.05]:
    for iter in range(2):
        print(p, iter)
        dataCenters = [DataCenter(device) for _ in range(N)]
        jobGenerator = JobGenerator(N)
        
        total_rewards = 0
        reward_history = []

        q_look_up = {}
        adv_look_up = {}
        state_look_up = {}

        for episode in range(2000):
            jobs = jobGenerator.generate_job()
            # advantages = [advantage.forward_pass(dataCenters[i].state.to(device)) for i in range(N)]
            actions = []

            # for i in range(N):
            #     q_values = models[i].forward_pass(torch.cat((dataCenters[i].state.to(device), jobs[i].to(device))).to(device))
            #     all_values = torch.zeros((N,)).to(device)
            #     all_values[i] = q_values[0]
            #     all_values[:i] = q_values[1]
            #     all_values[i+1:] = q_values[1]

            #     for j in range(N):
            #         if i == j:
            #             continue
            #         if random.random() < p or (i,j) not in adv_look_up:
            #             adv_look_up[(i,j)] = advantage[j].forward_pass(torch.cat((dataCenters[j].state.to(device), jobs[i].to(device))).to(device))
            #         adv_values = adv_look_up[(i,j)]
            #         all_values[:j] += adv_values[0]/(N-1)
            #         all_values[j:] += adv_values[0]/(N-1)
            #         all_values[j] += adv_values[1]
            #     actions.append(torch.argmax(all_values).item())

            for i in range(N):
                if random.random() < p or i not in state_look_up:
                    state_look_up[i] = dataCenters[i].state.clone()
            q_values = model.forward_pass(torch.cat((state_look_up[0], state_look_up[1], jobs[0].to(device), jobs[1].to(device))).to(device)) 
            actions = torch.argmax(q_values).item()
            actions = [actions % 2, 1 - actions // 2]
            # print(actions)

                

            reward = 0
            for i in range(N):
                if actions[i] == i:
                    reward += dataCenters[i].add_job(jobs[i])
                else:
                    jobs[i][1] *= 0.8
                    reward += dataCenters[actions[i]].add_job(jobs[i])
            for i in range(N):
                reward += dataCenters[i].update(1)
            total_rewards += reward
            reward_history.append(reward)
            # print(total_rewards, episode, end="\r")
        # print(reward_history, total_rewards)
            print(p, episode, total_rewards, end="\r")
