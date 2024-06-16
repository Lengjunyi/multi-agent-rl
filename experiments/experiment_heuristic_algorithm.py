import torch


class DataCenter():
    def __init__(self, device):
        # self.data_center_id = data_center_id
        # self.machine_num = machine_num
        # self.queue_num = queue_num
        self.state = torch.zeros(28).to(device)
        # self.compressor = StateCompressor(STATE_SIZE, QUERY_SIZE, VALUE_SIZE, device=device)
        # self.dqn = DQN(STATE_SIZE, VALUE_SIZE)
        # self.representations = torch.zeros(VALUE_SIZE).to(device)

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

TIMESTEPS = 2000

device = torch.device("cuda")


f = open("output.txt", "w")

import random

    
jobGenerator = JobGenerator(2)

vs = []

for n in range(30):
    break # uncomment this line to run the experiment again
# for n in range(0, 101, 5):
    dataCenter1 = DataCenter(device)
    dataCenter2 = DataCenter(device)
    reward_1 = 0
    reward_2 = 0

    for i in range(TIMESTEPS):
        jobs = jobGenerator.generate_job()
        job1 = jobs[0]
        job2 = jobs[1]

        ############ disconnected data centers ############
        # reward_1 += dataCenter1.add_job(job1)
        # reward_2 += dataCenter2.add_job(job2)

        ############ random ############
        # if random.random() < 0.01*n:
        #     reward_1 += dataCenter1.add_job(job1)
        # else:
        #     job1[1] *= 0.8
        #     reward_1 += dataCenter2.add_job(job1)
        # if random.random() < 0.01*n:
        #     reward_2 += dataCenter2.add_job(job2)
        # else:
        #     job2[1] *= 0.8
        #     reward_2 += dataCenter1.add_job(job2)

        ############ only when overflow ############
        if dataCenter1.state[25] == 0:
            reward_1 += dataCenter1.add_job(job1)
        else:
            job1[1] *= 0.8
            reward_1 += dataCenter2.add_job(job1)
        
        if dataCenter2.state[25] == 0:
            reward_2 += dataCenter2.add_job(job2)
        else:
            job2[1] *= 0.8
            reward_2 += dataCenter1.add_job(job2)
        

        reward_1 += dataCenter1.update(1)
        reward_2 += dataCenter2.update(1)

        # f.write(f"{i}: {dataCenter1.state} {dataCenter2.state} {reward_1} {reward_2}\n")
        print(reward_1 + reward_2, i, end="\r")
    value = (reward_1 + reward_2).item()
    print(value)
    vs.append(value)
    f.write(f"{value}\n")

    # f.write(f"{reward_1} {reward_2}\n")

f.write(f"{vs}")


disconnected_data_centers = (
"""1084.3883056640625
1097.19140625
1126.988525390625
1125.7257080078125
1097.2052001953125
1081.73193359375
1158.090087890625
1151.8619384765625
1170.876708984375
1163.0689697265625
1048.3648681640625
1118.810791015625
1089.93115234375
1114.591796875
1126.797607421875
1118.2529296875
1159.277587890625
1135.55224609375
1127.36083984375
1139.1766357421875
1077.44482421875
1116.06689453125
1053.217529296875
1187.486572265625
1127.9945068359375
1102.984619140625
1094.23779296875
1126.914794921875
1148.21337890625
1103.16455078125"""
)

just_avoiding_overflow_data_centers = (
"""1231.253173828125
1166.239501953125
1159.52685546875
1136.553466796875
1156.74658203125
1154.446533203125
1173.330322265625
1178.97705078125
1207.3768310546875
1126.309326171875
1190.70703125
1168.210205078125
1165.028076171875
1196.41748046875
1178.97998046875
1212.8079833984375
1110.5689697265625
1196.615478515625
1218.462158203125
1093.0771484375
1155.9566650390625
1162.60400390625
1067.654541015625
1206.18994140625
1062.2344970703125
1177.813720703125
1144.771728515625
1201.998779296875
1190.2064208984375
1185.33740234375"""
)



random_data_centers = (
"""889.2998046875
897.721923828125
915.5025634765625
1011.1196899414062
991.6707763671875
1051.8369140625
1079.1143798828125
1094.6954345703125
1138.109130859375
1108.37060546875
1139.7490234375
1189.9080810546875
1124.56689453125
1190.684814453125
1187.855712890625
1172.717041015625
1147.399169921875
1148.17431640625
1115.8031005859375
1143.048828125
1071.009521484375"""
).split("\n")[::-1]

import matplotlib.pyplot as plt
# plt.plot(list(map(float, random_data_centers)))
plt.plot(list(map(float, just_avoiding_overflow_data_centers.split("\n"))), label="Assign to other data center when queue is full")
plt.plot(list(map(float, disconnected_data_centers.split("\n"))), label="Always assign to the local data center")
# set y-min and y-max to 800 and 2000
plt.ylim(800, 1400)
plt.ylabel("Total reward")
plt.xlabel("#th run")
plt.legend()
plt.show()


import matplotlib.pyplot as plt
plt.plot(list(range(0,101,5)),list(map(float, random_data_centers)), label="Random assignment")

# set y-min and y-max to 800 and 2000
plt.ylim(700, 1400)
plt.ylabel("Total reward")
plt.xlabel("% of jobs assigned to other data center")
# plt.legend()
plt.show()