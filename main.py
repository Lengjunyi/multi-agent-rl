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
                self.underlying_state[i] = 1 - i
                
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

for n in range(0):
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
        # if random.random() < 0.02*n:
        #     reward_1 += dataCenter1.add_job(job1)
        # else:
        #     job1[1] *= 0.8
        #     reward_1 += dataCenter2.add_job(job1)
        
        # if random.random() < 0.02*n:
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
    value = (reward_1 + reward_2).item()
    print(value)
    vs.append(value)
    f.write(f"{value}\n")

    # f.write(f"{reward_1} {reward_2}\n")

f.write(f"{vs}")


disconnected_data_centers = (
"""982.88671875
977.7716674804688
971.9472045898438
970.0344848632812
972.865234375
977.2904663085938
989.1685791015625
977.6414794921875
973.8541259765625
985.7509155273438
984.637451171875
988.9825439453125
983.76220703125
988.6893310546875
991.6209106445312
976.3934326171875
986.739990234375
979.4825439453125
974.6717529296875
977.266357421875
979.830810546875
980.566162109375
980.3458862304688
986.3497924804688
977.451416015625
992.0072021484375
975.672607421875
973.6605224609375
979.1346435546875
983.1589965820312"""
)

just_avoiding_overflow_data_centers = (
"""1231.2259521484375
1243.989501953125
1233.24560546875
1227.610595703125
1210.585693359375
1220.613525390625
1232.963134765625
1231.71435546875
1233.6011962890625
1210.28857421875
1229.8204345703125
1206.9871826171875
1190.1260986328125
1209.704345703125
1261.60986328125
1219.3912353515625
1214.88037109375
1205.461669921875
1222.5242919921875
1232.6002197265625
1232.6824951171875
1240.631591796875
1243.6092529296875
1219.7412109375
1239.0947265625
1240.6329345703125
1252.828857421875
1240.4617919921875
1212.99365234375
1252.0548095703125"""
)



random_data_centers = (
"""769.5443115234375
779.619140625
825.79296875
809.602783203125
873.7089233398438
919.308837890625
913.9171142578125
984.151611328125
957.5198974609375
1013.5385131835938
1017.7440795898438
998.5821533203125
1065.005126953125
1079.9615478515625
1094.093017578125
1155.2197265625
1156.2213134765625
1216.6650390625
1198.864013671875
1263.84814453125
1244.9957275390625
1272.435546875
1276.43408203125
1324.04541015625
1318.4520263671875
1317.602294921875
1303.095947265625
1327.1240234375
1323.09716796875
1321.46142578125
1325.047607421875
1281.62548828125
1307.070068359375
1281.8626708984375
1246.170654296875
1262.0570068359375
1214.485595703125
1226.7447509765625
1235.604248046875
1194.10888671875
1159.998779296875
1149.289794921875
1101.4061279296875
1111.40576171875
1080.40087890625
1085.416015625
1057.541015625
1036.5570068359375
1013.5144653320312
1020.9532470703125
976.8478393554688"""
).split("\n")[::-1]

# import matplotlib.pyplot as plt
# # plt.plot(list(map(float, random_data_centers)))
# plt.plot(list(map(float, just_avoiding_overflow_data_centers.split("\n"))), label="Assign to other data center when queue is full")
# plt.plot(list(map(float, disconnected_data_centers.split("\n"))), label="Always assign to the local data center")
# # set y-min and y-max to 800 and 2000
# plt.ylim(800, 1400)
# plt.ylabel("Total reward")
# plt.xlabel("#th run")
# plt.legend()
# plt.show()


import matplotlib.pyplot as plt
plt.plot(list(range(0,101,2)),list(map(float, random_data_centers)), label="Random assignment")

# set y-min and y-max to 800 and 2000
plt.ylim(700, 1400)
plt.ylabel("Total reward")
plt.xlabel("% of jobs assigned to other data center")
# plt.legend()
plt.show()