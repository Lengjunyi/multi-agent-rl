from matplotlib import pyplot as plt
import numpy as np

files = [
    ("results/SingleAgent.txt", "Single Agent"),
    ("results\IQL.txt", "IQL"),
    # # "results\StateCompression.txt",
    # # ("results\IndependentLearnersLocalReward.txt", "IQL (Local)"),
    # ("results\AdvantageWithCompetitionDoubleQ.txt", "110%"),
    # ("results\AdvantageWithoutRewardSharing.txt", "100%"),
    # ("results\AdvantageWith70pcLocalReward.txt", "70%"),
    # ("results\AdvantageWithRewardSharing.txt", "50%"),
    # # ("results/Single2.txt", "Single Agent"),
    # ("advantage_no_sharing_double_q.txt", "adv"),
    
    # ("input.txt", "default")
]

for file, name in files:
    with open(file, "r") as f:
        data = f.readlines()
        # match the following pattern:
        # we got  tensor(137.7998,
        import re
        pattern = re.compile(r"we got  tensor\((\d+\.\d+),")
        # pattern = re.compile(r"tensor\((\d+\.\d+),")
        # extract the number if exists
        numbers = [
            float(pattern.match(line).group(1)) for line in data if pattern.match(line)
        ]

        actions = re.compile(r"we got  tensor\((\d+\.\d+),")
        # smooth the data by taking mean of 10 episodes
        # numbers = [np.mean(numbers[i:i+5]) for i in range(len(numbers)-5)]
        print(numbers)
        print(sum(numbers[-20:])/20, "average of last 20 episodes")
        print(len(numbers), "episodes trained")
        plt.plot(numbers, label=name)
# plt.legend(files)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.legend()

plt.show()


for file, name in files:
    with open(file, "r") as f:
        data = f.readlines()
        # match the following pattern:
        # we got  tensor(137.7998,
        import re
        # match actions tensor([3253.
        pattern = re.compile(r"actions tensor\(\[(\d+)\.")
        # pattern = re.compile(r"actions tensor([(\d+\.\d+")
        # extract the number if exists
        numbers = [
            float(pattern.search(line).group(1))/4000*100 for line in data if pattern.search(line)
        ][:110]

        print(len(numbers))

        # smooth the data by taking median of 10 episodes
        # numbers = [np.median(numbers[i:i+10]) for i in range(len(numbers)-10)]
        numbers = [np.mean(numbers[i:i+3]) for i in range(len(numbers)-3)]
        print(numbers)
        print(sum(numbers[-20:])/20, "average of last 20 episodes")
        print(len(numbers), "episodes trained")
        plt.plot(numbers, label=name)
# plt.legend(files)
plt.xlabel("Episodes")
plt.ylabel("Percentage of Jobs Assigned Locally")
plt.legend()
plt.ylim(40, 100)

plt.show()