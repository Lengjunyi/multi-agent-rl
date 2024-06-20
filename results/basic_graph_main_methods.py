from matplotlib import pyplot as plt
import numpy as np

files = [
    ("results/basic_SingleAgent.txt", "Single Agent"),

    # Independent-Q-Learning
    ("results/basic_IQL.txt", "IQL"),

    # Advantage
    ("results/basic_Advantage.txt", "Advantage"),


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
avg = -1
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
        numbers = [np.mean(numbers[i:i+5]) for i in range(len(numbers)-5)]
        print(numbers)
        # numbers = numbers[:100]
        print(sum(numbers[-20:])/20, "average of last 20 episodes")
        print(len(numbers), "episodes trained")
        plt.plot(numbers, label=name)
        if name == "Single Agent":
            avg = sum(numbers[-20:])/20
            # draw the baseline
        if name == "Advantage":
            avg_our = sum(numbers[-20:])/20

# draw the baseline?
plt.plot([avg]*140, label=f"Single-Agent Baseline (Avergage of Last 20) = {int(avg)}", linestyle="--")
plt.plot([avg_our]*140, label=f"Our result (Average of Last 20) = {int(avg_our)}", linestyle="--")

# no blank space left or right
plt.xlim(0, 140)


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
        ]#[:110]

        print(len(numbers))

        # smooth the data by taking median of 10 episodes
        # numbers = [np.median(numbers[i:i+10]) for i in range(len(numbers)-10)]
        # numbers = numbers[:100]
        print(numbers)
        print(sum(numbers[-20:])/20, "average of last 20 episodes")
        print(len(numbers), "episodes trained")
        plt.plot(numbers, label=name)
# plt.legend(files)
plt.xlabel("Episodes")
plt.ylabel("Percentage of Jobs Assigned Locally")
plt.legend()
plt.ylim(30, 100)

plt.show()