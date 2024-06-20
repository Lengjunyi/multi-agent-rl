from matplotlib import pyplot as plt
import numpy as np

files = [

    # Advantage
    ("results\\lossy-channel.txt", "Lossy"),

]
avg = -1
for file, name in files:
    with open(file, "r") as f:
        data = f.readlines()
        # match the following pattern:
        # we got  tensor(137.7998,
        import re
        pattern = re.compile(r"(\d+(.\d+)?) (\d+(.\d+)?) tensor\((\d+\.\d+),")

        numbers = [
            float(pattern.match(line).group(5)) for line in data if pattern.match(line)
        ]

        numbers = [sum(numbers[i:i+10])/10 for i in range(0, len(numbers), 10)]

        ps = [
            float(pattern.match(line).group(1)) for line in data if pattern.match(line)
        ][::10]
        print(ps)

        # smooth the data by taking mean of 10 episodes
        # numbers = [np.mean(numbers[i:i+5]) for i in range(len(numbers)-5)]
        print(numbers)
        numbers = numbers[:100]
        print(sum(numbers[-20:])/20, "average of last 20 episodes")
        print(len(numbers), "episodes trained")
        plt.plot(ps, numbers, label=name)
        if name == "Single Agent":
            avg = sum(numbers[-20:])/20
            # draw the baseline

# plt.legend()
plt.xlabel("Packet Loss Rate (%)")
plt.ylabel("Total Reward")
plt.show()
