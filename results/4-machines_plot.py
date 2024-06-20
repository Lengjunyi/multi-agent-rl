from matplotlib import pyplot as plt
import numpy as np

files = [

    # Advantage
    ("results\\4-machines.txt", "Advantage"),

]
avg = -1
for file, name in files:
    with open(file, "r") as f:
        data = f.readlines()
        # match the following pattern:
        # we got  tensor(137.7998,
        import re
        # we got 3034.024169921875 total reward
        actions = re.compile(r"we got (\d+\.\d+) total")

        numbers = [float(actions.search(line).group(1)) for line in data if actions.search(line)]
        # smooth the data by taking mean of 10 episodes
        # numbers = [np.mean(numbers[i:i+5]) for i in range(len(numbers)-5)]
        print(numbers)
        numbers = numbers[:100]
        print(sum(numbers[-20:])/20, "average of last 20 episodes")
        print(len(numbers), "episodes trained")
        plt.plot(numbers, label=name)
        if name == "Single Agent":
            avg = sum(numbers[-20:])/20
            # draw the baseline

plt.legend()
plt.show()