from matplotlib import pyplot as plt


files = [
    "results/SingleAgentBaseline.txt",
    # "results\IndependentLearnersLocalReward.txt",
    "results\IndependentLearners.txt",
    "results\StateCompression.txt",
    "input.txt"
]

for file in files:
    with open(file, "r") as f:
        data = f.readlines()
        # match the following pattern:
        # we got  tensor(137.7998,
        import re

        pattern = re.compile(r"we got  tensor\((\d+\.\d+),")
        # extract the number if exists
        numbers = [
            float(pattern.match(line).group(1)) for line in data if pattern.match(line)
        ]
        print(numbers)
        print(sum(numbers[-20:])/20, "average of last 20 episodes")
        print(len(numbers), "episodes trained")
        plt.plot(numbers)
plt.legend(files)

plt.show()
