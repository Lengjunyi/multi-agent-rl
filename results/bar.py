# draw bar graph

import matplotlib.pyplot as plt

# data to plot
n_groups = 2

# numbers attainable from running experiment_4.py
values = (2952, 1372*2)

plt.bar(range(n_groups), values, align='center', alpha=0.5)

plt.xticks(range(n_groups), ['Advantage Model', 'Two Single Agents'])

plt.ylabel('Total Reward')
# plt.title('Comparison of Total Reward Between Advantage Model and Two Single Agents')
plt.show()