import matplotlib.pyplot as plt
import numpy as np

# decay_steps = 50000
# learning_rate = 0.001
# end_learning_rate = 0.0000005
# power = 5

# learrning_rates = []
# for global_step in range(decay_steps):
#     global_step = np.minimum(global_step, decay_steps)
#     decayed_learning_rate = (learning_rate - end_learning_rate) * np.power((1.0 - float(global_step) / decay_steps), power) + end_learning_rate
#     learrning_rates.append(decayed_learning_rate)

# learrning_rates = np.array(learrning_rates)
# plt.plot(range(decay_steps), learrning_rates)
# plt.show()



data = []

# for fb loss refine 1
rangee = 150000
duration = 30000.0
change_value = 1.0
start_value = 0.0

starter = 100000

start_value = 0

# for fb loss refine 2
rangee = 20000
duration = 7000.0
change_value = 1.0
start_value = 0.0

starter = 10000

start_value = 0

for global_step in range(rangee):

	step = global_step - starter
	t = step/duration

	if t<0:
		t = 0
	elif t > 1:
		t = 1

	result = t*t*change_value + start_value
	data.append(result)

data = np.array(data)
plt.plot(range(rangee), data)
plt.show()
