import numpy as np

pitch0 = [-2.936882838231169, 13.596312363467867, 164.43331974570555, 56.98906341500281]
energy0 = [-1.353943109512329, 7.493575572967529, 46.59650969013405, 34.41540897481785]
pitch1 = [-1.91909618666375, 12.067643337106844, 167.8581406406227, 57.10709488778981]
energy1 = [-1.5235459804534912, 9.730969429016113, 43.71851257606588, 28.695237348554443]

num0 = 194720
num1 = 137099

pitch_min = min(pitch0[0], pitch1[0])
pitch_max = max(pitch0[1], pitch1[1])
energy_min = min(energy0[0], energy1[0])
energy_max = max(energy0[1], energy1[1])

pitch_mean = (pitch0[2] * num0 + pitch1[2] * num1) / (num0 + num1)
pitch_std = np.sqrt(((num0 - 1) * (pitch0[3] ** 2) + (num1 - 1) * (pitch1[3] ** 2) + (num0 * num1 * ((pitch0[2] - pitch1[2]) ** 2))/(num0 + num1)) / (num0 + num1 - 1))
energy_mean = (energy0[2] * num0 + energy1[2] * num1) / (num0 + num1)
energy_std = np.sqrt(((num1 - 1) * (energy0[3] ** 2) + (num1 - 1) * (energy1[3] ** 2) + (num0 * num1 * ((energy0[2] - energy1[2]) ** 2))/(num0 + num1)) / (num0 + num1 - 1))

pitch = [pitch_min,pitch_max,pitch_mean,pitch_std]
energy = [energy_min,energy_max,energy_mean,energy_std]

print("pitch", pitch)
print("energy", energy)