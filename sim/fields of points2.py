import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube
from math import pi, cos, sin , sqrt , exp , atan , atan2
from mpl_toolkits.mplot3d import Axes3D


arr = []
obs_list = []

parameter_static_x = 2.5
parameter_static_y = 2.5

# Создаем поле размером 100x100
field_x = np.linspace(0, 5, num=100) # change this if need other numbers of points
field_y = np.linspace(0, 5, num=100)
field_teta = np.linspace(0, 6.28, num=5)

# Получаем координаты всех точек в поле
points_x, points_y, points_teta = np.meshgrid(field_x, field_y, field_teta)


scaled_points_x = points_x.flatten()
scaled_points_y = points_y.flatten()
scaled_points_teta = points_teta.flatten()

# Создаем массив уникальных точек
points = list(zip(scaled_points_x , scaled_points_y, scaled_points_teta))
for x,y,teta in zip(scaled_points_x, scaled_points_y, scaled_points_teta):
    
    
    distance = ((parameter_static_x-x)*cos(teta)+(parameter_static_y-y)*sin(teta))**2/ 0.5**2 + (-(parameter_static_x-x)*sin(teta)+(parameter_static_y-y)*cos(teta))**2/ 0.5**2
    w2 = 10
    obst_stat = 5*(pi/2 + atan(w2 - distance*w2))

    arr.append([obst_stat, x, y, teta])
    obs_list.append(obst_stat)

first_column_max = max(row[3] for row in arr)
print('Max j2:', first_column_max)


arr = np.array(arr)
# indeces = np.where(arr[:,0] < 0.5)[0]
# indeces = np.random.choice(indeces, round(len(indeces) / 10)) # every 10 nummber
# print('Before: ',len(arr))
# indeces2 = np.where(arr[:,0] >= 0.5)[0]
# arr = np.concatenate ((arr[indeces], arr[indeces2]))

# print('After: ',len(arr))


with open('points_elipse.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(arr)


plt.subplot(211)
plt.plot(arr[:, 1],arr[:, 0], 'ro')
plt.subplot(212)
# Отображение точек на графике
plt.scatter(scaled_points_x , scaled_points_y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Уникальные точки на поле')
plt.grid(True)
plt.tight_layout()
plt.savefig("viewFun.pdf", dpi=300)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(arr[:,1], arr[:,2], arr[:,0])
plt.tight_layout()
plt.savefig("view3d.pdf", dpi=300)
plt.show()


