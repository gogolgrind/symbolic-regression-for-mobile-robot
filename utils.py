import os
import sys
sys.path.append('../symbolicregression/')
import numpy as np
import torch
import symbolicregression
import requests
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sympy as sp
from sympy import Number
from math import sin , cos , atan2, pi , atan
from matplotlib import colors , transforms
from sklearn.metrics import mean_squared_error as mse
from matplotlib import cm

params = {'text.usetex' : False,
          'font.size' : 36,
          'legend.fancybox':True,
          'legend.loc' : 'best',

          'legend.framealpha': 0.9,
          "legend.fontsize" : 21,
         }
plt.rcParams.update(params)
from sympy.printing import latex
from IPython.display import display
from time import perf_counter
from math import pi, cos, sin , sqrt , exp , atan , atan2

def get_latin(scale=5, parameter_static_x = 2.8, parameter_static_y = 2.5):
    import random
    
    from scipy.stats.qmc import LatinHypercube
    engine = LatinHypercube(d=2)
    samples = engine.random(n_samples)*scale
    x_coords = samples[:, 0]
    y_coords = samples[:, 1]
    arr = []
    
    for x,y in zip(x_coords, y_coords):
        distance = (parameter_static_x-x)**2/ 0.5**2 + (parameter_static_y-y)**2 / 0.5**2
        #distance = (paramters_static[0]-simX[i,0])**2/(0.2+radius)**2 + (paramters_static[0]-simX[i,1])**2 / (0.2+radius)**2
        #distance = (parameter_static_x-x)**2/(0.2+radius)**2 + (parameter_static_y-y)**2 / (0.2+radius)**2

        obst_stat = 5*(pi/2 + atan(w2 - distance*w2))
        arr.append([obst_stat, x, y])
    robot_df = pd.DataFrame(arr)
    return robot_df

def get_grid(scale=5,n_samples=300, 
             parameter_static_X = [2.5,3], 
             parameter_static_Y = [2.5,2],
             th = 0.5,
             w2=10):
    arr = []
    obs_list = []
    n_samples=int(n_samples**0.5)
    # Создаем поле размером 5x5
    field_x = np.linspace(0, scale, num=n_samples) # change this if need other numbers of points
    field_y = np.linspace(0, scale, num=n_samples)
    # Получаем координаты всех точек в поле
    points_x, points_y = np.meshgrid(field_x, field_y)
    scaled_points_x = points_x.flatten()
    scaled_points_y = points_y.flatten()
    # Создаем массив уникальных точек
    points = list(zip(scaled_points_x , scaled_points_y))
    for parameter_static_x,parameter_static_y in zip(parameter_static_X,parameter_static_Y):
        for x,y in zip(scaled_points_x, scaled_points_y):

            distance = (parameter_static_x-x)**2/ 0.5**2 + (parameter_static_y-y)**2 / 0.5**2
            obst_stat = 5*(pi/2 + atan(w2 - distance*w2))
            arr.append([obst_stat, x, y])
            obs_list.append(obst_stat)

    points = list(zip(scaled_points_x , scaled_points_y))
    arr = np.array(arr)
    indeces = np.where(arr[:,0] < th)[0]
    indeces = np.random.choice(indeces, round(len(indeces)*0.005))
    indeces2 = np.where(arr[:,0] >= th)[0]
    arr = np.concatenate ((arr[indeces], arr[indeces2]))
    robot_df = pd.DataFrame(arr)
    return robot_df

def get_robot_data(from_latin=True,from_file=False,
                   from_grid=False,
                   from_sample=False,
                   n_samples=300,
                   radius=0.3,
                   w2=10,
                   file_path='./data/hyper_random_xy.csv',
                   step=2,parameter_static_X=[3,2.5],parameter_static_Y=[2,2.5]):
    if from_file:
        robot_df=pd.read_csv(file_path,sep=',',header=None).dropna(axis=0).astype('float32')
        robot_df.columns='j2,x,y'.split(',')
        robot_df = robot_df[robot_df['j2'] > 1]
    elif from_sample:
        robot_df = robot_data_sample['j2,x,y'.split(',')]
    elif from_grid:
        robot_df = get_grid(n_samples=n_samples,parameter_static_X = parameter_static_X, 
             parameter_static_Y = parameter_static_Y)
    else:
        robot_df = get_latin(n_samples=n_samples)
    robot_df.columns='j2,x,y'.split(',')
    robot_df = robot_df.sort_values('j2')[::]
    return robot_df

def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(Number)})

def set_seed(seed=4242):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def get_transformer_model(max_input_points=20000,
                            n_trees_to_refine=500,model_path = "/tmp/model.pt"):
    
    try:
        if not os.path.isfile(model_path):
            url = "https://dl.fbaipublicfiles.com/symbolicregression/model1.pt"
            r = requests.get(url, allow_redirects=True)
            open(model_path, 'wb').write(r.content)
        if not torch.cuda.is_available():
            model = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            model = torch.load(model_path)
            model = model.cuda()
        print(model.device)
        print("Model successfully loaded!")

    except Exception as e:
        print("ERROR: model not loaded! path was: {}".format(model_path))
        print(e)

    return symbolicregression.model.SymbolicTransformerRegressor(
                            model=model,
                            max_input_points=max_input_points,
                            n_trees_to_refine=n_trees_to_refine,
                            rescale=True
                            )





def get_potential(x,w1=2,s=15):
    potential_at_x = (pi/2 + atan(w1 - (x) * w1)) * s
    return potential_at_x

def create_cost_map(num_cells = 256):

    oc_grid = np.zeros((num_cells,num_cells))

    for i in range(20,60):
        for j in range(90,130):
            oc_grid[i,j] = 100
    for i in range(100,180):
        for j in range(175,255):
            oc_grid[i,j] = 100
    for i in range(130,200):
        for j in range(30,100):
            oc_grid[i,j] = 100

    data_obst1 = np.zeros((num_cells,num_cells))

    first_obst_origin = np.array([20,90])
    first_obst_height = 40
    first_obst_width = 40
    data_obst1[first_obst_origin[0],first_obst_origin[1]] = -10

    for k in range (21):
        for i in range (first_obst_origin[0] + k,first_obst_origin[0]+first_obst_height +1 - k):
            data_obst1[i,first_obst_origin[1]+k] = -k
            data_obst1[i,first_obst_origin[1]+first_obst_width-k] = -k

        for i in range (first_obst_origin[1] + k,first_obst_origin[1]+first_obst_width - k):
            data_obst1[first_obst_origin[0]+k,i] = -k
            data_obst1[first_obst_origin[0]+first_obst_height -k, i] = -k

    for k in range (num_cells):
        for i in range (first_obst_origin[0] - k,first_obst_origin[0]+first_obst_height +1 + k):
            if(i<num_cells and i >=0):
                if((first_obst_origin[1]-k)>=0):
                    data_obst1[i,first_obst_origin[1]-k] = k
                if((first_obst_origin[1]+first_obst_width+k)<num_cells):
                    data_obst1[i,first_obst_origin[1]+first_obst_width+k] = k

        for i in range (first_obst_origin[1] - k,first_obst_origin[1]+first_obst_width + k):
            if(i<num_cells and i >=0):
                if((first_obst_origin[0]-k)>=0):
                    data_obst1[first_obst_origin[0]-k,i] = k
                if((first_obst_origin[0]+first_obst_height +k)<num_cells):
                    data_obst1[first_obst_origin[0]+first_obst_height +k, i] = k

    data_obst2 = np.zeros((num_cells,num_cells))


    second_obst_origin = np.array([100,175])
    second_obst_height = 80
    second_obst_width = 80
    data_obst2[second_obst_origin[0],second_obst_origin[1]] = -10

    for k in range (21):
        for i in range (second_obst_origin[0] + k,second_obst_origin[0]+second_obst_height +1 - k):
            data_obst2[i,second_obst_origin[1]+k] = -k
            data_obst2[i,second_obst_origin[1]+second_obst_width-k] = -k

        for i in range (second_obst_origin[1] + k,second_obst_origin[1]+second_obst_width - k):
            data_obst2[second_obst_origin[0]+k,i] = -k
            data_obst2[second_obst_origin[0]+second_obst_height -k, i] = -k

    for k in range (num_cells):
        for i in range (second_obst_origin[0] - k,second_obst_origin[0]+second_obst_height +1 + k):
            if(i<num_cells and i >=0):
                if((second_obst_origin[1]-k)>=0):
                    data_obst2[i,second_obst_origin[1]-k] = k
                if((second_obst_origin[1]+second_obst_width+k)<num_cells):
                    data_obst2[i,second_obst_origin[1]+second_obst_width+k] = k

        for i in range (second_obst_origin[1] - k,second_obst_origin[1]+second_obst_width + k):
            if(i<num_cells and i >=0):
                if((second_obst_origin[0]-k)>=0):
                    data_obst2[second_obst_origin[0]-k,i] = k
                if((second_obst_origin[0]+second_obst_height +k)<num_cells):
                    data_obst2[second_obst_origin[0]+second_obst_height +k, i] = k


    data_obst3 = np.zeros((num_cells,num_cells))

    third_obst_origin = np.array([130,30])
    third_obst_height = 70
    third_obst_width = 70
    data_obst3[third_obst_origin[0],third_obst_origin[1]] = -10

    for k in range (21):
        for i in range (third_obst_origin[0] + k,third_obst_origin[0]+third_obst_height +1 - k):
            data_obst3[i,third_obst_origin[1]+k] = -k
            data_obst3[i,third_obst_origin[1]+third_obst_width-k] = -k

        for i in range (third_obst_origin[1] + k,third_obst_origin[1]+third_obst_width - k):
            data_obst3[third_obst_origin[0]+k,i] = -k
            data_obst3[third_obst_origin[0]+third_obst_height -k, i] = -k

    for k in range (num_cells):
        for i in range (third_obst_origin[0] - k,third_obst_origin[0]+third_obst_height +1 + k):
            if(i<num_cells and i >=0):
                if((third_obst_origin[1]-k)>=0):
                    data_obst3[i,third_obst_origin[1]-k] = k
                if((third_obst_origin[1]+third_obst_width+k)<num_cells):
                    data_obst3[i,third_obst_origin[1]+third_obst_width+k] = k

        for i in range (third_obst_origin[1] - k,third_obst_origin[1]+third_obst_width + k):
            if(i<num_cells and i >=0):
                if((third_obst_origin[0]-k)>=0):
                    data_obst3[third_obst_origin[0]-k,i] = k
                if((third_obst_origin[0]+third_obst_height +k)<num_cells):
                    data_obst3[third_obst_origin[0]+third_obst_height +k, i] = k


    cost_map = np.zeros((num_cells,num_cells))

    for i in range(num_cells):
        for j in range(num_cells):
            cost_map[i,j] = 0.02 * min(data_obst1[i,j],data_obst2[i,j],data_obst3[i,j])

    cost_map = cost_map + 1


    cmap = colors.ListedColormap(['white' , 'black', 'red','green' ])
    #fig, ax = plt.subplots(figsize=(5,5))
    #ax.pcolor(oc_grid[::-1],cmap=cmap,edgecolors='w', linewidths=0.1)
    #ax.xaxis.set_visible(False)
    #ax.yaxis.set_visible(False)

    #plt.show()

    return cost_map
