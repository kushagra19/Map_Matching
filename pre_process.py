import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import shapely
import numpy as np
import pickle
from networkx import shortest_path as nx_shortest_path
import osmnx as ox
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.ops import substring
from osmnx.distance import nearest_edges
from osmnx.distance import great_circle_vec
from osmnx.utils_graph import get_route_edge_attributes
import multiprocessing as mp
import concurrent.futures
import os


def get_nearest(z,edge):
    x,y = z
    point = shapely.geometry.Point(y,x)
#     print(point)
    near_point = shapely.ops.nearest_points(point,edge)
#     print(near_point[1])
    dist = ox.distance.great_circle_vec(point.y,point.x,near_point[1].y,near_point[1].x)
#     print(dist)
    return near_point,dist

def get_arrays():
    with open('./saved_dicts/data_numpy','rb') as file:
        data_numpy = pickle.load(file)
    with open('./saved_dicts/edges_numpy','rb') as file:
        edges_numpy = pickle.load(file)

    return data_numpy,edges_numpy

def get_dict(point_id,thr = 200):
    
    data_numpy,edges_numpy = get_arrays()

    with open("progress.txt",'a') as file:
        file.write(str(point_id)+'\n')
    param = 1e-4
    z = (float(data_numpy[point_id][2]),float(data_numpy[point_id][3]))
    z = (z[0]+np.random.uniform(-param,param),z[1]+np.random.uniform(-param,param)) 

    dic = {}
    dic['self'] = z
    dic['x'] = []
    dic['edge_id'] = []
    dic['dist_bet_z_x'] = []
    dic['edge_geometry'] = []
    dic['edge_length'] = []
    dic['time_self'] = data_numpy[point_id][1]

    for ind in range(edges_numpy.shape[0]):
        dist = 0
        near_point,dist = get_nearest(z,edges_numpy[ind][10])

        if dist < thr and (near_point[1].y,near_point[1].x) not in dic['x']:
            dic['x'].append((near_point[1].y,near_point[1].x))
            dic['dist_bet_z_x'].append(dist)
            dic['edge_id'].append((edges_numpy[ind][0],edges_numpy[ind][1]))
            dic['edge_geometry'].append(edges_numpy[ind][10])
            dic['edge_length'].append(edges_numpy[ind][9])

    return dic

if __name__ == '__main__':
    print("Starting pre-processing")
    np.random.seed(0)

    with open('./saved_dicts/data_numpy','rb') as file:
        data_numpy = pickle.load(file)
    pid = []

    # for i in range(data_numpy.shape[0]):
    for i in range(4875,5315):
        # if i % 5 == 0:
        pid.append(i)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(get_dict,pid)

    adjoint = {}
    for result in results:
        adjoint[result['self']] = result

    with open("./saved_dicts/adjoint_list_custom",'wb') as file:
            pickle.dump(adjoint,file)
