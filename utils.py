import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import shapely
import numpy as np
from route_distance import *
import os

def get_graph(coordinate_leftup,coordinate_rightdown,net_type = 'drive'):
    if os.path.isfile("./graph.graphml"):
        print("Loading saved graph")
        g = ox.load_graphml('./graph.graphml')
    else:
        print("Forming new graph")
        x1,y1 = coordinate_leftup
        x2,y2 = coordinate_rightdown

        g = ox.graph_from_bbox(x1,x2,y1,y2,network_type = net_type)

    return g

def visualize_graph(graph):

    ox.plot_graph(graph)

def get_nodes_edges(graph):

    nodes, edges= ox.graph_to_gdfs(graph, nodes=True, edges=True)

    return nodes,edges

def load(path,delim = '\t'):
    data = pd.read_csv(path, delimiter = delim,float_precision='round_trip')   

    return data

def form_adj_list(adj,s,t):

    coords = []

    for keys in adj:
        coords.append(keys)

    adj_list = {}
    time_step = 0
    flag = False
    start = True
    end = False
    # print(t)
    for i in range(len(coords)):
        for locs in adj[coords[i]]['x']:
            if locs not in adj_list:
                if start:
                    adj_list['start'] = {}
                    adj_list['start']['index'] = time_step-1
                    adj_list['start']['coords_t+1'] = adj[coords[i]]['x']
                    # adj_list['start']['edge_id'] = adj[coords[i]]['edge_id'][adj[coords[i]]['x'].index(locs)]
                    # adj_list['start']['geometry'] = adj[coords[i]]['edge_geometry'][adj[coords[i]]['x'].index(locs)]
                    # adj_list['start']['time'] = adj[coords[i]]['time_self']
                    # adj_list['start']['z_t'] = coords[i]
                    start = False
                if t in adj[coords[i]]['x']:
                    adj_list[locs] = {}
                    adj_list[locs]['index'] = time_step
                    adj_list[locs]['coords_t+1'] = ['end']
                    adj_list[locs]['edge_id'] = adj[coords[i]]['edge_id'][adj[coords[i]]['x'].index(locs)]
                    adj_list[locs]['geometry'] = adj[coords[i]]['edge_geometry'][adj[coords[i]]['x'].index(locs)]
                    adj_list[locs]['time'] = adj[coords[i]]['time_self']
                    adj_list[locs]['z_t'] = coords[i]
                    continue

                adj_list[locs] = {}
                adj_list[locs]['index'] = time_step
                adj_list[locs]['coords_t+1'] = adj[coords[i+1]]['x']
                adj_list[locs]['edge_id'] = adj[coords[i]]['edge_id'][adj[coords[i]]['x'].index(locs)]
                adj_list[locs]['geometry'] = adj[coords[i]]['edge_geometry'][adj[coords[i]]['x'].index(locs)]
                adj_list[locs]['time'] = adj[coords[i]]['time_self']
                adj_list[locs]['z_t'] = coords[i]

        if end:
            adj_list['end'] = {}
            adj_list['end']['coords_t+1'] = []
            adj_list['end']['index'] = time_step+1
            break
        if t in adj[coords[i+1]]['x']:
            end = True
        
        time_step += 1
        # if flag:
        #     adj_list[t] = {}
        #     adj_list[t]['index'] = time_step
        #     adj_list[t]['coords_t+1'] = []
        #     adj_list[t]['edge_id'] = adj[coords[i+1]]['edge_id'][adj[coords[i+1]]['x'].index(t)]
        #     adj_list[t]['geometry'] = adj[coords[i+1]]['edge_geometry'][adj[coords[i+1]]['x'].index(t)]
        #     adj_list[t]['time'] = adj[coords[i+1]]['time_self']
        #     adj_list[t]['z_t'] = coords[i+1] 
        #     break
        

                                                                                        

    return adj_list


def add_noise(data,param = 1e-3):
    noisy_data = {}
    for keys in data:
        # new_keys = (keys[0]+np.random.uniform(-param,param),keys[1]+np.random.uniform(-param,param))
        new_keys = (np.random.normal(keys[0],param),np.random.normal(keys[1],param)) 
        noisy_data[new_keys] = data[keys]
        # print(keys,new_keys)
    return noisy_data


def trim_dic(data,k,s,t,start_time = '',stop_time = '',mode = 0):

    # mode = 0 , trim using s,t
    # mode = 1 , trim using time

    coords = []
    avoid = [4549933771,4549933770,4549933773,4549933774,4549961253,4549961254,4549961259,4549961256]

    for keys in data:
        coords.append(keys)

    if mode == 0 :
        tm_dic = {}
        flag = False

        for i in range(len(coords)):
            flag = False
            for j in data[coords[i]]['edge_id']:
                if j[0] in avoid or j[1] in avoid:
                    flag = True
                    break
            
            if not flag: 
                if i % k == 0 or s in data[coords[i]]['x'] or t in data[coords[i]]['x']:
                    tm_dic[coords[i]] = data[coords[i]]
                
                if t in data[coords[i]]['x']:
                    break
        return s,t,tm_dic

    elif mode == 1:
        tm_dic = {}
        flag = False

        for i in range(len(coords)):
            if data[coords[i]]['time_self'] == start_time:
                s1 = data[coords[i]]['x'][0]
            if data[coords[i]]['time_self'] == stop_time:
                t1 = data[coords[i]]['x'][0]
            
        start = False
        for i in range(len(coords)):
            flag = False
            for j in data[coords[i]]['edge_id']:
                if j[0] in avoid or j[1] in avoid:
                    flag = True
                    break
            if s1 in data[coords[i]]['x']:
                start = True
            if not flag and start: 
                if i % k == 0 or (s1 in data[coords[i]]['x']) or (t1 in data[coords[i]]['x']):
                    tm_dic[coords[i]] = data[coords[i]]
                
                if t1 in data[coords[i]]['x']:
                    break
    
        return s1,t1,tm_dic

    


def get_route_dis(graph,data):

    coords = []

    for keys in data:
        coords.append(keys)


    route_distances = {}

    for loc1 in coords:
        # print(loc1)
        if len(data[loc1]['coords_t+1']) != 0:
            for loc2 in data[loc1]['coords_t+1']:
                if loc1 == 'start':
                    route_distances[(loc1,loc2)] = 0
                elif loc2 == 'end':
                    route_distances[(loc1,loc2)] = 0
                else:
                    route = shortest_path(graph,loc1,loc2\
                                        ,(data[loc1]['edge_id'][0]\
                                        ,data[loc1]['edge_id'][1],0)\
                                        ,(data[loc2]['edge_id'][0]\
                                        ,data[loc2]['edge_id'][1],0))
                    route_distances[(loc1,loc2)] = route[0]

    return route_distances




def make_hist(adj_list):
    store = []
    for keys in adj_list:
        store.append(len(adj_list[keys]['x']))

    # print(np.max(store))

    plt.hist(store,bins = 10,edgecolor = 'black')
    plt.title("Histogram of number of neighbours for each node")
    plt.xlabel("num_neighbours")
    plt.ylabel("num_nodes")
    plt.savefig('Histogram')

