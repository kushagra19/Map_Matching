#Libraries required
import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import shapely
import numpy as np
import pickle
import queue_p
from keplergl import KeplerGl
import statistics

#abbreviate frequently used functions
from networkx import shortest_path as nx_shortest_path
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.ops import substring
from osmnx.distance import nearest_edges
from osmnx.distance import great_circle_vec
from osmnx.utils_graph import get_route_edge_attributes

class HMM():
    #Parameters(in meters)
    def __init__(self,adjacency_list,route_distances,SIGMA_Z = 3.74,BETA = 0.10841):
        self.adjacency_list = adjacency_list
        self.route_distances = route_distances
        # l = []
        # file = open('path', 'rb')

        # # dump information to that file
        # path = pickle.load(file)

        # # close the file
        # file.close()
        # pred = {}
        # for i in range(len(path)):
        #     if i != len(path)-1:
        #         pred[path[i]] = path[i+1]
        #     else:
        #         pred[path[i]] = None

        # for i in path:
        #     if pred[i] != None and pred[i] != 'start' and pred[i] != 'end' and i != 'start' and i != 'end':
        #         l.append(self.haversine_dist(i,self.adjacency_list[i]['z_t'])*1.4826)
        # SIGMA_Z = statistics.median(l)

        # l = []
        # for i in path:
        #     if pred[i] != None and pred[i] != 'start' and pred[i] != 'end' and i != 'start' and i != 'end' :
        #         l.append(np.abs(self.haversine_dist(self.adjacency_list[i]['z_t'],self.adjacency_list[pred[i]]['z_t']) - route_distances[(pred[i],i)])/np.log(2))
        # BETA = statistics.median(l)

        
        self.SIGMA_Z = SIGMA_Z
        self.BETA = BETA
        

    def haversine_dist(self,u,v):
        return ox.distance.great_circle_vec(u[0],u[1],v[0],v[1])

    def construct_path(self,pred,s,t):

        route_coords = []
        while t != None:
            prev_coord = pred[t]
            route_coords.append(t)
            t = prev_coord
        return route_coords

    # A gaussian distribution
    def emission_prob(self,u):
        if u == 'start' or u == 'end':
            return 0

        u_z = self.adjacency_list[u]['z_t']


        c = 1 / (self.SIGMA_Z * np.sqrt(2 * np.pi))
        c1 = 0.5*(1/self.SIGMA_Z**2)

        return (-c1*ox.distance.great_circle_vec(u[0],u[1],u_z[0],u_z[1])**2) + np.log(c)

    # A empirical distribution
    def transition_prob(self,u,v):
        if u == 'start' or v == 'end':
            return 0

        z_u = self.adjacency_list[u]['z_t']
        z_v = self.adjacency_list[v]['z_t']
        
        c = 1 / self.BETA
        

        delta = np.abs(self.route_distances[(u, v)] - ox.distance.great_circle_vec(z_u[0],z_u[1],z_v[0],z_v[1]))
        return (-delta*c) + np.log(c)
    
    def viterbi_search(self,s,t):
        # Initialize joint probability for each node

        print("SIGMA:", self.SIGMA_Z)
        print("BETA:", self.BETA)
        joint_prob = {}
        lt = []

        for u in self.adjacency_list:
            joint_prob[u] = -np.inf
        predecessor = {}

        q = queue_p.queue()

        joint_prob['start'] = 1
        q.push('start')
    
        predecessor['start'] = None
        i = 0
        while not q.Empty():
            #Book-keeping
            if i % 1000 == 0:
                print(i)
                with open("jp",'wb') as file:
                    pickle.dump(predecessor,file)
            
            # Extract node u
            u = q.pop()
            # append nodes that had a chance on queue
            lt.append(u)
 
            if u == 'end': break
            for v in self.adjacency_list[u]['coords_t+1']:
                if self.adjacency_list[v]['index'] - self.adjacency_list[u]['index'] == 1:
                    #calculate new_prob        
                    new_prob = joint_prob[u] + self.transition_prob(u,v) + self.emission_prob(v)
                
                    if joint_prob[v] < new_prob:
                        joint_prob[v] = new_prob
                        predecessor[v] = u
                    if not q.in_q(v):
                        q.push(v)
            i+=1
        return joint_prob, predecessor

    def get_route(self,predecessor,s,t,visualize = True):
        #back tracking
        path = self.construct_path(predecessor,s,'end')
        # file = open('path', 'wb')

        # # dump information to that file
        # pickle.dump(path, file)

        # # close the file
        # file.close()
        #data pre-processing for keplergl
        path.remove('start')
        path.remove('end')
        np_df_x,np_df_y,np_df_t = [],[],[]
        for coord in path:
            np_df_x.append(coord[0])
            np_df_y.append(coord[1])
            np_df_t.append(self.adjacency_list[coord]['time'])

        np_df = np.array([np_df_x,np_df_y,np_df_t]).T
        df = pd.DataFrame(np_df, columns = ['Latitude','Longitude','Time'])
        print(df.head())
        if visualize:
            map = KeplerGl(height=600, width=600)
            map.add_data(data=df, name='map')
            map.save_to_html(file_name='map.html')
            return df,path,map
        return df,path,None


class new_method():
    #Parameters(in meters)
    def __init__(self,adjacency_list,route_distances,SIGMA_Z = 3.74,BETA = 0.10841,GAMMA = 0.0158):
        self.adjacency_list = adjacency_list
        self.route_distances = route_distances
        # l = []
        # file = open('path', 'rb')

        # # dump information to that file
        # path = pickle.load(file)

        # # close the file
        # file.close()
        # pred = {}
        # for i in range(len(path)):
        #     if i != len(path)-1:
        #         pred[path[i]] = path[i+1]
        #     else:
        #         pred[path[i]] = None

        # for i in path:
        #     if pred[i] != None and pred[i] != 'start' and pred[i] != 'end' and i != 'start' and i != 'end':
        #         l.append(self.haversine_dist(i,self.adjacency_list[i]['z_t'])*1.4826)
        # SIGMA_Z = statistics.median(l)

        # l = []
        # for i in path:
        #     if pred[i] != None and pred[i] != 'start' and pred[i] != 'end' and i != 'start' and i != 'end' :
        #         l.append(np.abs(self.haversine_dist(self.adjacency_list[i]['z_t'],self.adjacency_list[pred[i]]['z_t']) - route_distances[(pred[i],i)])/np.log(2))
        # BETA = statistics.median(l)

        # l_a = []
        # for i in path:
        #     l_a.append(i)

        # l = []
        # for i in range(0,len(l_a)-4):
        # #     if pred[pred[l_a[i]]] != None:
        #     l.append(self.gmma_cost(pred[pred[l_a[i]]],pred[l_a[i]],l_a[i]))

        # GAMMA = statistics.median(l/np.log(2))

        self.SIGMA_Z = SIGMA_Z
        self.BETA = BETA
        self.GAMMA = GAMMA

    def construct_path(self,pred,t):

        route_coords = []
        while t != None:
            prev_coord = pred[t]
            route_coords.append(t[1])
            route_coords.append(t[0])
            t = prev_coord
        return route_coords

    # A gaussian distribution
    def emission_prob(self,u):
        if u == 'start' or u == 'end':
            return 0
        u_z = self.adjacency_list[u]['z_t']

        c = 1 / (self.SIGMA_Z * np.sqrt(2 * np.pi))
        c1 = 0.5*(1/self.SIGMA_Z**2)
        return -c1*(self.haversine_dist(u,u_z)**2) + np.log(c)

    # A empirical distribution
    def transition_prob(self,u,v):
        if u == 'start' or v == 'end':
            return 0

        z_u = self.adjacency_list[u]['z_t']
        z_v = self.adjacency_list[v]['z_t']
        
        c = 1 / self.BETA
        # c = 0

        delta = np.abs(self.route_distances[(u, v)] - self.haversine_dist(z_u,z_v))
        return (-delta*c) + np.log(c)

    def haversine_dist(self,u,v):
        return ox.distance.great_circle_vec(u[0],u[1],v[0],v[1])

    def gmma_cost(self,x,y,z):
        if x == 'start' or z == 'end':
            return 0
        h1 = self.haversine_dist(self.adjacency_list[y]['z_t'],self.adjacency_list[z]['z_t'])
        h2 = self.haversine_dist(self.adjacency_list[x]['z_t'],self.adjacency_list[y]['z_t'])
        r1 = self.route_distances[(y,z)]
        r2 = self.route_distances[(x,y)]
        return np.abs(np.abs(h1-h2)-np.abs(r1-r2))**.5

    def mod_cost(self,x,y,z):
        if x == 'start' or z == 'end':
            return 0
        h1 = self.haversine_dist(self.adjacency_list[y]['z_t'],self.adjacency_list[z]['z_t'])
        h2 = self.haversine_dist(self.adjacency_list[x]['z_t'],self.adjacency_list[y]['z_t'])
        r1 = self.route_distances[(y,z)]
        r2 = self.route_distances[(x,y)]
        
        c = 1 / self.GAMMA
        # c = 0

        # return -c*(np.abs(np.abs(h1-h2)-np.abs(r1-r2))**.5)
        return -c*(np.abs(np.abs(h1-h2)-np.abs(r1-r2)))
    
    def viterbi_search(self,s, t):
        # Initialize joint probability for each node
        print("SIGMA:", self.SIGMA_Z)
        print("BETA:", self.BETA)
        print("GAMMA:", self.GAMMA)

        joint_prob = {}
        lt = []
        init_points = []

        for u in self.route_distances:
            joint_prob[u] = -np.inf

            if u[0] == 'start':
                init_points.append(u)
        predecessor = {}

        q = queue_p.queue()

        for pts in init_points:
            joint_prob[pts] = 1
            print(joint_prob[pts],pts)
            q.push(pts)
            predecessor[pts] = None
        
        i = 0
        while not q.Empty():
            #Book-keeping
            if i % 1000 == 0:
                print(i)
                with open("jp",'wb') as file:
                    pickle.dump(predecessor,file)
            
            # Extract node u
            u = q.pop()
            # append nodes that had a chance on queue
            lt.append(u)
            
            if u[1] == 'end': break
            for v in self.adjacency_list[u[1]]['coords_t+1']:
                if self.adjacency_list[v]['index'] - self.adjacency_list[u[1]]['index'] == 1:
                    for z in self.adjacency_list[v]['coords_t+1']:
                        if self.adjacency_list[z]['index'] - self.adjacency_list[v]['index'] == 1:
                            if z[0] == t[0] and z[1] == t[1]:
                                new_prob = joint_prob[u] + self.transition_prob(u[1],v) + self.transition_prob(v,z) \
                                        + self.emission_prob(v) + self.emission_prob(z) \
                                        + self.mod_cost(u[0],u[1],v) + self.mod_cost(u[1],v,z)
                        
                                if joint_prob[(v,z)] < new_prob:
                                    joint_prob[(v,z)] = new_prob
                                    # predecessor[z] = v
                                    # predecessor[v] = u[1]
                                    # predecessor[u[1]] = u[0]
                                    predecessor[(v,z)] = u
                                if not q.in_q((v,z)):
                                    q.push((v,z))
                            
                            else:
                                #calculate new_prob        
                                new_prob = joint_prob[u] + self.transition_prob(u[1],v) + self.transition_prob(v,z) \
                                            + self.emission_prob(v) + self.emission_prob(z) \
                                            + self.mod_cost(u[0],u[1],v) + self.mod_cost(u[1],v,z)
                            
                                if joint_prob[(v,z)] < new_prob:
                                    joint_prob[(v,z)] = new_prob
                                    # predecessor[v] = u[1]
                                    # predecessor[u[1]] = u[0]
                                    predecessor[(v,z)] = u
                                if not q.in_q((v,z)):
                                    q.push((v,z))
            i+=1

        lt = []
        lt_v = []
        for keys in joint_prob:
                if keys[1] == 'end':
                    lt.append(keys)
                    lt_v.append(joint_prob[keys])
        

        
        return joint_prob,lt[np.argmax(lt_v)],predecessor

    def get_route(self,predecessor,s,t,visualize = True):

        path = self.construct_path(predecessor,t)
        path.remove('start')
        path.remove('end')
        np_df_x,np_df_y,np_df_t = [],[],[]
        for coord in path:
            np_df_x.append(coord[0])
            np_df_y.append(coord[1])
            np_df_t.append(self.adjacency_list[coord]['time'])

        np_df = np.array([np_df_x,np_df_y,np_df_t]).T
        df = pd.DataFrame(np_df, columns = ['Latitude','Longitude','Time'])
        print(df.head())
        if visualize:
            map = KeplerGl(height=600, width=600)
            map.add_data(data=df, name='map')
            map.save_to_html(file_name='map.html')
            return df,path,map
        return df,path,None


class closest_dist():
    def __init__(self,adjacency_list,route_distances,SIGMA_Z = 3.74,BETA = 0.10841,GAMMA = 0.158):
        self.SIGMA_Z = SIGMA_Z
        self.BETA = BETA
        self.GAMMA = GAMMA
        self.adjacency_list = adjacency_list
        self.route_distances = route_distances

    def get_route(self):
        for keys in adj:
            np_df_x.append(adj[keys]['x'][np.argmin(adj[keys]['dist_bet_z_x'])][0])
            np_df_y.append(adj[keys]['x'][np.argmin(adj[keys]['dist_bet_z_x'])][1])
        return


