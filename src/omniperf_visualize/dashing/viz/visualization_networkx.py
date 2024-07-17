import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import sys

def normalize_dict(value_dict):
    print('called')
    x = []
    for key,value in value_dict.items():
        if '$Y' in key:
            continue
        else:
            x.append(float(value))
    max_val = max(x)
    min_val = min(x)
    if max_val - min_val == 0:
        return value_dict
    for key,value in value_dict.items():
        if '$Y' in key:
            continue
        else:
            value_dict[key] = (float(value) - min_val) / (max_val - min_val)
    # print(max_val,min_val)
    # print(value_dict[1])
    return value_dict

# arg=sys.argv
# del arg[0]

def create_graph(data_loader):
    threshold=.1
    file_to_open='output/y_new_15.txt'
    f = open(file_to_open, "r")
    

    content_dict = {}
    num_latent=5
    content=f.readlines()
    dict_list=[]
    counter=1
    num_counters=0

    for i in content:

        if i[0]=='$' and num_counters!=0:
            break
        num_counters += 1
    for i in content:
        #print(i)
        key_val_list=[]
        if ',' in i:
            key_val_list=i.split(',')
        elif i[0]=='$':
            key_val_list=i.split(':')
        for j in range(len(key_val_list)):
            key_val_list[j]=key_val_list[j].strip()
        #print(key_val_list)
        content_dict[key_val_list[0]]=key_val_list[1]
        if counter%num_counters==0: #don't hardcode, parse through and find lines with y
            # print(type(content_dict))
            content_dict = normalize_dict(content_dict)
            dict_list.append(content_dict)
            content_dict = {}
        counter+=1
    #print(dict_list)
    node_sizes=[]
    y_dict={}

    for i in range(num_latent):
        y_dict['$Y'+str(i)]=dict_list[i]['$Y'+str(i)]
        del dict_list[i]['$Y'+str(i)]
    print(y_dict)
    # print(dict_list)
    graph_list=[]
    G = nx.Graph()
    for i in range(num_latent):

        data=np.array(list(dict_list[i].values())).reshape(-1, 1)
        # print("Zayed test", data)
        scaler = MinMaxScaler(feature_range=(0, 1))
        # data=list(scaler.fit_transform(data))
        # print("Zayed test", data)
        index=0
        for key, val in dict_list[i].items():
            dict_list[i][key]=round(float(data[index][0]), 3)
            index+=1
        for key, val in dict_list[i].items():
            # print("Zayed test", key, val)
            if val >=float(threshold): #take threshold as input
                G.add_edge('$Y'+str(i), str(key), weight=val)
                # print('Zayed coming here')
        #graph_list.append(G)
    #ALL LAYOUT TESTING

    # pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility, adjust layout
    # pos = nx.bipartite_layout(G, pos.keys())
    # pos = nx.circular_layout(G)
    # pos = nx.kamada_kawai_layout(G)
    # pos = nx.planar_layout(G)
    # pos = nx.random_layout(G)
    # pos = nx.rescale_layout_dict(pos)
    # pos = nx.shell_layout(G)
    # pos = nx.spectral_layout(G)
    pos = nx.spiral_layout(G)

    """values=[]
    for key, val in pos.items():
        if key[0]=='$':
            values.append(float(y_dict[key]))
    values=np.array(values).reshape(-1, 1)
    values=list(scaler.fit_transform(values))
    counter=0
    for key, val in pos.items():
        if key[0]=='$':
            y_dict[key]=values[counter]

            counter+=1
    """
    node_colors=[]
    print(pos)
    for key, val in pos.items():
        print(key, val)
        if key[0] == '$':
            node_sizes.append(float(y_dict[key])*10)
            #node_colors.append(rgb_to_hex(0,0,255*float(y_dict[key])))
            #print(float(y_dict[key]))
            #node_colors.append((0,0,float(y_dict[key])))
            # print('test Zayed', y_dict[key])
            node_colors.append(round(float(y_dict[key])*100))
        else:
            node_sizes.append(100.0)
            node_colors.append(0)

    #elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.6]
    #esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]


    node_colors=np.array(node_colors).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 100))
    print(node_colors)
    # node_colors=list(scaler.fit_transform(node_colors))
    for i in range(len(node_colors)):
        node_colors[i]=round(node_colors[i][0])
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.Blues) #y values have bubble, rest don't have bubble
    print(node_colors)

    # edges
    #print(G.edges)
    edge_sizes=[]
    for i in G.edges:
        if '$' in i[0]:
            index=int(i[0][2])
            edge_sizes.append(dict_list[index][i[1]])
        else:
            index = int(i[1][2])
            edge_sizes.append(dict_list[index][i[0]])


    nx.draw_networkx_edges(G, pos, edgelist=G.edges, width=edge_sizes, style='solid')

    # node labels
    nx.draw_networkx_labels(G, pos, font_size=7.5, font_family="sans-serif", font_color='black')
    # edge weight labels
    #edge_labels = nx.get_edge_attributes(i, "weight")
    #nx.draw_networkx_edge_labels(i, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.01)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    print('Zayed ', type(nx), type(plt))
    data_loader.options['charts'].append(G)
    #print(dict_list)

# if __name__=='__main__':
#     create_graph(threshold, file_to_open)


