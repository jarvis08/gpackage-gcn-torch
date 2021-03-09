import networkx as nx
import numpy as np
import json


def load_embedding_from_txt(file_name):
    names = []
    embeddings = []
    with open(file_name, 'r') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            splitted = line.split()
            names.append(splitted[0])
            embeddings.append([float(value) for value in splitted[1:]])
    print(len(names)," words loaded.")
    return names, embeddings


CROSS_VALIDATION = 10
PROTEIN_PROTEIN_FILE = './gdp/IRefindex_protein_protein.txt'
CELL_PROTEIN_FILE = './gdp/IRefindex_cell_protein.txt'

protein_protein_file = open(PROTEIN_PROTEIN_FILE, 'rb')
protein_protein_graph = nx.read_edgelist(protein_protein_file, data=(('weight', float),))

cell_protein_file = open(CELL_PROTEIN_FILE, 'rb')
cell_protein_graph = nx.read_edgelist(cell_protein_file, data=(('weight', float),))

base = nx.compose(protein_protein_graph, cell_protein_graph)

for cv_num in range(1, CROSS_VALIDATION + 1):
    SAVE_PATH = f"./results/FOLD-{cv_num}/"

    print(f"Start making train_graph-{cv_num}.json")
    CELL_DRUG_FILE = f'./gdp/fold/Cell-Drug-{cv_num}.txt'
    cell_drug_file = open(CELL_DRUG_FILE, 'rb')
    cell_drug_edges = nx.read_edgelist(cell_drug_file, data=(('weight', int),))
    
    graph = nx.compose(base, cell_drug_edges)

    data = nx.json_graph.node_link_data(graph)
    nodes = data["nodes"]

    # Make & Save id_map
    # { real_name : id }
    loaded_ids, loaded_embeddings = load_embedding_from_txt(f"./EmbeddingData/ORI/total_embedding-{cv_num}.txt")
    id_dict = dict()
    idx = 0
    for node in nodes:
        if node['id'] in loaded_ids:
            n = node['id']
            id_dict[n] = idx
            idx += 1
    print(">>> Make id_map.json")
    with open(f"{SAVE_PATH}graph_id_map.json", 'w') as f:
        json.dump(id_dict, f)
            
    # check length of nodes
    n_nodes = len(id_dict)
    print(f">>> Number of Nodes = {n_nodes}")

    # replace real id to numericc id
    id_graph = nx.relabel_nodes(graph, id_dict)

    # Save features.npy
    features = []
    idx = 0
    for k, v in id_dict.items():
        idx = loaded_ids.index(k)
        features.append(loaded_embeddings[idx])
    features = np.asarray(features)
    print(">>> Make train_feats.npy")
    print(features.shape)
    print(features[0])
    with open(f"{SAVE_PATH}train_feats.npy", "wb") as f:
        np.save(f, features)

    # Save graph_id.npy
    graph_ids = [1 for i in range(n_nodes)]
    print(">>> Check train_graph_id.npy")
    graph_ids = np.asarray(graph_ids)
    with open(f"{SAVE_PATH}train_graph_id.npy", "wb") as f:
        np.save(f, graph_ids)

    # Save train_labels.npy
    with open(CELL_DRUG_FILE, 'r') as f:
        c_d_edges = f.readlines()
    drugs = dict()
    idx = 0
    for i in range(len(c_d_edges)):
        if i % 2 == 0:
            edge = c_d_edges[i].split("\t")
            drug = edge[1]
            if drug not in drugs.keys():
                drugs[drug] = idx
                idx += 1
    n_drugs = len(drugs)
    with open(f"{SAVE_PATH}unfold_drugs.json", 'w') as f:
        json.dump(drugs, f)

    train_labels = np.zeros(shape=(n_nodes, n_drugs))
    for i in range(len(c_d_edges)):
        if i % 2 == 0:
            edge = c_d_edges[i].split("\t")
            cell = edge[0]
            drug = edge[1]
            cell_id = id_dict[cell]
            label_id = drugs[drug]
            train_labels[cell_id][label_id] = 1
    with open(f"{SAVE_PATH}train_labels.npy", "wb") as f:
        np.save(f, train_labels)
    

    with open(f"{SAVE_PATH}train_graph.json", "w") as f:
        data = nx.json_graph.node_link_data(id_graph)
        json.dump(data, f)

    exit()
