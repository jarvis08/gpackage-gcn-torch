import numpy as np
import random

#graph_ids = [(int(i / 22458)+1) for i in range(44906)]
graph_ids = []
for i in range(44906):
    graph_ids.append(random.randint(1, 20))
print(">>> Check train_graph_id.npy")
graph_ids = np.asarray(graph_ids)
with open(f"train_graph_id.npy", "wb") as f:
    np.save(f, graph_ids)

