import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_graphml("Results/Graphs/g.graphml")

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=300,
    font_size=8,
    edge_color="gray"
)

plt.savefig("graph.png", dpi=300, bbox_inches="tight")
plt.show()
