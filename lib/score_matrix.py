import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from networkx.drawing.nx_pydot import graphviz_layout


def score_matrix(G, attr_name):
    edge_data = {
        (u, v): attr[attr_name]
        for u, v, attr in G.edges(data=True)
        if attr_name in attr
    }

    # Creating a DataFrame
    nodes = list(G.nodes())
    df = pd.DataFrame(index=nodes, columns=nodes)  # Fill non-connections with 0

    # Populate the DataFrame with scores
    for (u, v), score in edge_data.items():
        df.at[u, v] = score

    # Visualizing the matrix
    plt.figure(figsize=(10, 8))
    # Create a mask for NaN values
    # Replace None with NaN
    df.replace(to_replace=[None], value=np.nan, inplace=True)

    # Create a mask for NaN values
    mask = df.isnull()
    # Visualizing the matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="rocket", mask=mask)
    plt.title("Node Comparison Matrix")
    plt.show()
