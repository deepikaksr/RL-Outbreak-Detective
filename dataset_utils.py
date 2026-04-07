import os
import networkx as nx
import requests
import gzip
import random

def download_file(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading {url} to {filepath}...")
        r = requests.get(url, stream=True)
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

def load_snap_livejournal(subgraph_size=None):
    """
    Loads the SNAP com-LiveJournal network.
    If subgraph_size is specified, it extracts a connected subgraph of that size using BFS.
    """
    url = "https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz"
    gz_path = "com-lj.ungraph.txt.gz"
    
    download_file(url, gz_path)
    print("Loading LiveJournal Graph...")
    
    G = nx.Graph()
    with gzip.open(gz_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            u, v = map(int, line.strip().split())
            G.add_edge(u, v)
            if subgraph_size and G.number_of_nodes() >= subgraph_size * 5:
                break
                
    if subgraph_size is not None:
        start_node = random.choice(list(G.nodes()))
        nodes_to_keep = set([start_node])
        queue = [start_node]
        
        while queue and len(nodes_to_keep) < subgraph_size:
            current = queue.pop(0)
            for neighbor in G.neighbors(current):
                if neighbor not in nodes_to_keep:
                    nodes_to_keep.add(neighbor)
                    queue.append(neighbor)
                    if len(nodes_to_keep) >= subgraph_size:
                        break
        
        G = G.subgraph(list(nodes_to_keep)).copy()
        
    # Relabel nodes sequentially 0...N-1 for BOTH full graph and subgraphs!
    G = nx.convert_node_labels_to_integers(G)
        
    print("\n" + "="*50)
    print("📈  DATASET STATISTICS  📈")
    print("="*50)
    print(f"Network Name:        SNAP com-LiveJournal")
    print(f"Total Nodes:         {G.number_of_nodes():,}")
    print(f"Total Edges:         {G.number_of_edges():,}")
    print(f"Is Directed:         {G.is_directed()}")
    print("="*50 + "\n")
    
    return G

if __name__ == "__main__":
    # Test loading the full graph
    g = load_snap_livejournal(subgraph_size=None)
    print("Test passed.")
