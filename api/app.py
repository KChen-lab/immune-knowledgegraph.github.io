from flask import Flask, request, jsonify
from flask_cors import CORS
import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter
import random
import requests
from io import StringIO

app = Flask(__name__)
CORS(app)

# GitHub raw content URL for your CSV file
CSV_URL = "https://raw.githubusercontent.com/chloehe1129/immune-knowledgegraph.github.io/main/data/network_data.csv"

def build_weighted_network(csv_url):
    """
    Build a weighted directed network from a CSV file hosted on GitHub
    """
    # Fetch CSV data from GitHub
    response = requests.get(csv_url)
    if response.status_code != 200:
        raise Exception("Failed to fetch data from GitHub")
    
    # Read CSV data from the response content
    csv_data = StringIO(response.text)
    df = pd.read_csv(csv_data)
    
    G = nx.DiGraph()
    
    # Add edges to the network
    for _, row in df.iterrows():
        G.add_edge(
            row['START_ID'],
            row['END_ID'],
            relationship=row['relationship'],
            weight=row['weight:int'],
            file_count=row['file_count:int'],
            sources=eval(row['sources'])
        )
    
    return G

weighted_network = None

@app.before_first_request
def initialize_network():
    global weighted_network
    weighted_network = build_weighted_network(CSV_URL)

def create_activate_only_graph(G):
    activate_G = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if data.get('relationship') == 'activate':
            activate_G.add_edge(u, v, weight=data.get('weight', 1))
    return activate_G

def pagerank_permutation_test(G, gene_set, n_permutations=1000, alpha=0.85):
    activate_G = create_activate_only_graph(G)
    valid_gene_set = [gene for gene in gene_set if gene in activate_G]
    
    if not valid_gene_set:
        return []

    real_personalization = {node: 1 if node in valid_gene_set else 0 for node in activate_G.nodes()}
    real_pagerank = nx.pagerank(activate_G, alpha=alpha, personalization=real_personalization, weight='weight')
    
    # Consider nodes that don't start with "CD" as pathway nodes
    all_nodes = list(G.nodes())
    pathway_nodes = [node for node in all_nodes if not node.startswith('CD')]
    
    permutation_results = []
    for _ in range(n_permutations):
        random_gene_set = random.sample(list(activate_G.nodes()), min(len(valid_gene_set), len(activate_G)))
        perm_personalization = {node: 1 if node in random_gene_set else 0 for node in activate_G.nodes()}
        perm_pagerank = nx.pagerank(activate_G, alpha=alpha, personalization=perm_personalization, weight='weight')
        permutation_results.append([perm_pagerank.get(node, 0) for node in pathway_nodes])
    
    permutation_results = np.array(permutation_results)
    real_results = np.array([real_pagerank.get(node, 0) for node in pathway_nodes])
    
    permutation_results = permutation_results.T
    real_results = real_results.reshape(-1, 1)
    
    p_values = ((permutation_results >= real_results).sum(axis=1) + 1) / (n_permutations + 1)
    ci_lower = np.percentile(permutation_results, 2.5, axis=1)
    ci_upper = np.percentile(permutation_results, 97.5, axis=1)
    
    return list(zip(pathway_nodes, real_results.flatten(), p_values, ci_lower, ci_upper))

@app.route('/analyze', methods=['POST'])
def analyze_genes():
    data = request.json
    gene_set = data.get('genes', [])
    
    if not gene_set:
        return jsonify({'error': 'No genes provided'}), 400
    
    try:
        results = pagerank_permutation_test(weighted_network, gene_set)
        significant_results = [result for result in results if result[2] < 0.05]
        sorted_results = sorted(significant_results, key=lambda x: x[1], reverse=True)
        
        formatted_results = [{
            'pathway': result[0],
            'score': float(result[1]),
            'p_value': float(result[2]),
            'ci_lower': float(result[3]),
            'ci_upper': float(result[4])
        } for result in sorted_results]
        
        return jsonify({'results': formatted_results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)