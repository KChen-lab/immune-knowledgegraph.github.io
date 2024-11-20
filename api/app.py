import io
import ast
import os
import random
from collections import Counter

# Third party imports
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from flask import Flask, request, jsonify
from flask_cors import CORS

import plotly.graph_objects as go
import json
import traceback  # Add this import
import sys


app = Flask(__name__)

CORS(app, resources={
    r"/*": {  # This allows all routes
        "origins": [
            # "https://chloehe1129.github.io",
            "https://kchen-lab.github.io",
            "http://localhost:5000",
            "http://127.0.0.1:5000",
            "http://localhost:8000",
            "http://127.0.0.1:8000"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"]
    }
})


def reconstruct_graph_from_csv(nodes_path, relationships_path):
    # Create a new directed graph
    G = nx.DiGraph()

    # Read nodes
    nodes_df = pd.read_csv(nodes_path)
    for _, row in nodes_df.iterrows():
        G.add_node(row['nodeId:ID'],
                   name=row['name'],
                   node_type=row['node_type'])
    # Read relationships
    rels_df = pd.read_csv(relationships_path)
    for _, row in rels_df.iterrows():
        # Convert string representation of sources back to list
        sources = ast.literal_eval(row['sources'])

        G.add_edge(row[':START_ID'],
                   row[':END_ID'],
                   relationship=row['relationship'],
                   weight=row['weight:int'],
                   file_count=row['file_count:int'],
                   sources=sources)

    return G

# Initialize network at startup
#def get_files_for_cell_type(cell_type):
#    prefix = f"{cell_type}_"  # Will create "T_", "NK_", or "B_"
#    return (
#        f'{prefix}nodes_pmid_gene_normalized.csv',
#        f'{prefix}relationships_pmid_gene_normalized.csv'
#    )

def get_files_for_cell_type(cell_type):
    # Get the absolute path to the directory containing your app.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prefix = f"{cell_type}_"  # Will create "T_", "NK_", or "B_"

    return (
        os.path.join(base_dir, f'{prefix}nodes_pmid_gene_normalized.csv'),
        os.path.join(base_dir, f'{prefix}relationships_pmid_gene_normalized.csv')
    )

# Initialize network with default cell type (NK)
default_cell_type = 'NK'
nodes_file, rels_file = get_files_for_cell_type(default_cell_type)
weighted_network = reconstruct_graph_from_csv(nodes_file, rels_file)

def weighted_choice(choices):
    total = sum(w for c, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices:
        if upto + w >= r:
            return c
        upto += w
    assert False, "Shouldn't get here"

def adjusted_random_walk(G, start_nodes, n_walks=100, walk_length=10, jump_prob=0.1):
    visited = Counter()
    for start_node in start_nodes:
        if start_node not in G:
            print(f"Warning: {start_node} not found in the graph.")
            continue
        for _ in range(n_walks):
            current_node = start_node
            for _ in range(walk_length):
                visited[current_node] += 1
                if random.random() < jump_prob:
                    current_node = random.choice(list(G.nodes()))
                else:
                    # Get only the "activate" edges with their weights
                    activate_successors = [
                        (node, G[current_node][node].get('weight', 1))
                        for node in G.successors(current_node)
                        if G[current_node][node].get('relationship') == 'activate'
                    ]
                    activate_predecessors = [
                        (node, G[node][current_node].get('weight', 1))
                        for node in G.predecessors(current_node)
                        if G[node][current_node].get('relationship') == 'activate'
                    ]
                    neighbors = activate_successors + activate_predecessors
                    if not neighbors:
                        break
                    current_node = weighted_choice(neighbors)
    return visited

def permutation_test(G, gene_set, n_permutations=1000, n_walks=100, walk_length=10, jump_prob=0.1):
    real_visits = adjusted_random_walk(G, gene_set, n_walks, walk_length, jump_prob)
    pathway_nodes = [node for node in G.nodes() if G.nodes[node].get('node_type') in ['pathways']]

    permutation_results = []
    for _ in range(n_permutations):
        random_start = random.sample(list(G.nodes()), len(gene_set))
        perm_visits = adjusted_random_walk(G, random_start, n_walks, walk_length, jump_prob)
        permutation_results.append([perm_visits[node] for node in pathway_nodes])

    permutation_results = np.array(permutation_results)
    real_results = np.array([real_visits[node] for node in pathway_nodes])

    real_results = real_results.reshape(1, -1)
    p_values = ((permutation_results >= real_results).sum(axis=0) + 1) / (n_permutations + 1)

    # Calculate confidence intervals (95% CI)
    ci_lower = np.percentile(permutation_results, 2.5, axis=0)
    ci_upper = np.percentile(permutation_results, 97.5, axis=0)

    return list(zip(pathway_nodes, real_results[0], p_values, ci_lower, ci_upper))

def find_top_pathway_nodes(G, gene_set, n_permutations=1000, n_walks=100, walk_length=10, jump_prob=0.1):
    results = permutation_test(G, gene_set, n_permutations, n_walks, walk_length, jump_prob)
    significant_results = [result for result in results if result[2] < 0.05]
    return sorted(significant_results, key=lambda x: x[1], reverse=True)

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
        print("Warning: None of the genes in gene_set are present in the activate-only graph.")
        return []

    real_personalization = {node: 1 if node in valid_gene_set else 0 for node in activate_G.nodes()}
    real_pagerank = nx.pagerank(activate_G, alpha=alpha, personalization=real_personalization, weight='weight')

    pathway_nodes = [node for node in G.nodes() if G.nodes[node].get('node_type') in ['pathways']]

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

def find_top_pathways_pagerank(G, gene_set, n_permutations=1000, alpha=0.85):
    results = pagerank_permutation_test(G, gene_set, n_permutations, alpha)
    if not results:
        print("No valid results from PageRank analysis.")
        return []
    significant_results = [result for result in results if result[2] < 0.05]
    return sorted(significant_results, key=lambda x: x[1], reverse=True)

def create_activation_barplot(gene_contributions, adjusted_scores, perturbation_node):
    try:
        # Separate activated and inhibited genes
        activated_genes = []
        inhibited_genes = []

        for gene, score in gene_contributions:
            effect = adjusted_scores[gene][1]
            if effect == 'activate':
                activated_genes.append((gene, score))
            elif effect == 'inhibit':
                inhibited_genes.append((gene, score))

        # Take top 10 each
        top_activated = sorted(activated_genes, key=lambda x: x[1], reverse=True)[:10]
        top_inhibited = sorted(inhibited_genes, key=lambda x: x[1], reverse=True)[:10]

        # Create plot
        plt.figure(figsize=(10, 8))
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot data
        activated_genes_names = [gene for gene, _ in top_activated]
        activated_scores = [-np.log2(score) for _, score in top_activated]
        inhibited_genes_names = [gene for gene, _ in top_inhibited]
        inhibited_scores = [np.log2(score) for _, score in top_inhibited]

        total_genes = len(activated_scores) + len(inhibited_scores)
        y_positions = np.arange(total_genes)

        ax.barh(y_positions[:len(activated_scores)],
                activated_scores,
                align='center',
                color='#2ecc71',
                alpha=0.8,
                label='Activated')
        ax.barh(y_positions[len(activated_scores):],
                inhibited_scores,
                align='center',
                color='#e74c3c',
                alpha=0.8,
                label='Inhibited')

        ax.set_yticks(y_positions)
        labels = activated_genes_names + inhibited_genes_names
        ax.set_yticklabels(labels)
        ax.set_xlabel('Log2(Normalized Score)')
        ax.set_title(f'Top 10 Activated and Inhibited Genes\nPerturbation: {perturbation_node}')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.legend()

        plt.tight_layout()

        # Convert to SVG
        buf = io.BytesIO()
        plt.savefig(buf, format='svg', bbox_inches='tight')
        plt.close('all')
        buf.seek(0)
        svg_string = buf.getvalue().decode('utf-8')
        buf.close()

        return svg_string

    except Exception as e:
        print(f"Error in create_activation_barplot: {str(e)}")
        return None


def pagerank_gene_analysis_tab3(G, perturbation_node, alpha=0.85, max_iter=100, tol=1e-06):
    try:
        # Initialize personalization
        personalization = {perturbation_node: 1}
        pagerank_scores = nx.pagerank(G, alpha=alpha, personalization=personalization,
                                    max_iter=max_iter, tol=tol, weight='weight')

        # Process scores and effects
        adjusted_scores = {}
        for node, score in pagerank_scores.items():
            if G.nodes[node].get('node_type') == 'gene':
                try:
                    path = nx.shortest_path(G, perturbation_node, node)
                    path_relationships = []
                    for i in range(len(path)-1):
                        source, target = path[i], path[i+1]
                        relationship = G[source][target].get('relationship')
                        path_relationships.append(relationship)
                    inhibit_count = path_relationships.count('inhibit')
                    net_effect = 'inhibit' if inhibit_count % 2 != 0 else 'activate'
                    adjusted_scores[node] = (score, net_effect)
                except nx.NetworkXNoPath:
                    continue

        # Calculate total score for normalization
        total_gene_score = sum(score for score, _ in adjusted_scores.values())

        # Format results
        results = []
        for gene, (score, effect) in adjusted_scores.items():
            if gene != perturbation_node:  # Exclude self
                normalized_score = (score / total_gene_score) * 100
                results.append({
                    'gene': gene,
                    'score': float(normalized_score),  # Ensure it's a float
                    'effect': effect
                })

        # Sort by absolute score value
        results = sorted(results, key=lambda x: abs(x['score']), reverse=True)

        # Create visualization
        gene_contributions = [(r['gene'], r['score']) for r in results[:20]]  # Top 20 genes
        plot = create_activation_barplot(gene_contributions,
                                      dict((r['gene'], (r['score'], r['effect'])) for r in results),
                                      perturbation_node)

        return results, plot

    except Exception as e:
        print(f"Error in pagerank_gene_analysis: {str(e)}")
        return [], None

def pagerank_gene_analysis(G, perturbation_node):
    """Run PageRank analysis and return adjusted scores."""
    try:
        print(f"Starting PageRank analysis for {perturbation_node}")
        personalization = {perturbation_node: 1}
        pagerank_scores = nx.pagerank(G, alpha=0.85, personalization=personalization)

        # Initialize adjusted scores
        adjusted_scores = {}

        for node, score in pagerank_scores.items():
            if G.nodes[node].get('node_type') == 'gene':
                try:
                    path = nx.shortest_path(G, perturbation_node, node)
                    path_relationships = []
                    for i in range(len(path)-1):
                        source, target = path[i], path[i+1]
                        relationship = G[source][target].get('relationship')
                        path_relationships.append(relationship)
                    inhibit_count = path_relationships.count('inhibit')
                    net_effect = 'inhibit' if inhibit_count % 2 != 0 else 'activate'
                    adjusted_scores[node] = (score, net_effect)
                except nx.NetworkXNoPath:
                    continue

        print(f"PageRank analysis complete. Found {len(adjusted_scores)} scores")
        return adjusted_scores

    except Exception as e:
        print(f"Error in pagerank_gene_analysis: {str(e)}")
        traceback.print_exc()
        return {}  # Return empty dict instead of None


def pagerank_gene_analysis_tab2(G, phenotype_node, alpha=0.85, max_iter=100, tol=1e-06):
    # Run PageRank on the graph
    personalization = {phenotype_node: 1}
    pagerank_scores = nx.pagerank(G, alpha=alpha, personalization=personalization,
                                  max_iter=max_iter, tol=tol, weight='weight')
    # Post-process scores to account for activation/inhibition
    adjusted_scores = {}
    for node, score in pagerank_scores.items():
        if G.nodes[node].get('node_type') == 'gene':
            # Count inhibit relationships
            inhibit_count = sum(1 for _, target, data in G.out_edges(node, data=True)
                                if target == phenotype_node and data.get('relationship') == 'inhibit')
            # Determine net effect
            net_effect = 'inhibit' if inhibit_count % 2 != 0 else 'activate'
            # Store both the score and the net effect
            adjusted_scores[node] = (score, net_effect)
    # Calculate total PageRank score of genes
    total_gene_score = sum(score for score, _ in adjusted_scores.values())
    # Convert scores to percentages of total gene score and filter out zeros
    gene_percentages = {gene: (score / total_gene_score) * 100
                        for gene, (score, _) in adjusted_scores.items()
                        if (score / total_gene_score) * 100 > 0}
    # Sort genes by percentage
    sorted_genes = sorted(gene_percentages.items(), key=lambda item: item[1], reverse=True)
    return sorted_genes, adjusted_scores

@app.route('/analyze', methods=['GET', 'POST'])
def analyze_genes():
    if request.method == 'GET':
        return jsonify({
            'status': 'ok',
            'message': 'API is running. Please use POST method to analyze genes.'
        })

    data = request.json
    gene_set = data.get('genes', [])
    cell_type = data.get('cellType', 'NK')

    if not gene_set:
        return jsonify({'error': 'No genes provided'}), 400

    try:
        nodes_file, rels_file = get_files_for_cell_type(cell_type)
        weighted_network = reconstruct_graph_from_csv(nodes_file, rels_file)
        results = find_top_pathways_pagerank(weighted_network, gene_set)

        formatted_results = [{
            'pathway': result[0],
            'score': float(result[1]),
            'p_value': float(result[2]),
            'ci_lower': float(result[3]),
            'ci_upper': float(result[4])
        } for result in results]

        return jsonify({'results': formatted_results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

from fuzzywuzzy import process

@app.route('/search-nodes', methods=['GET'])
def search_nodes():
    query = request.args.get('query', '')
    if not query:
        return jsonify([])

    # Get nodes of specified types
    valid_nodes = [node for node in weighted_network.nodes()
                  if weighted_network.nodes[node].get('node_type') in ['pathways', 'others', 'cell type']]

    # Fuzzy match
    matches = process.extractBests(query, valid_nodes, limit=10, score_cutoff=60)
    results = [{'value': match[0], 'score': match[1]} for match in matches]

    return jsonify(results)

@app.route('/generate-geneset', methods=['POST'])
def generate_geneset():
    try:
        data = request.json
        phenotype = data.get('phenotype', '')
        cell_type = data.get('cellType', 'NK')

        print(f"Generating gene set for phenotype: {phenotype}, cell type: {cell_type}")

        if not phenotype:
            return jsonify({'error': 'No phenotype provided'}), 400

        nodes_file, rels_file = get_files_for_cell_type(cell_type)
        network = reconstruct_graph_from_csv(nodes_file, rels_file)

        if phenotype not in network:
            return jsonify({'error': f'Phenotype "{phenotype}" not found in the network'}), 404

        # Run PageRank analysis using your original function
        gene_contributions, adjusted_scores = pagerank_gene_analysis_tab2(network, phenotype)

        # Take top 20 genes
        top_genes = gene_contributions[:10]

        # Format results with paths
        results = []
        paths_data = []

        for gene, score in top_genes:
            effect = adjusted_scores[gene][1]

            # Find shortest path
            path = find_gene_path(network, gene, phenotype)

            if path:
                # Collect edges and their PMIDs
                path_edges = []
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1]
                    relationship = network[u][v].get('relationship')
                    sources = network[u][v].get('sources', [])
                    pmids = [s.replace('[', '').replace(']', '').replace('.t', '')
                            for s in sources]
                    path_edges.append({
                        'source': u,
                        'target': v,
                        'relationship': relationship,
                        'pmids': pmids
                    })

                paths_data.append({
                    'nodes': path,
                    'edges': path_edges
                })

            results.append({
                'gene': gene,
                'score': float(score),
                'effect': effect
            })

        if not results:
            return jsonify({'error': 'No significant genes found for this phenotype'}), 404

        # Create visualization for all paths
        fig = visualize_multiple_paths(network, paths_data, phenotype)

        return jsonify({
            'results': results,
            'paths': paths_data,
            'plot': fig.to_json() if fig else None
        })

    except Exception as e:
        print(f"Error in generate_geneset: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def find_gene_path(G, source, target):
    """Find shortest path that only goes through gene nodes."""
    try:
        path = nx.shortest_path(G, source, target)
        return path
    except nx.NetworkXNoPath:
        return None

def visualize_multiple_paths(G, paths_data, phenotype_node):
    """Visualize multiple paths in a single plot with hierarchical layout."""
    try:
        # Collect all unique nodes and edges
        all_nodes = set([phenotype_node])
        activation_edges = []
        inhibition_edges = []
        target_genes = set()  # Genes from gene set (bottom row)
        intermediate_genes = set()  # Genes in middle row

        for path_info in paths_data:
            all_nodes.update(path_info['nodes'])
            target_genes.add(path_info['nodes'][0])  # First node is the target gene
            # Add intermediate nodes
            intermediates = path_info['nodes'][1:-1]  # Exclude first and last nodes
            intermediate_genes.update(intermediates)

            for edge in path_info['edges']:
                if edge['relationship'] == 'activate':
                    activation_edges.append((edge['source'], edge['target'], edge['pmids']))
                else:
                    inhibition_edges.append((edge['source'], edge['target'], edge['pmids']))

        # Create node positions
        pos = {}

        # Position phenotype at the top center
        pos[phenotype_node] = [0, 1]

        # Position target genes in bottom row
        sorted_targets = sorted(list(target_genes))
        num_targets = len(sorted_targets)
        for i, gene in enumerate(sorted_targets):
            x = (i - (num_targets - 1)/2) * (1.5/num_targets)  # Spread evenly
            pos[gene] = [x, -1]  # Bottom row

        # Position intermediate nodes in middle row
        sorted_intermediates = sorted(list(intermediate_genes))
        num_intermediates = len(sorted_intermediates)
        for i, gene in enumerate(sorted_intermediates):
            x = (i - (num_intermediates - 1)/2) * (1.5/max(num_intermediates, 1))
            pos[gene] = [x, 0]  # Middle row

        # Create node trace
        node_trace = go.Scatter(
            x=[],
            y=[],
            mode='markers+text',
            hoverinfo='text',
            text=[],
            textposition="bottom center",
            marker=dict(
                size=[],
                color=[],
                line_width=2
            )
        )

        # Add nodes
        for node in all_nodes:
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([node])

            if node == phenotype_node:
                node_trace['marker']['size'] += tuple([30])
                node_trace['marker']['color'] += tuple(['red'])
            elif node in target_genes:
                node_trace['marker']['size'] += tuple([25])
                node_trace['marker']['color'] += tuple(['lightpink'])
            else:  # intermediate nodes
                node_trace['marker']['size'] += tuple([20])
                node_trace['marker']['color'] += tuple(['lightgray'])

        def create_edge_trace(edge_list, color, dash=None, name=""):
            edge_x = []
            edge_y = []
            edge_text = []

            # Keep track of edges between same points to adjust curves
            edge_counts = {}
            for source, target, _ in edge_list:
                key = tuple(sorted([source, target]))
                edge_counts[key] = edge_counts.get(key, 0) + 1

            current_edges = {}
            for source, target, pmids in edge_list:
                x0, y0 = pos[source]
                x1, y1 = pos[target]

                # Get count of edges between these nodes
                key = tuple(sorted([source, target]))
                current_edges[key] = current_edges.get(key, 0) + 1
                current_edge_num = current_edges[key]
                total_edges = edge_counts[key]

                # Calculate curve parameters
                num_points = 100  # More points for smoother curves
                t = np.linspace(0, 1, num_points)

                # Adjust curvature based on node positions and edge count
                dx = x1 - x0
                dy = y1 - y0
                dist = np.sqrt(dx**2 + dy**2)

                # Calculate control point offset
                if total_edges > 1:
                    # Spread multiple edges
                    offset_scale = 0.2  # Base curve amount
                    offset = offset_scale * (2 * current_edge_num - total_edges - 1) / total_edges
                else:
                    # Single edge gets slight curve
                    offset = 0.1

                # Perpendicular vector for control point
                nx = -dy / dist
                ny = dx / dist

                # Control points for quadratic Bezier curve
                if y1 > y0:  # Going upward
                    cp1x = x0 + dx/4 + nx * offset
                    cp1y = y0 + dy/4 + ny * offset
                    cp2x = x0 + 3*dx/4 + nx * offset
                    cp2y = y0 + 3*dy/4 + ny * offset
                else:  # Going downward
                    cp1x = x0 + dx/4 - nx * offset
                    cp1y = y0 + dy/4 - ny * offset
                    cp2x = x0 + 3*dx/4 - nx * offset
                    cp2y = y0 + 3*dy/4 - ny * offset

                # Cubic Bezier curve
                x = (1-t)**3 * x0 + 3*(1-t)**2*t * cp1x + 3*(1-t)*t**2 * cp2x + t**3 * x1
                y = (1-t)**3 * y0 + 3*(1-t)**2*t * cp1y + 3*(1-t)*t**2 * cp2y + t**3 * y1

                edge_x.extend(x)
                edge_y.extend(y)

                hover_text = f"From: {source}<br>To: {target}<br>PMIDs:<br>" + "<br>".join(pmids)
                edge_text.extend([hover_text] * num_points)

                # Add None for gap between edges
                edge_x.append(None)
                edge_y.append(None)
                edge_text.append(None)

            line_style = dict(color=color, width=2)
            if dash:
                line_style['dash'] = dash

            return go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=line_style,
                hoverinfo='text',
                text=edge_text,
                name=name,
                showlegend=True,
                hoveron='points'
            )

        # Create edge traces
        activation_trace = create_edge_trace(activation_edges, 'green', name='Activation')
        inhibition_trace = create_edge_trace(inhibition_edges, 'red', dash='dash', name='Inhibition')

        # Create figure
        fig = go.Figure(data=[activation_trace, inhibition_trace, node_trace])

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Gene Paths to {phenotype_node}',
                x=0.5,
                y=0.95
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)",
                title=dict(text="Legend"),
                itemsizing="constant",
                traceorder="normal"
            ),
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=1200,
            height=800
        )

        # Add legend items for node types
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color='lightpink'),
            name='Genes in the gene set'
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='Phenotype Node'
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color='lightgray'),
            name='Other Genes'
        ))

        return fig

    except Exception as e:
        print(f"Error in visualize_multiple_paths: {str(e)}")
        traceback.print_exc()
        return None

@app.route('/predict-perturbation', methods=['POST'])
def predict_perturbation():
    try:
        data = request.json
        perturbation_node = data.get('gene', '')
        cell_type = data.get('cellType', 'NK')

        if not perturbation_node:
            return jsonify({'error': 'No perturbation gene provided'}), 400

        # Get the network for the selected cell type
        nodes_file, rels_file = get_files_for_cell_type(cell_type)
        network = reconstruct_graph_from_csv(nodes_file, rels_file)

        if perturbation_node not in network:
            return jsonify({'error': f'Gene {perturbation_node} not found in the network'}), 404

        # Run analysis
        results, plot = pagerank_gene_analysis_tab3(network, perturbation_node)

        print(f"Analysis complete - Found {len(results)} results")

        if not results:
            return jsonify({'error': 'No significant results found for this gene'}), 404

        return jsonify({
            'results': results,
            'plot': plot
        })

    except Exception as e:
        print(f"Error in predict_perturbation: {str(e)}")
        return jsonify({'error': str(e)}), 500


def find_gene_path(G, source, target):
    """
    Find shortest path that only goes through gene nodes.
    """
    def gene_path_cost(u, v, edge):
        # Return high cost for non-gene nodes (except source and target)
        if (u != source and u != target and
            G.nodes[u].get('node_type') != 'gene'):
            return float('inf')
        return 1

    try:
        # Use shortest path with custom weight function to avoid non-gene nodes
        path = nx.shortest_path(G, source, target, weight=gene_path_cost)
        return path
    except nx.NetworkXNoPath:
        return None

def visualize_shortest_paths(G, perturbation_node, genes_of_interest):
    """Creates a visualization showing paths from perturbation node to genes of interest."""
    try:
        print(f"Starting visualization for {perturbation_node} to {len(genes_of_interest)} genes")

        # Get paths and effects
        adjusted_scores = pagerank_gene_analysis(G, perturbation_node)
        if not adjusted_scores:
            raise ValueError("No scores generated from PageRank analysis")

        # Initialize node collections
        all_nodes = set([perturbation_node])
        activation_edges = []
        inhibition_edges = []
        intermediates = set()

        # Find paths and collect nodes
        for target in genes_of_interest:
            path = find_gene_path(G, perturbation_node, target)
            if path:
                all_nodes.update(path)
                intermediates.update([node for node in path[1:-1]
                                   if node not in genes_of_interest])
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1]
                    if G[u][v].get('relationship') == 'activate':
                        activation_edges.append((u, v))
                    else:
                        inhibition_edges.append((u, v))

        print(f"Found {len(all_nodes)} nodes and {len(activation_edges) + len(inhibition_edges)} edges")

        if not (activation_edges or inhibition_edges):
            raise ValueError("No valid paths found between perturbation gene and target genes")

        # Create node trace
        node_trace = go.Scatter(
            x=[],
            y=[],
            mode='markers+text',
            hoverinfo='text',
            text=[],
            textposition="bottom center",
            marker=dict(
                size=[],
                color=[],
                line_width=2
            )
        )

        # Position nodes
        pos = {
            perturbation_node: [0, 1]  # Perturbation node at top
        }

        # Position intermediate nodes in middle
        for i, node in enumerate(intermediates):
            pos[node] = [i / (len(intermediates) + 1) - 0.5, 0.5]

        # Position target genes at bottom
        for i, node in enumerate(genes_of_interest):
            pos[node] = [i / (len(genes_of_interest) + 1) - 0.5, 0]

        # Add nodes to trace
        for node in all_nodes:
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([node])

            if node == perturbation_node:
                node_trace['marker']['size'] += tuple([20])
                node_trace['marker']['color'] += tuple(['red'])
            elif node in genes_of_interest:
                effect = adjusted_scores.get(node, (0, 'unknown'))[1]
                node_trace['marker']['size'] += tuple([15])
                if effect == 'activate':
                    node_trace['marker']['color'] += tuple(['lightgreen'])
                else:
                    node_trace['marker']['color'] += tuple(['salmon'])
            else:
                node_trace['marker']['size'] += tuple([10])
                node_trace['marker']['color'] += tuple(['lightgray'])

        def create_edge_trace(edge_list, color, dash=None, name=""):
            edge_x = []
            edge_y = []
            edge_text = []

            # Create a dictionary to store edge counts between node pairs
            edge_counts = {}
            for u, v in edge_list:
                #  Sort nodes to handle both directions
                nodes = tuple(sorted([u, v]))
                edge_counts[nodes] = edge_counts.get(nodes, 0) + 1

            for idx, (u, v) in enumerate(edge_list):
                x0, y0 = pos[u]
                x1, y1 = pos[v]

                # Calculate curve parameters
                nodes = tuple(sorted([u, v]))
                current_edge_num = edge_counts[nodes]

                # Add curvature based on the edge number and total edges between these nodes
                if current_edge_num > 1:
                    # Calculate the offset for this edge
                    offset = 0.2 * (idx % 2 * 2 - 1)  # Alternates between -0.2 and 0.2

                    # Create curved path using quadratic Bezier curve
                    t = np.linspace(0, 1, 20)
                    # Midpoint
                    mp_x = (x0 + x1) / 2
                    mp_y = (y0 + y1) / 2
                    # Control point perpendicular to midpoint
                    dx = x1 - x0
                    dy = y1 - y0
                    # Perpendicular vector
                    nx = -dy * offset
                    ny = dx * offset
                    # Control point
                    cp_x = mp_x + nx
                    cp_y = mp_y + ny

                    # Quadratic Bezier curve
                    x = (1-t)**2 * x0 + 2*(1-t)*t * cp_x + t**2 * x1
                    y = (1-t)**2 * y0 + 2*(1-t)*t * cp_y + t**2 * y1
                else:
                    # For single edges, use a slight curve
                    t = np.linspace(0, 1, 20)
                    offset = 0.1
                    # Create a slight curve
                    cp_x = (x0 + x1) / 2
                    cp_y = (y0 + y1) / 2 + offset
                    x = (1-t)**2 * x0 + 2*(1-t)*t * cp_x + t**2 * x1
                    y = (1-t)**2 * y0 + 2*(1-t)*t * cp_y + t**2 * y1

                edge_x.extend(x)
                edge_y.extend(y)

                # Get sources for this edge
                sources = G[u][v].get('sources', [])
                pmids = [s.replace('[', '').replace(']', '').replace('.t', '')
                        for s in sources]
                hover_text = f"From: {u}<br>To: {v}<br>PMIDs:<br>" + "<br>".join(pmids)

                # Add hover text for all points along the curve
                edge_text.extend([hover_text] * len(x))

                # Add None to create a gap between edges
                edge_x.append(None)
                edge_y.append(None)
                edge_text.append(None)

            line_style = dict(color=color, width=2)
            if dash:
                line_style['dash'] = dash

            return go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=line_style,
                hoverinfo='text',
                text=edge_text,
                name=name,
                showlegend=True,
                hoveron='points'
            )

        # Create activation and inhibition traces
        activation_trace = create_edge_trace(
            activation_edges,
            color='green',
            name='Activation'
        )

        inhibition_trace = create_edge_trace(
            inhibition_edges,
            color='red',
            dash='dash',
            name='Inhibition'
        )

        # Create figure
        fig = go.Figure(data=[activation_trace, inhibition_trace, node_trace])

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Mechanism Network: {perturbation_node} Perturbation',
                x=0.5,
                y=0.95
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=800,
            height=600
        )

        print("Visualization completed successfully")
        return fig

    except Exception as e:
        print(f"Error in visualize_shortest_paths: {str(e)}")
        traceback.print_exc()
        raise

@app.route('/test-mechanism', methods=['GET'])
def test_mechanism():
    return jsonify({
        'status': 'ok',
        'message': 'Mechanism endpoint is working'
    })

@app.route('/analyze-mechanism', methods=['GET', 'POST'])
def analyze_mechanism():
    print("Analyze mechanism endpoint called with method:", request.method)

    if request.method == 'GET':
        return jsonify({
            'status': 'ok',
            'message': 'Mechanism analysis endpoint is ready'
        })

    try:
        if not request.is_json:
            print("Request is not JSON")
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        print("Received data:", data)

        perturbation_gene = data.get('perturbationGene')
        genes_of_interest = data.get('genes', [])
        cell_type = data.get('cellType', 'NK')

        print(f"Processing request for: {perturbation_gene} with genes: {genes_of_interest}")

        if not perturbation_gene:
            return jsonify({'error': 'Missing perturbation gene'}), 400

        if not genes_of_interest:
            return jsonify({'error': 'Missing genes of interest'}), 400

        # Get the network for the selected cell type
        try:
            nodes_file, rels_file = get_files_for_cell_type(cell_type)
            print(f"Loading network from: {nodes_file}, {rels_file}")
            network = reconstruct_graph_from_csv(nodes_file, rels_file)
            print(f"Network loaded with {len(network.nodes())} nodes")
        except Exception as e:
            print(f"Error loading network: {str(e)}")
            return jsonify({'error': f'Error loading network: {str(e)}'}), 500

        # Check if genes exist in network
        missing_genes = []
        for gene in [perturbation_gene] + genes_of_interest:
            if gene not in network:
                missing_genes.append(gene)

        if missing_genes:
            print(f"Missing genes: {missing_genes}")
            return jsonify({
                'error': f'The following genes were not found in the network: {", ".join(missing_genes)}'
            }), 404

        try:
            print("Generating visualization...")
            fig = visualize_shortest_paths(network, perturbation_gene, genes_of_interest)
            plot_json = fig.to_json()
            print("Visualization generated successfully")

            return jsonify({
                'plot': plot_json
            })

        except Exception as viz_error:
            print(f"Error generating visualization: {str(viz_error)}")
            traceback.print_exc()
            return jsonify({'error': f'Error generating visualization: {str(viz_error)}'}), 500

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

