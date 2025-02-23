import streamlit as st
import networkx as nx
from pyvis.network import Network
import json
import yaml
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import tempfile
import streamlit.components.v1 as components

def load_graph_data(file_path):
    """Load graph data from JSON file"""
    with open(file_path, 'r') as f:
        graph_data = json.load(f)
    G = nx.node_link_graph(graph_data)
    return G

def get_available_runs():
    """Get list of available runs"""
    data_dir = Path("data")
    runs = sorted([d for d in data_dir.glob("run_*") if d.is_dir()],
                 key=lambda x: x.name,
                 reverse=True)  # Most recent first
    return runs

def format_run_name(run_dir):
    """Format run directory name into readable datetime"""
    try:
        # Extract timestamp from run_YYYYMMDD_HHMMSS format
        timestamp = run_dir.name.split('_', 1)[1]  # Get everything after first underscore
        dt = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
        return f"Run from {dt.strftime('%Y-%m-%d %H:%M:%S')}"
    except (IndexError, ValueError):
        return f"Run {run_dir.name}"  # Fallback if parsing fails

def load_run_config(run_dir):
    """Load configuration for a specific run"""
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None

def create_pyvis_graph(G, height="750px"):
    """Create an interactive Pyvis graph"""
    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="black")
    net.force_atlas_2based()
    
    # Add nodes
    for node in G.nodes(data=True):
        node_id = node[0]
        metadata = node[1].get('metadata', {})
        discovery_time = node[1].get('discovery_time', '')
        
        # Format tooltip with metadata
        tooltip = f"Topic: {node_id}<br>"
        tooltip += f"Discovery time: {discovery_time}<br>"
        for k, v in metadata.items():
            tooltip += f"{k}: {v}<br>"
        
        net.add_node(node_id, title=tooltip, label=node_id)
    
    # Add edges
    for edge in G.edges():
        net.add_edge(edge[0], edge[1])
    
    # Generate HTML file in a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
        net.save_graph(tmpfile.name)
        return tmpfile.name

def create_graph_metrics(G):
    """Calculate and display graph metrics"""
    metrics = {
        "Number of nodes": G.number_of_nodes(),
        "Number of edges": G.number_of_edges(),
        "Average degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "Number of connected components": nx.number_connected_components(G.to_undirected()),
        "Average clustering coefficient": nx.average_clustering(G.to_undirected()),
    }
    return metrics

def create_node_table(G):
    """Create a table of nodes with their metadata"""
    data = []
    for node, attrs in G.nodes(data=True):
        metadata = attrs.get('metadata', {})
        discovery_time = attrs.get('discovery_time', '')
        row = {
            'Topic': node,
            'Discovery Time': discovery_time,
            'Degree': G.degree(node),
            'Out Degree': G.out_degree(node),
            'In Degree': G.in_degree(node),
        }
        row.update(metadata)
        data.append(row)
    return pd.DataFrame(data)

def main():
    st.set_page_config(page_title="Refusal Graph Visualization", layout="wide")
    
    st.title("Refusal Graph Visualization Dashboard")
    
    # Run selection
    runs = get_available_runs()
    if not runs:
        st.error("No runs found in the data directory.")
        return
        
    selected_run = st.selectbox(
        "Select a run to visualize",
        runs,
        format_func=format_run_name
    )
    
    # Load and display run configuration
    run_config = load_run_config(selected_run)
    if run_config:
        with st.expander("Run Configuration"):
            st.json(run_config)
    
    # Get list of graph files for selected run
    graph_dir = selected_run / "graphs"
    graph_files = sorted(graph_dir.glob("refusal_graph_*.json"), 
                        key=lambda x: int(x.stem.split('_')[-1]) if x.stem.split('_')[-1].isdigit() else float('inf'))
    
    if not graph_files:
        st.warning(f"No graph files found in {graph_dir}")
        return
    
    # File selection
    selected_file = st.selectbox(
        "Select a graph file to visualize",
        graph_files,
        format_func=lambda x: f"Iteration {x.stem.split('_')[-1]} ({x.stat().st_mtime_ns})"
    )
    
    if selected_file:
        # Load and display graph
        G = load_graph_data(selected_file)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Network Visualization", "Graph Metrics", "Node Table"])
        
        with tab1:
            st.subheader("Interactive Network Visualization")
            # Create and display interactive graph
            html_path = create_pyvis_graph(G)
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=800)
            
        with tab2:
            st.subheader("Graph Metrics")
            metrics = create_graph_metrics(G)
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.2f}" if isinstance(value, float) else value)
            
            # Add degree distribution plot
            degrees = [d for n, d in G.degree()]
            fig = go.Figure(data=[go.Histogram(x=degrees, nbinsx=20)])
            fig.update_layout(
                title="Degree Distribution",
                xaxis_title="Degree",
                yaxis_title="Count"
            )
            st.plotly_chart(fig)
            
        with tab3:
            st.subheader("Node Table")
            df = create_node_table(G)
            st.dataframe(df)
            
            # Add download button
            csv = df.to_csv(index=False)
            st.download_button(
                "Download node data as CSV",
                csv,
                "node_data.csv",
                "text/csv",
                key='download-csv'
            )

if __name__ == "__main__":
    main() 