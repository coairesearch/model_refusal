import networkx as nx
import json
from datetime import datetime
from typing import Dict, List, Optional

class RefusalGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def add_topic(self, topic: str, parent_topic: Optional[str] = None, metadata: Optional[Dict] = None):
        """Add a new topic to the graph"""
        if topic not in self.graph:
            self.graph.add_node(
                topic,
                metadata=metadata or {},
                discovery_time=datetime.now().isoformat()
            )
            
        if parent_topic:
            if parent_topic not in self.graph:
                self.graph.add_node(
                    parent_topic,
                    metadata={},
                    discovery_time=datetime.now().isoformat()
                )
            self.graph.add_edge(parent_topic, topic)
            
    def get_topics(self) -> List[str]:
        """Get all topics in the graph"""
        return list(self.graph.nodes())
        
    def get_connected_components(self) -> List[List[str]]:
        """Get clusters of semantically related topics"""
        return list(nx.connected_components(self.graph.to_undirected()))
        
    def get_topic_metadata(self, topic: str) -> Dict:
        """Get metadata for a specific topic"""
        return self.graph.nodes[topic]
        
    def save_graph(self, filepath: str):
        """Save graph to file"""
        # Convert datetime objects to strings for serialization
        graph_data = nx.node_link_data(self.graph)
        
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)
            
    @classmethod
    def load_graph(cls, filepath: str) -> 'RefusalGraph':
        """Load graph from file"""
        with open(filepath, 'r') as f:
            graph_data = json.load(f)
            
        graph = cls()
        graph.graph = nx.node_link_graph(graph_data)
        return graph
        
    def get_frontier_topics(self, n: int = 5) -> List[str]:
        """Get n topics with the fewest outgoing edges"""
        out_degrees = dict(self.graph.out_degree())
        sorted_topics = sorted(out_degrees.items(), key=lambda x: x[1])
        return [topic for topic, _ in sorted_topics[:n]]
        
    def update_topic_metadata(self, topic: str, metadata: Dict):
        """Update metadata for a topic"""
        if topic in self.graph:
            current_metadata = self.graph.nodes[topic].get('metadata', {})
            current_metadata.update(metadata)
            self.graph.nodes[topic]['metadata'] = current_metadata 