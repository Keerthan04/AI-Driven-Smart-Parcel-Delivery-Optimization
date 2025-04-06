import numpy as np
import pandas as pd
import random
import os
import networkx as nx
from collections import defaultdict

class HierarchicalCurriculumGenerator:
    def __init__(self, graph_path="udupi.graphml", env_class=None):
        """
        Initialize curriculum generator for hierarchical delivery environment
        
        Args:
            graph_path: Path to the graphml file
            env_class: The environment class to use (HierarchicalUdupiDeliveryEnv)
        """
        self.graph_path = graph_path
        self.env_class = env_class
        
        # Create a temporary environment to access the graph
        if env_class:
            self.env = env_class(graph_path=graph_path)
            self.G = self.env.G
        else:
            # Fallback to loading graph directly
            import osmnx as ox
            self.G = ox.load_graphml(graph_path)
            
        self.nodes = list(self.G.nodes)
        
        # Extract graph properties for better curriculum generation
        self._analyze_network()
    
    def _analyze_network(self):
        """Analyze network properties for better delivery planning"""
        # Identify node importance
        self.degrees = dict(self.G.degree())
        
        # Basic centrality metrics (using a sample for efficiency)
        self.centrality = nx.betweenness_centrality(self.G, k=100)
        
        # Get node coordinates
        self.node_coords = {}
        for node in self.nodes:
            self.node_coords[node] = (self.G.nodes[node]['y'], self.G.nodes[node]['x'])
        
        # Find map center
        avg_lat = sum(y for y, _ in self.node_coords.values()) / len(self.node_coords)
        avg_lon = sum(x for _, x in self.node_coords.values()) / len(self.node_coords)
        self.center = (avg_lat, avg_lon)
        
        # Calculate distances from center
        self.distances_from_center = {}
        for node, (lat, lon) in self.node_coords.items():
            self.distances_from_center[node] = ((lat - self.center[0])**2 + (lon - self.center[1])**2)**0.5
        
        # Create a simplified clustered representation (similar to _partition_network)
        self._create_clusters(num_clusters=5)
    
    def _create_clusters(self, num_clusters=5):
        """Create geographic clusters for hierarchical planning"""
        # Simple geographical partitioning
        min_lat = min(y for y, _ in self.node_coords.values())
        max_lat = max(y for y, _ in self.node_coords.values())
        min_lon = min(x for _, x in self.node_coords.values())
        max_lon = max(x for _, x in self.node_coords.values())
        
        # Create a grid
        lat_step = (max_lat - min_lat) / num_clusters
        lon_step = (max_lon - min_lon) / num_clusters
        
        # Assign nodes to clusters
        self.node_to_cluster = {}
        self.clusters = defaultdict(list)
        
        for node, (lat, lon) in self.node_coords.items():
            cluster_y = min(num_clusters-1, int((lat - min_lat) / lat_step))
            cluster_x = min(num_clusters-1, int((lon - min_lon) / lon_step))
            cluster_id = cluster_y * num_clusters + cluster_x
            self.node_to_cluster[node] = cluster_id
            self.clusters[cluster_id].append(node)
        
        # Identify gateway nodes - connecting different clusters
        self.gateways = set()
        for u, v in self.G.edges():
            if self.node_to_cluster.get(u, -1) != self.node_to_cluster.get(v, -1):
                self.gateways.add(u)
                self.gateways.add(v)
    
    def _select_central_nodes(self, percentage=50):
        """Select nodes from the central area of the map"""
        # Sort nodes by distance from center
        sorted_nodes = sorted(self.distances_from_center.keys(), 
                             key=lambda node: self.distances_from_center[node])
        
        # Return the closest X% of nodes
        num_central = int(len(sorted_nodes) * percentage / 100)
        return sorted_nodes[:num_central]
    
    def _select_cluster_nodes(self, num_clusters):
        """Select nodes from specific number of clusters"""
        # Choose a subset of clusters
        selected_clusters = random.sample(list(self.clusters.keys()), num_clusters)
        
        # Get all nodes from these clusters
        selected_nodes = []
        for cluster_id in selected_clusters:
            selected_nodes.extend(self.clusters[cluster_id])
            
        return selected_nodes
    
    def generate_delivery_dataset(self, num_deliveries, difficulty_level):
        """
        Generate deliveries with increasing difficulty using hierarchical knowledge:
        - Level 1: Deliveries in single cluster, wide time windows
        - Level 2: Deliveries in 2-3 adjacent clusters, reasonable windows
        - Level 3: Multi-cluster deliveries, some requiring gateways, tighter windows
        - Level 4: Full city coverage, tight windows, overlapping time slots
        - Level 5: Complex routing with prioritized deliveries and time constraints
        """
        deliveries = []
        
        # Select nodes based on difficulty level
        if difficulty_level == 1:
            # Use single cluster for easiest learning
            selected_cluster = random.choice(list(self.clusters.keys()))
            nodes_to_use = self.clusters[selected_cluster]
            window_sizes = [3]  # 3-hour windows
            time_ranges = [(9, 16)]  # 9 AM to 4 PM
        elif difficulty_level == 2:
            # Use 2-3 adjacent clusters
            nodes_to_use = self._select_cluster_nodes(min(3, len(self.clusters)))
            window_sizes = [2, 3]  # 2-3 hour windows
            time_ranges = [(9, 17)]  # 9 AM to 5 PM
        elif difficulty_level == 3:
            # Use central 80% of nodes
            nodes_to_use = self._select_central_nodes(percentage=80)
            # Add some gateway nodes to teach their importance
            gateway_sample = random.sample(list(self.gateways), min(len(self.gateways), num_deliveries // 3))
            nodes_to_use.extend(gateway_sample)
            window_sizes = [1, 2]  # 1-2 hour windows
            time_ranges = [(8, 18)]  # 8 AM to 6 PM
        elif difficulty_level == 4:
            # Use all nodes, balanced distribution
            nodes_to_use = self.nodes
            window_sizes = [1, 2]  # 1-2 hour windows
            time_ranges = [(8, 20)]  # 8 AM to 8 PM
        else:  # Level 5 (hard)
            # Use all nodes with priority on important ones
            nodes_to_use = self.nodes
            # Ensure some important nodes are selected
            important_nodes = [n for n, c in sorted(self.centrality.items(), 
                                                  key=lambda x: x[1], reverse=True)[:num_deliveries//3]]
            window_sizes = [1]  # 1-hour windows
            time_ranges = [(8, 20)]  # 8 AM to 8 PM
            
            # Add high-priority deliveries with tight windows
            for i in range(num_deliveries // 3):
                if important_nodes:
                    loc = important_nodes.pop()
                else:
                    loc = random.choice(nodes_to_use)
                    
                # Generate tight time window
                start_hour = random.randint(10, 16)
                end_hour = start_hour + 1
                
                deliveries.append({
                    "id": i,
                    "node": loc,
                    "slot_start": f"{start_hour}:00",
                    "slot_end": f"{end_hour}:00",
                    "priority": "high"  # Additional feature for future use
                })
        
        # Generate remaining deliveries
        remaining = num_deliveries - len(deliveries)
        for i in range(len(deliveries), len(deliveries) + remaining):
            loc = random.choice(nodes_to_use)
            
            # Select time window size and constraints
            window_size = random.choice(window_sizes)
            time_range = random.choice(time_ranges)
            
            # Generate start time within allowed range
            start_hour = random.randint(time_range[0], time_range[1] - window_size)
            end_hour = start_hour + window_size
            
            # Add priority field for level 5 (optional use in environment)
            priority = "normal"
            if difficulty_level == 5 and random.random() < 0.2:
                priority = "high"
            
            deliveries.append({
                "id": i,
                "node": loc,
                "slot_start": f"{start_hour}:00",
                "slot_end": f"{end_hour}:00",
                "priority": priority
            })
        
        return pd.DataFrame(deliveries)
    
    def save_curriculum_datasets(self, base_path="./curriculum_data"):
        """Generate and save complete curriculum datasets"""
        # Create directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)
        
        # Define curriculum stages
        stages = [
            {"difficulty": 1, "deliveries": 5, "filename": "curriculum_level1.csv"},
            {"difficulty": 2, "deliveries": 8, "filename": "curriculum_level2.csv"},
            {"difficulty": 3, "deliveries": 10, "filename": "curriculum_level3.csv"},
            {"difficulty": 4, "deliveries": 12, "filename": "curriculum_level4.csv"},
            {"difficulty": 5, "deliveries": 15, "filename": "curriculum_level5.csv"}
        ]
        
        # Generate and save datasets
        for stage in stages:
            df = self.generate_delivery_dataset(
                num_deliveries=stage['deliveries'],
                difficulty_level=stage['difficulty']
            )
            
            # Save to file
            file_path = os.path.join(base_path, stage['filename'])
            df.to_csv(file_path, index=False)
            print(f"Saved curriculum dataset: {file_path}")
            
        return stages