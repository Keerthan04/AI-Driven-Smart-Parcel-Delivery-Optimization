import osmnx as ox

place_name = "Udupi, Karnataka, India"
# Get the graph for the place
G = ox.graph_from_place(place_name, network_type='drive')
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)
'''
The function ox.add_edge_speeds(G) goes through each road segment (edge) in your network and assigns a speed value to it. It looks for speed limits (typically provided in the OSM "maxspeed" tags) and, if not available, it assigns a default speed based on the type of road.

Then, ox.add_edge_travel_times(G) uses these speed values along with each edge’s length to calculate the travel time—typically in seconds—for traversing that segment. This adds a new travel_time attribute to every edge, which is very useful for routing or time-based analysis.
'''

#save the graph to a file
ox.save_graphml(G, "udupi.graphml")

#get the nodes and edges from the graph
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

#save the nodes and edges to csv files
nodes.to_csv("nodes.csv", index=False)
edges.to_csv("edges.csv", index=False)

#view the first few rows of the nodes and edges dataframes
print(nodes.head())
print(edges.head())