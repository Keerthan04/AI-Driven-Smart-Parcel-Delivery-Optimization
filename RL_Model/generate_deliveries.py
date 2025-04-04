import osmnx as ox
import random
import pandas as pd
path = "./udupi.graphml"
G = ox.load_graphml(path)

nodes = list(G.nodes)

def generate_deliveries(num=10):
    deliveries = []
    for i in range(num):
        loc = random.choice(nodes)
        start_hour = random.choice([9, 11, 13, 15])  # delivery window
        end_hour = start_hour + 1
        deliveries.append({
            "id": i,
            "node": loc,
            "slot_start": f"{start_hour}:00",
            "slot_end": f"{end_hour}:00"
        })
    return pd.DataFrame(deliveries)

# Create delivery set and save
df_deliveries = generate_deliveries(10)
df_deliveries.to_csv("udupi_deliveries.csv", index=False)
