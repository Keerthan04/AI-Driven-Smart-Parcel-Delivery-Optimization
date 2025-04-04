import gym
import numpy as np
import networkx as nx
import pandas as pd
import random
import osmnx as ox

class UdupiDeliveryEnv(gym.Env):
    def __init__(self, graph_path="udupi.graphml", delivery_file="udupi_deliveries.csv"):
        super(UdupiDeliveryEnv, self).__init__()

        self.G = ox.load_graphml(graph_path)
        self.deliveries_df = pd.read_csv(delivery_file)

        self.deliveries = self.deliveries_df.to_dict('records')
        self.num_deliveries = len(self.deliveries)

        self.depot = random.choice(list(self.G.nodes))
        self.current_node = self.depot
        self.current_time = 8  # start day at 8:00

        self.undelivered = list(range(self.num_deliveries))  # delivery IDs
        self.max_time = 20  # ends at 8 PM

        # Observation: current node, time, delivery slots (flattened)
        self.observation_space = gym.spaces.Box(
            low=0, high=999999, shape=(2 + self.num_deliveries * 3,), dtype=np.float32
        )

        # Action: choose next delivery index
        self.action_space = gym.spaces.Discrete(self.num_deliveries)

    def reset(self):
        self.current_node = self.depot
        self.current_time = 8
        self.undelivered = list(range(self.num_deliveries))
        return self._get_obs()

    def _get_obs(self):
        obs = [self.current_node, self.current_time]
        for i in range(self.num_deliveries):
            if i in self.undelivered:
                delivery = self.deliveries[i]
                obs.extend([delivery['node'], self._parse_hour(delivery['slot_start']), self._parse_hour(delivery['slot_end'])])
            else:
                obs.extend([-1, -1, -1])
        return np.array(obs, dtype=np.float32)

    def _parse_hour(self, time_str):
        return int(time_str.split(":")[0])

    def _get_travel_time(self, u, v):
        try:
            path = nx.shortest_path(self.G, u, v, weight='travel_time')
            time = sum(self.G[path[i]][path[i+1]][0]['travel_time'] for i in range(len(path)-1))
            return time / 60.0  # convert seconds to minutes
        except:
            return 999  # large penalty if path not found

    #IMP-> the normal function which choses actions randomly and returns the reward
    # def step(self, action):
    #     if action not in self.undelivered:
    #         return self._get_obs(), -20, False, {}  # penalty for invalid move

    #     delivery = self.deliveries[action]
    #     target_node = delivery['node']
    #     travel_time = self._get_travel_time(self.current_node, target_node)
    #     self.current_time += travel_time / 60.0  # convert min to hours

    #     slot_start = self._parse_hour(delivery['slot_start'])
    #     slot_end = self._parse_hour(delivery['slot_end'])

    #     reward = -travel_time  # penalize time
    #     if slot_start <= self.current_time <= slot_end:
    #         reward += 20
    #     else:
    #         reward -= 10

    #     self.current_node = target_node
    #     self.undelivered.remove(action)
    #     done = len(self.undelivered) == 0 or self.current_time >= self.max_time

    #     return self._get_obs(), reward, done, {}

    #IMP->this step function prioritize the delivereis based on time slots and choose the one which has near about time slot
    def step(self, action):
        # Ensure the action is valid (i.e., the delivery has not been completed yet)
        if action not in self.undelivered:
            return self._get_obs(), -20, False, {}  # penalty for invalid move
        
        # Get the selected delivery based on the action
        delivery = self.deliveries[action]
        target_node = delivery['node']
        travel_time = self._get_travel_time(self.current_node, target_node)
        self.current_time += travel_time / 60.0  # convert min to hours
        
        # Extract the start and end hours of the delivery window
        slot_start = self._parse_hour(delivery['slot_start'])
        slot_end = self._parse_hour(delivery['slot_end'])
        
        # Default reward is based on travel time (penalty for long travel)
        reward = -travel_time  
        
        # If the agent delivers within the preferred time window
        if slot_start <= self.current_time <= slot_end:
            reward += 20  # Positive reward for delivering within the time window
            self.undelivered.remove(action)  # Mark the delivery as completed
        else:
            reward -= 10  # Penalty for being outside the time window

        # Now, prioritize the next action based on the closest time window
        # Sort the remaining undelivered deliveries by their time slot start time
        sorted_undelivered = sorted(self.undelivered, key=lambda x: self._parse_hour(self.deliveries[x]['slot_start']))

        # Choose the next delivery based on the sorted undelivered list
        # The next action will be the first task in the sorted list
        if sorted_undelivered:
            next_action = sorted_undelivered[0]
        else:
            next_action = None

        done = len(self.undelivered) == 0 or self.current_time >= self.max_time
        
        return self._get_obs(), reward, done, {"next_action": next_action}

    # def render(self, mode="human"):
    #     print(f"Current Node: {self.current_node}, Time: {self.current_time:.2f}h, Undelivered: {self.undelivered}")
    def render(self, mode="human"):
        # Display the current state in a more detailed manner
        print(f"Current Node: {self.current_node}")
        print(f"Current Time: {self.current_time:.2f}h")
        print(f"Undelivered Tasks: {self.undelivered}")
        
        # Show the current delivery being completed (if any)
        if len(self.undelivered) < self.num_deliveries:
            print(f"Last Delivery Completed: Delivery ID {self.num_deliveries - len(self.undelivered)}")
        
        # Print remaining deliveries
        remaining_deliveries = [self.deliveries[i] for i in self.undelivered]
        print(f"Remaining Deliveries: {len(self.undelivered)}")
        
        # Show the list of undelivered deliveries with their time slots
        for task in remaining_deliveries:
            print(f"Delivery ID {task['id']}: Location {task['node']} | Slot {task['slot_start']} to {task['slot_end']}")

        print(f"Total Deliveries: {self.num_deliveries}")
        print(f"Completed Deliveries: {self.num_deliveries - len(self.undelivered)}")


from udupi_delivery_env import UdupiDeliveryEnv

env = UdupiDeliveryEnv()
obs = env.reset()

done = False
while not done:
    env.render()
    action = random.choice(env.undelivered)
    obs, reward, done, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}")
