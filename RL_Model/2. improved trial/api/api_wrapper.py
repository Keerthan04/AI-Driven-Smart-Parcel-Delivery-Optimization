import numpy as np
import pandas as pd
import os
from stable_baselines3 import DQN
from hierarchical_udupi_delivery_env import HierarchicalUdupiDeliveryEnv

class DeliveryRoutePlannerAPI:
    """API for using trained model to plan delivery routes in Udupi"""
    
    def __init__(self, model_path, graph_path="../udupi.graphml"):
        """
        Initialize the route planner with a trained model
        
        Args:
            model_path: Path to the trained DQN model
            graph_path: Path to the Udupi road network GraphML file
        """
        # Load the trained model
        self.model = DQN.load(model_path)
        self.graph_path = graph_path
        
        # We'll create the environment when we need it
        self.env = None
    
    def initialize_with_deliveries(self, deliveries_df=None, delivery_file=None, 
                                  start_node=None, start_time=8.0):
        """
        Initialize a new planning session with a set of deliveries
        
        Args:
            deliveries_df: Pandas DataFrame with delivery information
            delivery_file: Alternative - path to a CSV file with delivery information
            start_node: Starting node ID (default: random depot)
            start_time: Starting time in 24h format (default: 8.0 = 8:00 AM)
            
        Returns:
            Initial state information
        """
        # Create environment with the provided deliveries
        if deliveries_df is not None:
            # Save temporary file
            temp_file = "temp_deliveries.csv"
            deliveries_df.to_csv(temp_file, index=False)
            self.env = HierarchicalUdupiDeliveryEnv(graph_path=self.graph_path, 
                                                   delivery_file=temp_file)
            os.remove(temp_file)  # Clean up
        elif delivery_file is not None:
            self.env = HierarchicalUdupiDeliveryEnv(graph_path=self.graph_path,
                                                   delivery_file=delivery_file)
        else:
            # Use default deliveries
            self.env = HierarchicalUdupiDeliveryEnv(graph_path=self.graph_path)
        
        # Reset environment to initial state
        obs, _ = self.env.reset()
        
        # Override starting node and time if specified
        if start_node is not None:
            self.env.current_node = start_node
            
        if start_time is not None:
            self.env.current_time = start_time
            
        # Get updated observation after changes
        obs = self.env._get_obs()
        
        # Return the current state information
        return self._get_state_info()
    
    def get_next_delivery(self):
        """
        Get the recommended next delivery
        
        Returns:
            Dictionary with the recommended delivery information
        """
        if self.env is None:
            raise ValueError("Environment not initialized. Call initialize_with_deliveries first.")
        
        # Get current observation
        obs = self.env._get_obs()
        
        # Get model's recommendation
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Get information about the recommended delivery
        if action < len(self.env.deliveries) and action in self.env.undelivered:
            delivery_info = self.env.deliveries[action].copy()
            
            # Add estimated travel info
            current_node = self.env.current_node
            target_node = delivery_info['node']
            travel_time, path = self.env._get_optimal_path(current_node, target_node)
            
            delivery_info.update({
                "action_id": int(action),
                "estimated_travel_time_minutes": travel_time,
                "estimated_arrival_time": self.env.current_time + travel_time/60.0,
                "path": path
            })
            
            return delivery_info
        else:
            return {"error": "Invalid action recommended by model", "action_id": int(action)}
    
    def step_to_next_delivery(self):
        """
        Execute the recommended next delivery and update the system state
        
        Returns:
            Next state information, reward, and completion status
        """
        if self.env is None:
            raise ValueError("Environment not initialized. Call initialize_with_deliveries first.")
        
        # Get model's recommendation
        obs = self.env._get_obs()
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Take the step in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Return updated state info
        state_info = self._get_state_info()
        state_info.update({
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "done": terminated or truncated,
            "action_taken": int(action),
            "stats": info.get("stats", {})
        })
        
        return state_info
    
    def _get_state_info(self):
        """Get detailed information about the current state"""
        if self.env is None:
            raise ValueError("Environment not initialized")
            
        # Get remaining deliveries info
        remaining_deliveries = []
        for i in self.env.undelivered:
            delivery = self.env.deliveries[i].copy()
            travel_time, _ = self.env._get_optimal_path(self.env.current_node, delivery['node'])
            eta = self.env.current_time + travel_time/60.0
            slot_start = self.env._parse_hour(delivery['slot_start'])
            slot_end = self.env._parse_hour(delivery['slot_end'])
            
            status = "ON TIME" if slot_start <= eta <= slot_end else "EARLY" if eta < slot_start else "LATE"
            
            delivery.update({
                "id": i,
                "estimated_travel_time_minutes": travel_time,
                "estimated_arrival_time": eta,
                "arrival_status": status
            })
            remaining_deliveries.append(delivery)
        
        # Return state information
        return {
            "current_node": self.env.current_node,
            "current_time": self.env.current_time,
            "current_time_formatted": f"{int(self.env.current_time)}:{int((self.env.current_time % 1) * 60):02d}",
            "deliveries_completed": self.env.num_deliveries - len(self.env.undelivered),
            "deliveries_remaining": len(self.env.undelivered),
            "remaining_deliveries": remaining_deliveries
        }
    
    def plan_full_route(self):
        """
        Plan the entire delivery route from current state to completion
        
        Returns:
            List of delivery steps in order
        """
        if self.env is None:
            raise ValueError("Environment not initialized. Call initialize_with_deliveries first.")
        
        # Save the current environment state
        import copy
        saved_env_state = (
            copy.deepcopy(self.env.current_node),
            self.env.current_time,
            copy.deepcopy(self.env.undelivered),
            copy.deepcopy(self.env.stats)
        )
        
        # Plan the full route
        route_plan = []
        done = False
        
        while not done and len(self.env.undelivered) > 0:
            # Get next delivery recommendation
            next_delivery = self.get_next_delivery()
            step_result = self.step_to_next_delivery()
            
            # Add to route plan
            route_step = {
                "action_id": next_delivery.get("action_id"),
                "node": next_delivery.get("node"),
                "time_window": f"{next_delivery.get('slot_start')} - {next_delivery.get('slot_end')}",
                "travel_time_minutes": next_delivery.get("estimated_travel_time_minutes"),
                "estimated_arrival": step_result["current_time_formatted"],
                "status": "Completed"
            }
            route_plan.append(route_step)
            
            done = step_result["done"]
        
        # Restore the environment state
        self.env.current_node, self.env.current_time, self.env.undelivered, self.env.stats = saved_env_state
        
        return route_plan

# Example usage:
def demo_route_planner():
    """
    Demo of how to use the DeliveryRoutePlannerAPI
    """
    # Initialize the planner with a trained model
    planner = DeliveryRoutePlannerAPI(
        model_path="./hierarchical_models/hierarchical_dqn_final"
    )
    
    # Create sample deliveries
    deliveries = pd.DataFrame([
        {"node": 2345678, "slot_start": "9:00", "slot_end": "10:00"},
        {"node": 3456789, "slot_start": "10:30", "slot_end": "12:30"},
        {"node": 4567890, "slot_start": "13:00", "slot_end": "15:00"},
        {"node": 5678901, "slot_start": "14:00", "slot_end": "16:00"},
        {"node": 6789012, "slot_start": "16:30", "slot_end": "18:00"}
    ])
    
    # Initialize with these deliveries
    state = planner.initialize_with_deliveries(
        deliveries_df=deliveries,
        start_node=1234567,  # Example starting node
        start_time=8.5       # Start at 8:30 AM
    )
    
    print("Initial State:")
    print(f"Current Location: Node {state['current_node']}")
    print(f"Current Time: {state['current_time_formatted']}")
    print(f"Deliveries Remaining: {state['deliveries_remaining']}")
    
    # Get next recommended delivery
    next_delivery = planner.get_next_delivery()
    print("\nRecommended Next Delivery:")
    print(f"Delivery ID: {next_delivery['action_id']}")
    print(f"Delivery Node: {next_delivery['node']}")
    print(f"Time Window: {next_delivery['slot_start']} - {next_delivery['slot_end']}")
    print(f"Estimated Travel Time: {next_delivery['estimated_travel_time_minutes']:.1f} minutes")
    print(f"Estimated Arrival: {next_delivery['estimated_arrival_time']:.2f} hours")
    
    # Plan full route
    print("\nPlanning full route...")
    route_plan = planner.plan_full_route()
    
    print("\nFull Route Plan:")
    for i, step in enumerate(route_plan):
        print(f"Step {i+1}: Delivery to Node {step['node']} (Window: {step['time_window']})")
        print(f"  Travel Time: {step['travel_time_minutes']:.1f} min, Arrival: {step['estimated_arrival']}")
    
    return planner

if __name__ == "__main__":
    demo_route_planner()