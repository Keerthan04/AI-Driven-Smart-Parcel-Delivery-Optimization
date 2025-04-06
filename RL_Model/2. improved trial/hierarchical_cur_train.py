import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np
import time

def train_hierarchical_with_curriculum(env_class, curriculum_generator, base_path="./models"):
    """
    Train a DQN agent on the hierarchical environment using curriculum learning
    
    Args:
        env_class: The environment class (HierarchicalUdupiDeliveryEnv)
        curriculum_generator: Initialized curriculum generator
        base_path: Directory to save models and logs
    """
    # Create paths
    os.makedirs(base_path, exist_ok=True)
    log_path = os.path.join(base_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    
    # Generate or load curriculum datasets
    curriculum_data_path = os.path.join(base_path, "curriculum_data")
    stages = curriculum_generator.save_curriculum_datasets(base_path=curriculum_data_path)
    
    # Initialize tracking variables
    model = None
    best_mean_reward = -np.inf
    
    # Training loop through curriculum stages
    for i, stage in enumerate(stages):
        print(f"\n{'='*20} CURRICULUM STAGE {i+1}/{len(stages)} {'='*20}")
        print(f"Difficulty: {stage['difficulty']}, Deliveries: {stage['deliveries']}")
        
        # Set up environment with this curriculum stage
        delivery_file = os.path.join(curriculum_data_path, stage['filename'])
        env = env_class(graph_path=curriculum_generator.graph_path, 
                       delivery_file=delivery_file)
        
        # Create a monitored environment for logging
        monitor_path = os.path.join(log_path, f"stage_{i+1}")
        os.makedirs(monitor_path, exist_ok=True)
        env = Monitor(env, monitor_path)
        
        # Set up callbacks for evaluation and checkpoints
        eval_callback = EvalCallback(
            env,
            best_model_save_path=os.path.join(base_path, f"best_model_stage_{i+1}"),
            log_path=os.path.join(log_path, f"eval_stage_{i+1}"),
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=os.path.join(base_path, f"checkpoints_stage_{i+1}"),
            name_prefix="hierarchical_dqn"
        )
        
        # Define hyperparameters - increase complexity with curriculum stage
        learning_rate = 0.0003 * (1.0 - 0.1 * i)  # Decrease learning rate slightly as complexity increases
        
        # Either create new model or continue training
        if model is None:
            model = DQN(
                policy="MlpPolicy",
                env=env,
                verbose=1,
                learning_rate=learning_rate,
                buffer_size=100000,
                learning_starts=2000,
                batch_size=64,
                tau=0.005,
                gamma=0.98,
                train_freq=4,
                target_update_interval=1000,
                exploration_fraction=0.3 - 0.05 * i,  # Decrease exploration as we progress
                exploration_final_eps=0.05,
                device='auto'
            )
        else:
            # Update the model to work with the new environment
            model.set_env(env)
            # Optionally adjust learning rate for continued training
            model.learning_rate = learning_rate
        
        # Calculate training steps based on curriculum stage
        timesteps = 50000 * (1 + i // 2)  # More steps for more complex stages
        
        # Train the model
        start_time = time.time()
        model.learn(
            total_timesteps=timesteps,
            callback=[eval_callback, checkpoint_callback],
            reset_num_timesteps=False  # Continue timestep counting across stages
        )
        training_time = time.time() - start_time
        
        # Save stage model
        model_path = os.path.join(base_path, f"hierarchical_dqn_stage_{i+1}")
        model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        # Evaluate current stage
        mean_reward, std_reward = evaluate_hierarchical_model(
            model, env, n_eval_episodes=10, stage_name=f"Stage {i+1}"
        )
        
        # Track best model
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            # Save a copy of this as the best so far
            best_model_path = os.path.join(base_path, "hierarchical_dqn_best")
            model.save(best_model_path)
            print(f"New best model saved to: {best_model_path}")
        
        print(f"Stage {i+1} completed in {training_time:.2f} seconds")
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Final training on full problem
    print("\n" + "="*20 + " FINAL TRAINING " + "="*20)
    
    # Create final environment with real data
    env = env_class(graph_path=curriculum_generator.graph_path)  # Use default delivery file
    env = Monitor(env, os.path.join(log_path, "final"))
    
    # Set up final callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(base_path, "best_model_final"),
        log_path=os.path.join(log_path, "eval_final"),
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(base_path, "checkpoints_final"),
        name_prefix="hierarchical_dqn"
    )
    
    # Update environment
    model.set_env(env)
    
    # Final training
    start_time = time.time()
    model.learn(
        total_timesteps=100000,
        callback=[eval_callback, checkpoint_callback],
        reset_num_timesteps=False
    )
    training_time = time.time() - start_time
    
    # Save final model
    final_model_path = os.path.join(base_path, "hierarchical_dqn_final")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Final evaluation
    mean_reward, std_reward = evaluate_hierarchical_model(
        model, env, n_eval_episodes=20, stage_name="Final"
    )
    
    print(f"Final training completed in {training_time:.2f} seconds")
    print(f"Final mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return model, final_model_path

def evaluate_hierarchical_model(model, env, n_eval_episodes=10, stage_name=""):
    """
    Evaluate model performance with extended metrics
    
    Args:
        model: Trained model to evaluate
        env: Environment to evaluate on
        n_eval_episodes: Number of episodes to evaluate
        stage_name: Name of curriculum stage for reporting
    
    Returns:
        mean_reward, std_reward
    """
    print(f"\n{'='*10} Evaluating Model: {stage_name} {'='*10}")
    
    # Track metrics
    rewards = []
    delivery_counts = []
    on_time_percentages = []
    early_percentages = []
    late_percentages = []
    
    for ep in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        # Collect metrics
        rewards.append(total_reward)
        
        # Extract stats from info
        stats = info.get('stats', {})
        total_deliveries = stats.get('on_time_deliveries', 0) + \
                          stats.get('early_arrivals', 0) + \
                          stats.get('late_deliveries', 0)
        
        if total_deliveries > 0:
            delivery_counts.append(total_deliveries)
            on_time_percentages.append(stats.get('on_time_deliveries', 0) / total_deliveries * 100)
            early_percentages.append(stats.get('early_arrivals', 0) / total_deliveries * 100)
            late_percentages.append(stats.get('late_deliveries', 0) / total_deliveries * 100)
        
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}, Deliveries = {total_deliveries}")
    
    # Calculate overall metrics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    # Print evaluation summary
    print(f"\n{stage_name} Evaluation Summary:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    if delivery_counts:
        print(f"Average deliveries completed: {np.mean(delivery_counts):.1f}")
        print(f"On-time delivery rate: {np.mean(on_time_percentages):.1f}%")
        print(f"Early arrival rate: {np.mean(early_percentages):.1f}%")
        print(f"Late delivery rate: {np.mean(late_percentages):.1f}%")
    
    return mean_reward, std_reward