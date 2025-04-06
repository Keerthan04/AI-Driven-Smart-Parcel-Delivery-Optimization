import os
import argparse
from hierarchical_curri_gen import HierarchicalCurriculumGenerator
from hierarchical_cur_train import train_hierarchical_with_curriculum, evaluate_hierarchical_model
from hierarchical_udupi_env import HierarchicalUdupiDeliveryEnv
from stable_baselines3 import DQN

def main():
    """Main training pipeline for hierarchical delivery agent"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train hierarchical delivery agent')
    parser.add_argument('--graph_path', type=str, default="../udupi.graphml", 
                        help='Path to GraphML file')
    parser.add_argument('--model_dir', type=str, default="./hierarchical_models",
                        help='Directory to save models')
    parser.add_argument('--skip_curriculum', action='store_true', 
                        help='Skip curriculum training and train directly on full problem')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate existing model without training')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model for evaluation')
    
    args = parser.parse_args()
    
    # Initialize curriculum generator
    print("Initializing curriculum generator...")
    curriculum_generator = HierarchicalCurriculumGenerator(
        graph_path=args.graph_path,
        env_class=HierarchicalUdupiDeliveryEnv
    )
    
    if args.eval_only:
        # Evaluation mode
        if not args.model_path:
            print("Error: Model path required for evaluation")
            return
        
        print(f"Loading model from {args.model_path} for evaluation")
        model = DQN.load(args.model_path)
        
        # Create environment
        env = HierarchicalUdupiDeliveryEnv(graph_path=args.graph_path)
        
        # Run evaluation
        mean_reward, std_reward = evaluate_hierarchical_model(
            model, env, n_eval_episodes=20, stage_name="Model Evaluation"
        )
        
        print(f"Evaluation complete! Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    elif args.skip_curriculum:
        # Direct training on full problem
        print("Skipping curriculum, training directly on full problem...")
        
        # Create environment
        env = HierarchicalUdupiDeliveryEnv(graph_path=args.graph_path)
        
        # Create model
        model = DQN(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=0.0003,
            buffer_size=100000,
            learning_starts=5000,
            batch_size=64,
            tau=0.005,
            gamma=0.98,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            device='auto'
        )
        
        # Train model
        print("Training model...")
        model.learn(total_timesteps=200000)
        
        # Save model
        os.makedirs(args.model_dir, exist_ok=True)
        model_path = os.path.join(args.model_dir, "hierarchical_dqn_direct")
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Evaluate model
        mean_reward, std_reward = evaluate_hierarchical_model(
            model, env, n_eval_episodes=20, stage_name="Direct Training"
        )
        
        print(f"Training and evaluation complete! Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    else:
        # Curriculum training
        print("Starting curriculum training...")
        model, model_path = train_hierarchical_with_curriculum(
            env_class=HierarchicalUdupiDeliveryEnv,
            curriculum_generator=curriculum_generator,
            base_path=args.model_dir
        )
        
        print(f"Curriculum training complete! Final model saved to {model_path}")

if __name__ == "__main__":
    main()