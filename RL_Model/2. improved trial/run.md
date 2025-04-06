# Implementation Notes

Key Adaptations for Hierarchical Environment:

The curriculum generator now leverages the hierarchical structure (clusters, gateways)
Training gradually increases complexity by adding more clusters and tighter time windows
The evaluation metrics are enhanced to track on-time/early/late delivery statistics

## How Training Works

Stage 1: Simple deliveries within a single cluster
Stage 2: Multiple adjacent clusters with comfortable time windows
Stage 3: Introduction of gateway nodes and tighter constraints
Stage 4: Full map coverage with realistic time windows
Stage 5: Complex prioritized deliveries with tight scheduling

## Running the Training

To run the full curriculum training:

python hierarchical_training_main.py --graph_path path/to/udupi.graphml --model_dir ./hierarchical_models

For direct training (skipping curriculum):
python hierarchical_training_main.py --skip_curriculum

For evaluation only:
python hierarchical_training_main.py --eval_only --model_path ./hierarchical_models/hierarchical_dqn_final

## Advanced Tips

- Hyperparameter Tuning: The learning rate and exploration rate decrease as curriculum stages progress. You might want to experiment with these values.
- Monitoring Progress: The training script saves models at each curriculum stage and includes monitoring callbacks for tracking learning progress.
- Custom Evaluation: The evaluation function provides detailed metrics including on-time delivery rates, which are critical for logistics problems.
- Integration with Baseline3: The scripts are compatible with Stable-Baselines3's ecosystem, allowing you to easily switch between different RL algorithms if needed.
