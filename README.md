TurtleBot3 RLHF Integration - Quick Start Guide
What is RLHF?
Reinforcement Learning from Human Feedback (RLHF) lets you train robots by providing feedback on their behavior. Instead of just programming rules, you can show the robot what you like and don't like.
The RLHF Loop

Collect robot trajectories
Provide human feedback (which trajectories you prefer)
Train a reward model based on your preferences
Train a policy using the learned reward model
Repeat to continuously improve performance

Commands
Collect Trajectories
bashCopysource ~/ros2_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 run turtlebot3_gym rlhf --collect_trajectories --num_trajectories 5
Provide Feedback
bashCopysource ~/ros2_ws/install/setup.bash
ros2 run turtlebot3_gym rlhf --start_feedback_server
Then:

Open http://localhost:5000 in a browser
Compare trajectory pairs
Select which one you prefer (A, B, or Similar)
Optionally provide a reason for your preference
Continue providing feedback on multiple pairs

Train Reward Model
bashCopysource ~/ros2_ws/install/setup.bash
ros2 run turtlebot3_gym rlhf --train_reward
Train Policy with Learned Reward
bashCopysource ~/ros2_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 run turtlebot3_gym rlhf --train_policy --policy_steps 10000
Evaluate Results
bashCopysource ~/ros2_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 run turtlebot3_gym rlhf --evaluate
Tips

Provide feedback on at least 4-5 trajectory pairs for initial training
Focus on specific behaviors you want to encourage or discourage
After training a policy, collect new trajectories with that policy and provide more feedback
Use --policy_steps to control training duration (lower for quicker iterations)
Use --original_reward_weight (0-1) to balance between original task reward and human preferences
