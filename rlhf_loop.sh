#!/bin/bash

# rlhf_loop.sh - Automates the RLHF training loop for TurtleBot3
# Usage: ./rlhf_loop.sh [options]

# Default values
FEEDBACK_THRESHOLD=5  # Number of feedback samples needed to proceed
ITERATIONS=3          # Number of iterations to run
TRAJECTORIES=10       # Number of trajectories to collect per iteration
POLICY_STEPS=10000    # Steps for policy training
REWARD_WEIGHT=0.3     # Weight for original reward vs learned reward

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --feedback)
            FEEDBACK_THRESHOLD="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --trajectories)
            TRAJECTORIES="$2"
            shift 2
            ;;
        --policy-steps)
            POLICY_STEPS="$2"
            shift 2
            ;;
        --reward-weight)
            REWARD_WEIGHT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./rlhf_loop.sh [options]"
            echo "Options:"
            echo "  --feedback N       Set feedback threshold to N samples (default: 5)"
            echo "  --iterations N     Run N iterations of the RLHF loop (default: 3)"
            echo "  --trajectories N   Collect N trajectories per iteration (default: 3)"
            echo "  --policy-steps N   Train policy for N steps (default: 10000)"
            echo "  --reward-weight W  Set original reward weight to W (0-1) (default: 0.3)"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Ensure ROS2 is sourced
source ~/ros2_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger

# Function to count feedback samples
count_feedback() {
    FEEDBACK_FILE=~/ros2_ws/src/turtlebot3_gym/feedback_log.json
    if [ -f "$FEEDBACK_FILE" ]; then
        # Count array elements in the JSON file
        COUNT=$(grep -o -i "\{" $FEEDBACK_FILE | wc -l)
        echo $COUNT
    else
        echo 0
    fi
}

# Function to run processes with timeout
run_with_timeout() {
    local cmd="$1"
    local timeout="$2"
    local message="$3"
    
    echo "$message"
    
    # Start the command in background
    eval "$cmd" &
    local pid=$!
    
    # Wait for timeout or process completion
    local count=0
    while kill -0 $pid 2>/dev/null; do
        sleep 1
        ((count++))
        if [ $count -ge $timeout ]; then
            echo "Timeout reached ($timeout seconds), terminating process..."
            kill $pid 2>/dev/null
            wait $pid 2>/dev/null
            return 1
        fi
    done
    
    # Process completed normally
    wait $pid
    return $?
}

# Create a modified feedback server command that will exit automatically
create_feedback_server_script() {
    local threshold=$1
    local initial_count=$2
    local timeout=$3
    
    cat > ~/temp_feedback_server.py << EOF
#!/usr/bin/env python3
import subprocess
import time
import os
import json
import signal
import sys

# Function to count feedback samples
def count_feedback():
    feedback_file = os.path.expanduser('~/ros2_ws/src/turtlebot3_gym/feedback_log.json')
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r') as f:
            try:
                data = json.load(f)
                return len(data)
            except json.JSONDecodeError:
                return 0
    return 0

# Start the feedback server
server_process = subprocess.Popen(
    ["ros2", "run", "turtlebot3_gym", "rlhf", "--start_feedback_server"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

initial_count = $initial_count
threshold = $threshold
timeout = $timeout
start_time = time.time()

print(f"Feedback server started. Please provide {threshold} feedback samples.")
print(f"Visit http://localhost:5000 in your browser.")

try:
    while True:
        # Check if server is still running
        if server_process.poll() is not None:
            print("Feedback server stopped unexpectedly.")
            break
            
        # Check feedback count
        current_count = count_feedback()
        new_feedback = current_count - initial_count
        
        if new_feedback >= threshold:
            print(f"Received {new_feedback} feedback samples. Threshold reached!")
            break
            
        # Check timeout
        if (time.time() - start_time) > timeout:
            print(f"Timeout reached ({timeout} seconds). Moving to next step.")
            break
            
        # Print status every 5 seconds
        if int(time.time()) % 5 == 0:
            print(f"Waiting for more feedback: {new_feedback}/{threshold} samples provided...")
            time.sleep(1)
        else:
            time.sleep(0.5)
            
except KeyboardInterrupt:
    print("User interrupted. Moving to next step.")
finally:
    # Terminate the server
    if server_process.poll() is None:
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
    
    print("Feedback collection completed.")
EOF

    chmod +x ~/temp_feedback_server.py
}

# Main RLHF loop
for ((i=1; i<=$ITERATIONS; i++)); do
    echo "=========================================================="
    echo "               RLHF ITERATION $i/$ITERATIONS"
    echo "=========================================================="
    
    # Step 1: Collect trajectories
    echo "Step 1: Collecting $TRAJECTORIES trajectories..."
    run_with_timeout "ros2 run turtlebot3_gym rlhf --collect_trajectories --num_trajectories $TRAJECTORIES" 600 "Collecting trajectories (timeout: 10 minutes)..."
    
    # Step 2: Gather human feedback
    echo "Step 2: Starting feedback server. Please provide feedback on trajectories."
    echo "Please provide at least $FEEDBACK_THRESHOLD feedback samples."
    
    # Initial feedback count
    INITIAL_COUNT=$(count_feedback)
    
    # Create and run self-terminating feedback server script
    create_feedback_server_script $FEEDBACK_THRESHOLD $INITIAL_COUNT 1800  # 30 minute timeout
    python3 ~/temp_feedback_server.py
    rm ~/temp_feedback_server.py
    
    # Step 3: Train reward model
    echo "Step 3: Training reward model..."
    run_with_timeout "ros2 run turtlebot3_gym rlhf --train_reward" 1800 "Training reward model (timeout: 30 minutes)..."
    
    # Step 4: Train policy with learned reward
    echo "Step 4: Training policy with learned reward..."
    
    # Create a self-terminating training script
    cat > ~/temp_policy_trainer.sh << EOF
#!/bin/bash
source ~/ros2_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger

# Launch Gazebo
ros2 launch turtlebot3_gym turtlebot3_gym.launch.py &
GAZEBO_PID=\$!

# Wait for Gazebo to initialize
echo "Waiting for Gazebo to initialize (15 seconds)..."
sleep 15

# Train the policy
echo "Training policy for $POLICY_STEPS steps..."
ros2 run turtlebot3_gym rlhf --train_policy --policy_steps $POLICY_STEPS --original_reward_weight $REWARD_WEIGHT

# Stop Gazebo when done
kill \$GAZEBO_PID 2>/dev/null
wait \$GAZEBO_PID 2>/dev/null
echo "Policy training completed."
EOF
    
    chmod +x ~/temp_policy_trainer.sh
    run_with_timeout "~/temp_policy_trainer.sh" 7200 "Training policy (timeout: 2 hours)..."
    rm ~/temp_policy_trainer.sh
    
    # Step 5: Evaluate the policy
    echo "Step 5: Evaluating trained policy..."
    
    # Create a self-terminating evaluation script
    cat > ~/temp_policy_evaluator.sh << EOF
#!/bin/bash
source ~/ros2_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger

# Launch Gazebo
ros2 launch turtlebot3_gym turtlebot3_gym.launch.py &
GAZEBO_PID=\$!

# Wait for Gazebo to initialize
echo "Waiting for Gazebo to initialize (15 seconds)..."
sleep 15

# Evaluate the policy
echo "Evaluating policy..."
ros2 run turtlebot3_gym rlhf --evaluate

# Stop Gazebo when done
kill \$GAZEBO_PID 2>/dev/null
wait \$GAZEBO_PID 2>/dev/null
echo "Policy evaluation completed."
EOF
    
    chmod +x ~/temp_policy_evaluator.sh
    run_with_timeout "~/temp_policy_evaluator.sh" 900 "Evaluating policy (timeout: 15 minutes)..."
    rm ~/temp_policy_evaluator.sh
    
    echo "Iteration $i/$ITERATIONS complete!"
    
    if [ $i -lt $ITERATIONS ]; then
        echo "Starting next iteration in 5 seconds..."
        sleep 5
    fi
done

echo "=========================================================="
echo "               RLHF TRAINING COMPLETE"
echo "=========================================================="
echo "Completed $ITERATIONS iterations of the RLHF loop."
echo "The final policy model is saved at ~/ros2_ws/src/turtlebot3_gym/policy_model.zip"