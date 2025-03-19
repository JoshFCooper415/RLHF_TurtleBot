#!/bin/bash
# setup_templates.sh - Script to set up template and static directories for TurtleBot3 RLHF

# Set up paths
PACKAGE_DIR=~/ros2_ws/src/turtlebot3_gym
TEMPLATE_DIR=$PACKAGE_DIR/templates
STATIC_DIR=$PACKAGE_DIR/static
CSS_DIR=$STATIC_DIR/css

# Create directories if they don't exist
mkdir -p $TEMPLATE_DIR
mkdir -p $CSS_DIR

# Copy templates from the paste
echo "Creating trajectory.html template..."
cat > $TEMPLATE_DIR/trajectory.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trajectory Details - TurtleBot3 RLHF Feedback</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    <style>
        /* Add styles for canvas-based visualization */
        .trajectory-canvas {
            display: block;
            margin: 0 auto;
            background-color: #f9f9f9;
            border: 1px solid #eee;
            border-radius: 4px;
        }
        .obstacle-info {
            margin-top: 10px;
            padding: 10px;
            background-color: #f8f8f8;
            border-left: 3px solid #666;
        }
        .legend {
            margin-top: 10px;
            font-size: 12px;
        }
        .legend-item {
            display: inline-block;
            margin-right: 15px;
        }
        .legend-color {
            display: inline-block;
            width: 12px;
            height: 12px;
            margin-right: 5px;
            vertical-align: middle;
        }
        .start-point {
            background-color: green;
        }
        .end-point.success {
            background-color: #4CAF50;
        }
        .end-point.failure {
            background-color: #F44336;
        }
        .target-point {
            border: 2px dashed #FF9800;
            background-color: transparent;
        }
        .obstacle-point {
            background-color: rgba(128, 128, 128, 0.5);
            border: 1px solid #666;
        }
    </style>
</head>
<body>
    <header>
        <h1>TurtleBot3 Trajectory Details</h1>
        <a href="{{ url_for('index') }}" class="back-btn">← Back to List</a>
    </header>
    
    <main class="trajectory-detail">
        <div class="trajectory-info">
            <h2>Trajectory: {{ trajectory.filename or trajectory.id }}</h2>
            
            <div class="trajectory-visualization">
                {% if trajectory.plot_path %}
                    <img src="{{ url_for('static', filename=trajectory.plot_path) }}" alt="Trajectory plot" class="full-size-plot">
                {% else %}
                    <div id="trajectory-canvas-container" data-trajectory="{{ trajectory|tojson }}">
                        <canvas id="trajectory-canvas" class="trajectory-canvas" width="600" height="600"></canvas>
                        <div class="legend">
                            <div class="legend">
                                <div class="legend-item">
                                    <span class="legend-color start-point"></span>
                                    <span>Start</span>
                                </div>
                                <div class="legend-item">
                                    <span class="legend-color end-point {% if trajectory.metadata.success %}success{% else %}failure{% endif %}"></span>
                                    <span>End</span>
                                </div>
                                <div class="legend-item">
                                    <span class="legend-color target-point"></span>
                                    <span>Target</span>
                                </div>
                                <div class="legend-item">
                                    <span class="legend-color obstacle-point"></span>
                                    <span>Obstacle</span>
                                </div>
                            </div>
                            </div>
                        </div>
                    </div>
                {% endif %}
            </div>
            
            <div class="trajectory-stats">
                <h3>Statistics</h3>
                <ul>
                    {% if trajectory.metadata %}
                        {% if trajectory.metadata.success is defined %}
                            <li class="success-tag {% if trajectory.metadata.success %}success{% else %}failure{% endif %}">
                                Outcome: {% if trajectory.metadata.success %}Success{% else %}Failure{% endif %}
                            </li>
                        {% endif %}
                        
                        {% if trajectory.metadata.steps %}
                            <li>Steps: {{ trajectory.metadata.steps }}</li>
                        {% endif %}
                        
                        {% if trajectory.metadata.final_distance %}
                            <li>Final Distance to Target: {{ "%.2f"|format(trajectory.metadata.final_distance) }}m</li>
                        {% endif %}
                        
                        {% if trajectory.rewards %}
                            <li>Total Reward: {{ "%.2f"|format(sum(trajectory.rewards)) }}</li>
                        {% endif %}
                        
                        {% if trajectory.metadata.obstacles or trajectory.visualization_data.obstacles %}
                            <li class="obstacle-info">
                                Obstacles: {{ (trajectory.metadata.obstacles|length) if trajectory.metadata.obstacles 
                                               else (trajectory.visualization_data.obstacles|length) if trajectory.visualization_data.obstacles 
                                               else 0 }}
                            </li>
                        {% endif %}
                    {% endif %}
                </ul>
            </div>
        </div>
        
        <div class="feedback-section">
            <h3>Provide Feedback</h3>
            
            {% if feedback %}
                <div class="existing-feedback">
                    <h4>Existing Feedback</h4>
                    {% for f in feedback %}
                        <div class="feedback-item">
                            <p>Rating: <span class="rating">{{ f.rating }}/5</span></p>
                            {% if f.comment %}
                                <p>Comment: {{ f.comment }}</p>
                            {% endif %}
                            <p class="feedback-time">Submitted: {{ f.timestamp }}</p>
                        </div>
                    {% endfor %}
                </div>
            {% endif %}
            
            <form id="feedback-form" class="feedback-form">
                <input type="hidden" name="trajectory_id" value="{{ trajectory.id }}">
                
                <div class="rating-input">
                    <label>Rate this trajectory (1-5):</label>
                    <div class="star-rating">
                        <input type="radio" id="star5" name="rating" value="5" required>
                        <label for="star5" title="5 stars">★</label>
                        <input type="radio" id="star4" name="rating" value="4">
                        <label for="star4" title="4 stars">★</label>
                        <input type="radio" id="star3" name="rating" value="3">
                        <label for="star3" title="3 stars">★</label>
                        <input type="radio" id="star2" name="rating" value="2">
                        <label for="star2" title="2 stars">★</label>
                        <input type="radio" id="star1" name="rating" value="1">
                        <label for="star1" title="1 star">★</label>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="comment">Comments (optional):</label>
                    <textarea id="comment" name="comment" rows="4" placeholder="Enter any comments about this trajectory..."></textarea>
                </div>
                
                <button type="submit" class="submit-btn">Submit Feedback</button>
            </form>
            
            <div id="feedback-result" class="feedback-result hidden"></div>
        </div>
    </main>
    
    <footer>
        <p>TurtleBot3 Reinforcement Learning from Human Feedback (RLHF)</p>
    </footer>
    
    <script>
        // Function to render trajectory on canvas
        function renderTrajectoryCanvas() {
            const container = document.getElementById('trajectory-canvas-container');
            if (!container) return;
            
            let trajectoryData;
            try {
                trajectoryData = JSON.parse(container.getAttribute('data-trajectory'));
            } catch (e) {
                console.error("Error parsing trajectory data:", e);
                return;
            }
            
            const canvas = document.getElementById('trajectory-canvas');
            if (!canvas) return;
            
            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;
            const margin = 40;
            
            // Clear canvas
            ctx.clearRect(0, 0, width, height);
            
            // Extract trajectory positions
            const positions = trajectoryData.visualization_data && trajectoryData.visualization_data.positions 
                ? trajectoryData.visualization_data.positions 
                : [];
            
            // Get target position if available
            let targetPos = [0, 0];
            if (trajectoryData.visualization_data && trajectoryData.visualization_data.target_position) {
                targetPos = trajectoryData.visualization_data.target_position;
            } else if (trajectoryData.metadata && trajectoryData.metadata.target_position) {
                targetPos = trajectoryData.metadata.target_position;
            }
            
            // Get obstacles
            let obstacles = [];
            if (trajectoryData.visualization_data && trajectoryData.visualization_data.obstacles) {
                obstacles = trajectoryData.visualization_data.obstacles;
            } else if (trajectoryData.metadata && trajectoryData.metadata.obstacles) {
                obstacles = trajectoryData.metadata.obstacles;
            }
            
            // Find bounds of all elements
            let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
            
            // Include positions
            positions.forEach(pos => {
                if (Array.isArray(pos) && pos.length >= 2) {
                    xMin = Math.min(xMin, pos[0]);
                    xMax = Math.max(xMax, pos[0]);
                    yMin = Math.min(yMin, pos[1]);
                    yMax = Math.max(yMax, pos[1]);
                }
            });
            
            // Include target
            if (Array.isArray(targetPos) && targetPos.length >= 2) {
                xMin = Math.min(xMin, targetPos[0]);
                xMax = Math.max(xMax, targetPos[0]);
                yMin = Math.min(yMin, targetPos[1]);
                yMax = Math.max(yMax, targetPos[1]);
            }
            
            // Include obstacles
            obstacles.forEach(obs => {
                if (Array.isArray(obs)) {
                    // [x, y, radius] format
                    if (obs.length >= 3) {
                        xMin = Math.min(xMin, obs[0] - obs[2]);
                        xMax = Math.max(xMax, obs[0] + obs[2]);
                        yMin = Math.min(yMin, obs[1] - obs[2]);
                        yMax = Math.max(yMax, obs[1] + obs[2]);
                    } else if (obs.length >= 2) {
                        xMin = Math.min(xMin, obs[0]);
                        xMax = Math.max(xMax, obs[0]);
                        yMin = Math.min(yMin, obs[1]);
                        yMax = Math.max(yMax, obs[1]);
                    }
                } else if (typeof obs === 'object') {
                    if (obs.type === 'circle' && typeof obs.x === 'number' && typeof obs.y === 'number') {
                        const radius = obs.radius || 0.5;
                        xMin = Math.min(xMin, obs.x - radius);
                        xMax = Math.max(xMax, obs.x + radius);
                        yMin = Math.min(yMin, obs.y - radius);
                        yMax = Math.max(yMax, obs.y + radius);
                    } else if (obs.type === 'rectangle' && typeof obs.x === 'number' && typeof obs.y === 'number') {
                        const width = obs.width || 1;
                        const height = obs.height || 1;
                        xMin = Math.min(xMin, obs.x - width/2);
                        xMax = Math.max(xMax, obs.x + width/2);
                        yMin = Math.min(yMin, obs.y - height/2);
                        yMax = Math.max(yMax, obs.y + height/2);
                    } else if (obs.position && Array.isArray(obs.position)) {
                        if (obs.radius) {
                            xMin = Math.min(xMin, obs.position[0] - obs.radius);
                            xMax = Math.max(xMax, obs.position[0] + obs.radius);
                            yMin = Math.min(yMin, obs.position[1] - obs.radius);
                            yMax = Math.max(yMax, obs.position[1] + obs.radius);
                        } else if (obs.size || obs.dimensions) {
                            const size = obs.size || obs.dimensions;
                            xMin = Math.min(xMin, obs.position[0] - size[0]/2);
                            xMax = Math.max(xMax, obs.position[0] + size[0]/2);
                            yMin = Math.min(yMin, obs.position[1] - size[1]/2);
                            yMax = Math.max(yMax, obs.position[1] + size[1]/2);
                        }
                    }
                }
            });
            
            // Add padding
            const padding = Math.max((xMax - xMin) * 0.1, (yMax - yMin) * 0.1, 0.5);
            xMin -= padding;
            xMax += padding;
            yMin -= padding;
            yMax += padding;
            
            // If we still don't have valid bounds, set defaults
            if (!isFinite(xMin) || !isFinite(xMax) || !isFinite(yMin) || !isFinite(yMax)) {
                xMin = -5; xMax = 5; yMin = -5; yMax = 5;
            }
            
            // Create scale functions
            const xScale = d => (d - xMin) / (xMax - xMin) * (width - 2 * margin) + margin;
            const yScale = d => height - ((d - yMin) / (yMax - yMin) * (height - 2 * margin) + margin);
            
            // Draw background grid
            ctx.strokeStyle = '#eee';
            ctx.lineWidth = 1;
            
            // Draw grid lines
            const gridStep = Math.ceil(Math.max(xMax - xMin, yMax - yMin) / 10);
            
            for (let x = Math.floor(xMin / gridStep) * gridStep; x <= xMax; x += gridStep) {
                ctx.beginPath();
                ctx.moveTo(xScale(x), margin);
                ctx.lineTo(xScale(x), height - margin);
                ctx.stroke();
            }
            
            for (let y = Math.floor(yMin / gridStep) * gridStep; y <= yMax; y += gridStep) {
                ctx.beginPath();
                ctx.moveTo(margin, yScale(y));
                ctx.lineTo(width - margin, yScale(y));
                ctx.stroke();
            }
            
            // Draw obstacles
            ctx.fillStyle = 'rgba(128, 128, 128, 0.5)';
            ctx.strokeStyle = '#666';
            ctx.lineWidth = 1;
            
            obstacles.forEach(obs => {
                if (Array.isArray(obs)) {
                    // [x, y, radius] format
                    if (obs.length >= 3) {
                        ctx.beginPath();
                        ctx.arc(xScale(obs[0]), yScale(obs[1]), xScale(obs[0] + obs[2]) - xScale(obs[0]), 0, Math.PI * 2);
                        ctx.fill();
                        ctx.stroke();
                    } else if (obs.length >= 2) {
                        ctx.beginPath();
                        ctx.arc(xScale(obs[0]), yScale(obs[1]), 5, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.stroke();
                    }
                } else if (typeof obs === 'object') {
                    if (obs.type === 'circle' && typeof obs.x === 'number' && typeof obs.y === 'number') {
                        const radius = obs.radius || 0.5;
                        ctx.beginPath();
                        ctx.arc(xScale(obs.x), yScale(obs.y), xScale(obs.x + radius) - xScale(obs.x), 0, Math.PI * 2);
                        ctx.fill();
                        ctx.stroke();
                    } else if (obs.type === 'rectangle' && typeof obs.x === 'number' && typeof obs.y === 'number') {
                        const width = obs.width || 1;
                        const height = obs.height || 1;
                        const rotation = obs.rotation || 0;
                        
                        ctx.save();
                        ctx.translate(xScale(obs.x), yScale(obs.y));
                        ctx.rotate(-rotation); // Negative because canvas Y is flipped
                        
                        const scaledWidth = xScale(obs.x + width/2) - xScale(obs.x - width/2);
                        const scaledHeight = yScale(obs.y - height/2) - yScale(obs.y + height/2);
                        
                        ctx.beginPath();
                        ctx.rect(-scaledWidth/2, -scaledHeight/2, scaledWidth, scaledHeight);
                        ctx.fill();
                        ctx.stroke();
                        
                        ctx.restore();
                    } else if (obs.position && Array.isArray(obs.position)) {
                        if (obs.radius) {
                            ctx.beginPath();
                            ctx.arc(
                                xScale(obs.position[0]), 
                                yScale(obs.position[1]), 
                                xScale(obs.position[0] + obs.radius) - xScale(obs.position[0]), 
                                0, Math.PI * 2
                            );
                            ctx.fill();
                            ctx.stroke();
                        } else if (obs.size || obs.dimensions) {
                            const size = obs.size || obs.dimensions;
                            ctx.beginPath();
                            ctx.rect(
                                xScale(obs.position[0] - size[0]/2),
                                yScale(obs.position[1] + size[1]/2),
                                xScale(obs.position[0] + size[0]/2) - xScale(obs.position[0] - size[0]/2),
                                yScale(obs.position[1] - size[1]/2) - yScale(obs.position[1] + size[1]/2)
                            );
                            ctx.fill();
                            ctx.stroke();
                        }
                    }
                }
            });
            
            // Draw trajectory path
            ctx.strokeStyle = '#1976D2';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            for (let i = 0; i < positions.length; i++) {
                const [x, y] = positions[i];
                if (i === 0) {
                    ctx.moveTo(xScale(x), yScale(y));
                } else {
                    ctx.lineTo(xScale(x), yScale(y));
                }
            }
            
            ctx.stroke();
            
            // Draw starting point
            if (positions.length > 0) {
                const [x, y] = positions[0];
                ctx.fillStyle = 'green';
                ctx.beginPath();
                ctx.arc(xScale(x), yScale(y), 6, 0, Math.PI * 2);
                ctx.fill();
            }
            
            // Draw ending point
            if (positions.length > 0) {
                const [x, y] = positions[positions.length - 1];
                const success = trajectoryData.metadata && trajectoryData.metadata.success;
                ctx.fillStyle = success ? '#4CAF50' : '#F44336';
                ctx.beginPath();
                ctx.arc(xScale(x), yScale(y), 6, 0, Math.PI * 2);
                ctx.fill();
            }
            
            // Draw target
            if (Array.isArray(targetPos) && targetPos.length >= 2) {
                ctx.strokeStyle = '#FF9800';
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);
                ctx.beginPath();
                ctx.arc(xScale(targetPos[0]), yScale(targetPos[1]), 8, 0, Math.PI * 2);
                ctx.stroke();
                ctx.setLineDash([]);
            }
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            renderTrajectoryCanvas();
            
            // Handle form submission
            document.getElementById('feedback-form').addEventListener('submit', function(event) {
                event.preventDefault();
                
                const formData = new FormData(this);
                
                fetch('/submit_feedback', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    const resultDiv = document.getElementById('feedback-result');
                    resultDiv.classList.remove('hidden');
                    
                    if (data.success) {
                        resultDiv.classList.add('success');
                        resultDiv.classList.remove('error');
                        resultDiv.innerHTML = `${data.message}. Total feedback: ${data.feedback_count}`;
                        
                        // Reset form
                        document.getElementById('feedback-form').reset();
                        
                        // Refresh page after a short delay
                        setTimeout(() => {
                            location.reload();
                        }, 2000);
                    } else {
                        resultDiv.classList.add('error');
                        resultDiv.classList.remove('success');
                        resultDiv.textContent = 'Error: ' + data.message;
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    const resultDiv = document.getElementById('feedback-result');
                    resultDiv.classList.remove('hidden');
                    resultDiv.classList.add('error');
                    resultDiv.classList.remove('success');
                    resultDiv.textContent = 'Error submitting feedback. Please try again.';
                });
            });
        });
    </script>
</body>
</html>
EOF

echo "Creating compare.html template..."
cat > $TEMPLATE_DIR/compare.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>TurtleBot3 Trajectory Comparison</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .container {
            display: flex;
            flex-direction: row;
            gap: 20px;
            margin-bottom: 20px;
        }
        .trajectory {
            flex: 1;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
        }
        .trajectory h2 {
            margin-top: 0;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        .metadata {
            margin: 15px 0;
            font-size: 14px;
        }
        .metadata-item {
            margin-bottom: 5px;
        }
        .controls {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            margin: 25px 0;
        }
        button {
            padding: 12px 25px;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        #prefA {
            background-color: #4CAF50;
            color: white;
        }
        #prefB {
            background-color: #2196F3;
            color: white;
        }
        #similar {
            background-color: #9E9E9E;
            color: white;
        }
        button:hover {
            opacity: 0.9;
        }
        .feedback {
            max-width: 600px;
            margin: 0 auto;
        }
        textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            resize: vertical;
        }
        .loading {
            text-align: center;
            padding: 20px;
            font-style: italic;
            color: #666;
        }
        .status {
            text-align: center;
            margin-top: 15px;
            padding: 10px;
            border-radius: 4px;
        }
        .success {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
        .error {
            background-color: #ffebee;
            color: #c62828;
        }
        svg {
            display: block;
            margin: 0 auto;
            background-color: #f9f9f9;
            border: 1px solid #eee;
            border-radius: 4px;
        }
        .obstacle-circle {
            fill: rgba(128, 128, 128, 0.5);
            stroke: #666;
            stroke-width: 1;
        }
        .obstacle-rect {
            fill: rgba(128, 128, 128, 0.5);
            stroke: #666;
            stroke-width: 1;
        }
        .legend {
            font-size: 12px;
        }
    </style>
</head>
<body>
    <header>
        <h1>Which TurtleBot3 trajectory is better?</h1>
        <a href="{{ url_for('index') }}" class="back-btn">← Back to List</a>
    </header>
    
    <div id="loading" class="loading">Loading trajectories...</div>
    
    <div id="content" style="display: none;">
        <div class="container">
            <div class="trajectory" id="traj1">
                <h2>Trajectory A</h2>
                <div id="vis1"></div>
                <div id="metadata1" class="metadata"></div>
            </div>
            
            <div class="trajectory" id="traj2">
                <h2>Trajectory B</h2>
                <div id="vis2"></div>
                <div id="metadata2" class="metadata"></div>
            </div>
        </div>
        
        <div class="controls">
            <button id="prefA">Prefer A</button>
            <button id="prefB">Prefer B</button>
            <button id="similar">Similar/Can't Decide</button>
        </div>
        
        <div class="feedback">
            <h3>Feedback (optional):</h3>
            <textarea id="feedbackText" rows="4" placeholder="Why did you prefer this trajectory? What made it better?"></textarea>
        </div>
        
        <div id="status" class="status" style="display: none;"></div>
    </div>
    
    <footer>
        <p>TurtleBot3 Reinforcement Learning from Human Feedback (RLHF)</p>
    </footer>
    
    <script>
        let currentPair = null;
        
        // Fetch a pair of trajectories to compare
        function fetchTrajectoryPair() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('content').style.display = 'none';
            document.getElementById('status').style.display = 'none';
            
            fetch('/api/get_comparison_pair')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Failed to fetch trajectory pair');
                    }
                    return response.json();
                })
                .then(data => {
                    currentPair = data;
                    console.log("Received trajectory data:", data);
                    renderTrajectories(data);
                    
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('content').style.display = 'block';
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('loading').textContent = 
                        'Error loading trajectories: ' + error.message;
                });
        }
        
        // Render the trajectories
        function renderTrajectories(data) {
            // Clear previous visualizations
            document.getElementById('vis1').innerHTML = '';
            document.getElementById('vis2').innerHTML = '';
            
            console.log("Visualizing trajectory 1 obstacles:", data.trajectory1.visualization_data.obstacles);
            console.log("Visualizing trajectory 2 obstacles:", data.trajectory2.visualization_data.obstacles);
            
            // Visualization for Trajectory 1
            visualizeTrajectory('vis1', data.trajectory1.visualization_data);
            displayMetadata('metadata1', data.trajectory1.metadata);
            
            // Visualization for Trajectory 2
            visualizeTrajectory('vis2', data.trajectory2.visualization_data);
            displayMetadata('metadata2', data.trajectory2.metadata);
        }
                
        function visualizeTrajectory(containerId, data) {
            const width = 300;
            const height = 300;
            const margin = 30;
            
            // Extract data - ensure we have valid arrays for all properties
            const positions = Array.isArray(data.positions) ? data.positions : [];
            const target = Array.isArray(data.target_position) ? data.target_position : [0, 0];
            
            // Process obstacles ensuring they're in a standard format
            let obstacles = [];
            if (Array.isArray(data.obstacles)) {
                obstacles = data.obstacles;
            } else if (data.obstacles && typeof data.obstacles === 'object') {
                obstacles = [data.obstacles]; // Single obstacle as object
            }
            
            console.log(`${containerId} obstacles:`, obstacles);
            
            // Find the bounds of the data
            let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
            
            // Include positions in bounds calculation
            positions.forEach(pos => {
                if (Array.isArray(pos) && pos.length >= 2) {
                    xMin = Math.min(xMin, pos[0]);
                    xMax = Math.max(xMax, pos[0]);
                    yMin = Math.min(yMin, pos[1]);
                    yMax = Math.max(yMax, pos[1]);
                }
            });
            
            // Include target in bounds calculation
            if (Array.isArray(target) && target.length >= 2) {
                xMin = Math.min(xMin, target[0]);
                xMax = Math.max(xMax, target[0]);
                yMin = Math.min(yMin, target[1]);
                yMax = Math.max(yMax, target[1]);
            }
            
            // Ensure we have valid bounds before processing obstacles
            if (xMin === Infinity) xMin = -5;
            if (xMax === -Infinity) xMax = 5;
            if (yMin === Infinity) yMin = -5;
            if (yMax === -Infinity) yMax = 5;
            
            // Include obstacles in bounds calculation with robust handling
            obstacles.forEach(obs => {
                try {
                    // Handle array format obstacles
                    if (Array.isArray(obs)) {
                        if (obs.length >= 3) {
                            // [x, y, radius] format
                            xMin = Math.min(xMin, obs[0] - obs[2]);
                            xMax = Math.max(xMax, obs[0] + obs[2]);
                            yMin = Math.min(yMin, obs[1] - obs[2]);
                            yMax = Math.max(yMax, obs[1] + obs[2]);
                        } else if (obs.length >= 2) {
                            // [x, y] point format
                            xMin = Math.min(xMin, obs[0]);
                            xMax = Math.max(xMax, obs[0]);
                            yMin = Math.min(yMin, obs[1]);
                            yMax = Math.max(yMax, obs[1]);
                        }
                    } 
                    // Handle object format obstacles
                    else if (obs && typeof obs === 'object') {
                        if (obs.type === 'circle' && typeof obs.x === 'number' && typeof obs.y === 'number' && typeof obs.radius === 'number') {
                            xMin = Math.min(xMin, obs.x - obs.radius);
                            xMax = Math.max(xMax, obs.x + obs.radius);
                            yMin = Math.min(yMin, obs.y - obs.radius);
                            yMax = Math.max(yMax, obs.y + obs.radius);
                        } else if (obs.type === 'rectangle' && typeof obs.x === 'number' && typeof obs.y === 'number') {
                            const width = obs.width || 1;
                            const height = obs.height || 1;
                            xMin = Math.min(xMin, obs.x - width/2);
                            xMax = Math.max(xMax, obs.x + width/2);
                            yMin = Math.min(yMin, obs.y - height/2);
                            yMax = Math.max(yMax, obs.y + height/2);
                        } else if (obs.type === 'point' && typeof obs.x === 'number' && typeof obs.y === 'number') {
                            xMin = Math.min(xMin, obs.x);
                            xMax = Math.max(xMax, obs.x);
                            yMin = Math.min(yMin, obs.y);
                            yMax = Math.max(yMax, obs.y);
                        } else if (obs.position && Array.isArray(obs.position) && obs.position.length >= 2) {
                            if (typeof obs.radius === 'number') {
                                // Circle with position and radius
                                xMin = Math.min(xMin, obs.position[0] - obs.radius);
                                xMax = Math.max(xMax, obs.position[0] + obs.radius);
                                yMin = Math.min(yMin, obs.position[1] - obs.radius);
                                yMax = Math.max(yMax, obs.position[1] + obs.radius);
                            } else if (obs.size && Array.isArray(obs.size) && obs.size.length >= 2) {
                                // Rectangle with position and size
                                xMin = Math.min(xMin, obs.position[0] - obs.size[0]/2);
                                xMax = Math.max(xMax, obs.position[0] + obs.size[0]/2);
                                yMin = Math.min(yMin, obs.position[1] - obs.size[1]/2);
                                yMax = Math.max(yMax, obs.position[1] + obs.size[1]/2);
                            } else if (obs.dimensions && Array.isArray(obs.dimensions) && obs.dimensions.length >= 2) {
                                // Rectangle with position and dimensions
                                xMin = Math.min(xMin, obs.position[0] - obs.dimensions[0]/2);
                                xMax = Math.max(xMax, obs.position[0] + obs.dimensions[0]/2);
                                yMin = Math.min(yMin, obs.position[1] - obs.dimensions[1]/2);
                                yMax = Math.max(yMax, obs.position[1] + obs.dimensions[1]/2);
                            } else {
                                // Just a point
                                xMin = Math.min(xMin, obs.position[0]);
                                xMax = Math.max(xMax, obs.position[0]);
                                yMin = Math.min(yMin, obs.position[1]);
                                yMax = Math.max(yMax, obs.position[1]);
                            }
                        }
                    }
                } catch (e) {
                    console.error("Error processing obstacle:", obs, e);
                }
            });
            
            // Add some padding
            const padding = Math.max((xMax - xMin) * 0.1, (yMax - yMin) * 0.1, 1);
            xMin -= padding;
            xMax += padding;
            yMin -= padding;
            yMax += padding;
            
            // Create scales
            const xScale = d3.scaleLinear()
                .domain([xMin, xMax])
                .range([margin, width - margin]);
            
            const yScale = d3.scaleLinear()
                .domain([yMin, yMax])
                .range([height - margin, margin]);
            
            // Create SVG
            const svg = d3.select(`#${containerId}`)
                .append('svg')
                .attr('width', width)
                .attr('height', height);
            
            // Draw obstacles first (so they're behind the path)
            obstacles.forEach(obstacle => {
                try {
                    if (Array.isArray(obstacle)) {
                        // Handle array format obstacles
                        if (obstacle.length >= 3) {
                            // [x, y, radius] format
                            svg.append('circle')
                                .attr('cx', xScale(obstacle[0]))
                                .attr('cy', yScale(obstacle[1]))
                                .attr('r', xScale(obstacle[0] + obstacle[2]) - xScale(obstacle[0]))
                                .attr('class', 'obstacle-circle');
                        } else if (obstacle.length >= 2) {
                            // [x, y] point format
                            svg.append('circle')
                                .attr('cx', xScale(obstacle[0]))
                                .attr('cy', yScale(obstacle[1]))
                                .attr('r', 5)
                                .attr('class', 'obstacle-circle');
                        }
                    } else if (obstacle && typeof obstacle === 'object') {
                        if (obstacle.type === 'circle' && typeof obstacle.x === 'number' && typeof obstacle.y === 'number') {
                            const radius = obstacle.radius || 0.5;
                            svg.append('circle')
                                .attr('cx', xScale(obstacle.x))
                                .attr('cy', yScale(obstacle.y))
                                .attr('r', xScale(obstacle.x + radius) - xScale(obstacle.x))
                                .attr('class', 'obstacle-circle');
                        } else if (obstacle.type === 'rectangle' && typeof obstacle.x === 'number' && typeof obstacle.y === 'number') {
                            const width = obstacle.width || 1;
                            const height = obstacle.height || 1;
                            svg.append('rect')
                                .attr('x', xScale(obstacle.x - width/2))
                                .attr('y', yScale(obstacle.y + height/2))
                                .attr('width', xScale(obstacle.x + width/2) - xScale(obstacle.x - width/2))
                                .attr('height', yScale(obstacle.y - height/2) - yScale(obstacle.y + height/2))
                                .attr('class', 'obstacle-rect');
                        } else if (obstacle.type === 'point' && typeof obstacle.x === 'number' && typeof obstacle.y === 'number') {
                            svg.append('circle')
                                .attr('cx', xScale(obstacle.x))
                                .attr('cy', yScale(obstacle.y))
                                .attr('r', 5)
                                .attr('class', 'obstacle-circle');
                        } else if (obstacle.position && Array.isArray(obstacle.position) && obstacle.position.length >= 2) {
                            if (typeof obstacle.radius === 'number') {
                                // Circle with position and radius
                                svg.append('circle')
                                    .attr('cx', xScale(obstacle.position[0]))
                                    .attr('cy', yScale(obstacle.position[1]))
                                    .attr('r', xScale(obstacle.position[0] + obstacle.radius) - xScale(obstacle.position[0]))
                                    .attr('class', 'obstacle-circle');
                            } else if (obstacle.size && Array.isArray(obstacle.size) && obstacle.size.length >= 2) {
                                // Rectangle with position and size
                                svg.append('rect')
                                    .attr('x', xScale(obstacle.position[0] - obstacle.size[0]/2))
                                    .attr('y', yScale(obstacle.position[1] + obstacle.size[1]/2))
                                    .attr('width', xScale(obstacle.position[0] + obstacle.size[0]/2) - xScale(obstacle.position[0] - obstacle.size[0]/2))
                                    .attr('height', yScale(obstacle.position[1] - obstacle.size[1]/2) - yScale(obstacle.position[1] + obstacle.size[1]/2))
                                    .attr('class', 'obstacle-rect');
                            } else if (obstacle.dimensions && Array.isArray(obstacle.dimensions) && obstacle.dimensions.length >= 2) {
                                // Rectangle with position and dimensions
                                svg.append('rect')
                                    .attr('x', xScale(obstacle.position[0] - obstacle.dimensions[0]/2))
                                    .attr('y', yScale(obstacle.position[1] + obstacle.dimensions[1]/2))
                                    .attr('width', xScale(obstacle.position[0] + obstacle.dimensions[0]/2) - xScale(obstacle.position[0] - obstacle.dimensions[0]/2))
                                    .attr('height', yScale(obstacle.position[1] - obstacle.dimensions[1]/2) - yScale(obstacle.position[1] + obstacle.dimensions[1]/2))
                                    .attr('class', 'obstacle-rect');
                            } else {
                                // Just a point
                                svg.append('circle')
                                    .attr('cx', xScale(obstacle.position[0]))
                                    .attr('cy', yScale(obstacle.position[1]))
                                    .attr('r', 5)
                                    .attr('class', 'obstacle-circle');
                            }
                        }
                    }
                } catch (e) {
                    console.error("Error rendering obstacle:", obstacle, e);
                }
            });
            
            // Create a line generator
            const line = d3.line()
                .x(d => xScale(d[0]))
                .y(d => yScale(d[1]))
                .curve(d3.curveBasis);
            
            // Draw the path
            if (positions.length > 1) {
                svg.append('path')
                    .datum(positions)
                    .attr('fill', 'none')
                    .attr('stroke', '#1976D2')
                    .attr('stroke-width', 2)
                    .attr('d', line);
            }
            
            // Draw the starting point
            if (positions.length > 0) {
                svg.append('circle')
                    .attr('cx', xScale(positions[0][0]))
                    .attr('cy', yScale(positions[0][1]))
                    .attr('r', 6)
                    .attr('fill', 'green');
            }
            
            // Draw the ending point
            if (positions.length > 0) {
                svg.append('circle')
                    .attr('cx', xScale(positions[positions.length - 1][0]))
                    .attr('cy', yScale(positions[positions.length - 1][1]))
                    .attr('r', 6)
                    .attr('fill', data.success ? '#4CAF50' : '#F44336');
            }
            
            // Draw the target
            svg.append('circle')
                .attr('cx', xScale(target[0]))
                .attr('cy', yScale(target[1]))
                .attr('r', 8)
                .attr('fill', 'none')
                .attr('stroke', '#FF9800')
                .attr('stroke-width', 2)
                .attr('stroke-dasharray', '3,3');
            
            // Add coordinate axes
            // X-axis
            svg.append('line')
                .attr('x1', margin)
                .attr('y1', height/2)
                .attr('x2', width - margin)
                .attr('y2', height/2)
                .attr('stroke', '#ccc')
                .attr('stroke-width', 1);
            
            // Y-axis
            svg.append('line')
                .attr('x1', width/2)
                .attr('y1', margin)
                .attr('x2', width/2)
                .attr('y2', height - margin)
                .attr('stroke', '#ccc')
                .attr('stroke-width', 1);
            
            // Add legend
            const legend = svg.append('g')
                .attr('class', 'legend')
                .attr('transform', `translate(${width - margin - 80}, ${margin})`);
            
            // Starting point
            legend.append('circle')
                .attr('cx', 0)
                .attr('cy', 0)
                .attr('r', 4)
                .attr('fill', 'green');
            
            legend.append('text')
                .attr('x', 10)
                .attr('y', 3)
                .text('Start');
            
            // Ending point
            legend.append('circle')
                .attr('cx', 0)
                .attr('cy', 20)
                .attr('r', 4)
                .attr('fill', data.success ? '#4CAF50' : '#F44336');
            
            legend.append('text')
                .attr('x', 10)
                .attr('y', 23)
                .text('End');
            
            // Target
            legend.append('circle')
                .attr('cx', 0)
                .attr('cy', 40)
                .attr('r', 4)
                .attr('fill', 'none')
                .attr('stroke', '#FF9800')
                .attr('stroke-width', 2)
                .attr('stroke-dasharray', '2,2');
            
            legend.append('text')
                .attr('x', 10)
                .attr('y', 43)
                .text('Target');
            
            // Obstacle (if any)
            if (obstacles && obstacles.length > 0) {
                legend.append('rect')
                    .attr('x', -4)
                    .attr('y', 56)
                    .attr('width', 8)
                    .attr('height', 8)
                    .attr('class', 'obstacle-rect');
                
                legend.append('text')
                    .attr('x', 10)
                    .attr('y', 63)
                    .text('Obstacle');
            }
        }

        // Display metadata
        function displayMetadata(containerId, metadata) {
            const container = document.getElementById(containerId);
            if (!container) return;
            container.innerHTML = '';
            
            // Format and display metadata
            const items = [
                { label: 'Success', value: metadata.success ? 'Yes' : 'No' },
                { label: 'Steps', value: metadata.steps },
                { label: 'Final Distance to Target', value: metadata.final_distance ? metadata.final_distance.toFixed(2) + 'm' : 'N/A' },
                { label: 'Obstacles', value: metadata.obstacle_count || '0' }
            ];
            
            items.forEach(item => {
                const div = document.createElement('div');
                div.className = 'metadata-item';
                div.innerHTML = `<strong>${item.label}:</strong> ${item.value}`;
                container.appendChild(div);
            });
        }
        
        // Submit preference
        function submitPreference(preferred) {
            if (!currentPair) return;
            
            const data = {
                preferred: preferred === 'A' ? currentPair.trajectory1.id : 
                           preferred === 'B' ? currentPair.trajectory2.id : 'similar',
                rejected: preferred === 'A' ? currentPair.trajectory2.id : 
                          preferred === 'B' ? currentPair.trajectory1.id : 'similar',
                reason: document.getElementById('feedbackText').value,
                confidence: 1.0
            };
            
            fetch('/api/submit_preference', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to submit preference');
                }
                return response.json();
            })
            .then(result => {
                const status = document.getElementById('status');
                status.textContent = 'Preference submitted successfully! Loading new pair...';
                status.className = 'status success';
                status.style.display = 'block';
                
                // Clear feedback textarea
                document.getElementById('feedbackText').value = '';
                
                // Load a new pair after a short delay
                setTimeout(fetchTrajectoryPair, 1500);
            })
            .catch(error => {
                console.error('Error:', error);
                const status = document.getElementById('status');
                status.textContent = 'Error submitting preference: ' + error.message;
                status.className = 'status error';
                status.style.display = 'block';
            });
        }
        
        // Add event listeners to buttons
        document.getElementById('prefA').addEventListener('click', () => submitPreference('A'));
        document.getElementById('prefB').addEventListener('click', () => submitPreference('B'));
        document.getElementById('similar').addEventListener('click', () => submitPreference('similar'));
        
        // Load the first pair of trajectories when the page loads
        document.addEventListener('DOMContentLoaded', fetchTrajectoryPair);
    </script>
</body>
</html>
EOF

echo "Creating index.html template..."
cat > $TEMPLATE_DIR/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TurtleBot3 RLHF Feedback</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <header>
        <h1>TurtleBot3 Trajectory Feedback</h1>
        <div class="stats">
            <p>Total Feedback: <span id="feedback-count">{{ feedback_count }}</span></p>
            <p>Trajectories Rated: <span id="rated-count">{{ rated_count }}</span> / {{ total_count }}</p>
            <button id="refresh-btn">Refresh Trajectories</button>
        </div>
    </header>
    
    <main>
        <div class="trajectories">
            <h2>Available Trajectories</h2>
            
            <div class="trajectory-actions">
                <a href="{{ url_for('compare') }}" class="action-btn">Compare Trajectories</a>
            </div>
            
            {% if trajectories %}
                <div class="trajectory-list">
                    {% for trajectory in trajectories %}
                        <div class="trajectory-card">
                            <h3>Trajectory: {{ trajectory.filename or trajectory.id }}</h3>
                            
                            {% if trajectory.plot_path %}
                                <div class="trajectory-preview">
                                    <img src="{{ url_for('static', filename=trajectory.plot_path) }}" alt="Trajectory plot">
                                </div>
                            {% endif %}
                            
                            <div class="trajectory-meta">
                                {% if trajectory.metadata %}
                                    {% if trajectory.metadata.success is defined %}
                                        <p class="success-tag {% if trajectory.metadata.success %}success{% else %}failure{% endif %}">
                                            {% if trajectory.metadata.success %}Success{% else %}Failure{% endif %}
                                        </p>
                                    {% endif %}
                                    
                                    {% if trajectory.metadata.steps %}
                                        <p>Steps: {{ trajectory.metadata.steps }}</p>
                                    {% endif %}
                                    
                                    {% if trajectory.metadata.final_distance %}
                                        <p>Final Distance: {{ "%.2f"|format(trajectory.metadata.final_distance) }}m</p>
                                    {% endif %}
                                {% endif %}
                            </div>
                            
                            <a href="{{ url_for('view_trajectory', trajectory_id=trajectory.id) }}" class="view-btn">View & Rate</a>
                        </div>
                    {% endfor %}
                </div>
            {% else %}
                <p class="no-trajectories">No trajectories available. Please collect some trajectories first.</p>
            {% endif %}
        </div>
    </main>
    
    <footer>
        <p>TurtleBot3 Reinforcement Learning from Human Feedback (RLHF)</p>
    </footer>
    
    <script>
        document.getElementById('refresh-btn').addEventListener('click', function() {
            fetch('/refresh', {
                method: 'POST',
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('feedback-count').textContent = data.feedback_count;
                    document.getElementById('rated-count').textContent = data.rated_count;
                    location.reload();
                } else {
                    alert('Error refreshing trajectories: ' + data.message);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error refreshing trajectories');
            });
        });
    </script>
</body>
</html>
EOF

echo "Creating style.css..."
cat > $CSS_DIR/style.css << 'EOF'
/* Base Styles */
:root {
    --primary-color: #3498db;
    --secondary-color: #2980b9;
    --success-color: #27ae60;
    --danger-color: #e74c3c;
    --light-color: #ecf0f1;
    --dark-color: #2c3e50;
    --gray-color: #95a5a6;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f8f9fa;
    padding-bottom: 60px;
}

h1, h2, h3, h4 {
    margin-bottom: 0.5rem;
    color: var(--dark-color);
}

a {
    text-decoration: none;
    color: var(--primary-color);
}

button {
    cursor: pointer;
}

/* Layout */
header {
    background-color: var(--primary-color);
    color: white;
    padding: 1rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

header h1 {
    color: white;
}

.stats {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 1.5rem;
    margin-top: 0.5rem;
}

.stats p {
    margin: 0;
    font-size: 0.9rem;
}

#feedback-count, #rated-count {
    font-weight: bold;
}

main {
    width: 90%;
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem;
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

footer {
    text-align: center;
    margin-top: 2rem;
    padding: 1rem;
    background-color: var(--dark-color);
    color: white;
    position: fixed;
    bottom: 0;
    width: 100%;
}

/* Trajectory List */
.trajectory-list {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1.5rem;
    margin-top: 1rem;
}

.trajectory-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s, box-shadow 0.2s;
}

.trajectory-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
}

.trajectory-preview img {
    width: 100%;
    height: 200px;
    object-fit: contain;
    border-radius: 4px;
    margin: 0.5rem 0;
}

.trajectory-meta {
    margin: 1rem 0;
    font-size: 0.9rem;
}

.success-tag {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-weight: bold;
    margin-bottom: 0.5rem;
}

.success-tag.success {
    background-color: rgba(39, 174, 96, 0.2);
    color: var(--success-color);
    border: 1px solid var(--success-color);
}

.success-tag.failure {
    background-color: rgba(231, 76, 60, 0.2);
    color: var(--danger-color);
    border: 1px solid var(--danger-color);
}

.view-btn, .action-btn {
    display: block;
    width: 100%;
    padding: 0.5rem;
    background-color: var(--primary-color);
    color: white;
    text-align: center;
    border-radius: 4px;
    transition: background-color 0.2s;
}

.view-btn:hover, .action-btn:hover {
    background-color: var(--secondary-color);
}

.trajectory-actions {
    margin-bottom: 1.5rem;
}

.action-btn {
    display: inline-block;
    width: auto;
    padding: 0.5rem 1rem;
    margin-right: 1rem;
}

/* Buttons */
#refresh-btn, .back-btn, .submit-btn {
    background-color: var(--primary-color);
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    font-size: 0.9rem;
    transition: background-color 0.2s;
}

#refresh-btn:hover, .back-btn:hover, .submit-btn:hover {
    background-color: var(--secondary-color);
}

.back-btn {
    margin-top: 0.5rem;
    display: inline-block;
}

/* Trajectory Detail Page */
.trajectory-detail {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
}

.full-size-plot {
    width: 100%;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.trajectory-stats ul {
    list-style: none;
    margin: 1rem 0;
}

.trajectory-stats li {
    margin-bottom: 0.5rem;
    padding: 0.5rem;
    background-color: #f8f9fa;
    border-radius: 4px;
}

/* Feedback Section */
.feedback-section {
    padding: 1rem;
    background-color: #f8f9fa;
    border-radius: 8px;
}

.existing-feedback {
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #e0e0e0;
}

.feedback-item {
    background-color: white;
    padding: 1rem;
    margin-bottom: 1rem;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.rating {
    font-weight: bold;
    color: var(--primary-color);
}

.feedback-time {
    font-size: 0.8rem;
    color: var(--gray-color);
    margin-top: 0.5rem;
}

/* Feedback Form */
.feedback-form {
    margin-top: 1rem;
}

.form-group {
    margin-bottom: 1rem;
}

label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: bold;
}

textarea {
    width: 100%;
    padding: 0.5rem;
    border: 1px solid #ddd;
    border-radius: 4px;
    resize: vertical;
}

/* Star Rating */
.rating-input {
    margin-bottom: 1rem;
}

.star-rating {
    direction: rtl;
    display: inline-block;
    margin-top: 0.5rem;
}

.star-rating input {
    display: none;
}

.star-rating label {
    display: inline-block;
    cursor: pointer;
    font-size: 30px;
    color: #ccc;
    transition: color 0.2s;
}

.star-rating label:hover,
.star-rating label:hover ~ label,
.star-rating input:checked ~ label {
    color: #ffcc00;
}

/* Feedback Result */
.feedback-result {
    margin-top: 1rem;
    padding: 1rem;
    border-radius: 4px;
    text-align: center;
}

.feedback-result.success {
    background-color: rgba(39, 174, 96, 0.2);
    color: var(--success-color);
}

.feedback-result.error {
    background-color: rgba(231, 76, 60, 0.2);
    color: var(--danger-color);
}

.hidden {
    display: none;
}

/* Mobile Responsive */
@media (max-width: 768px) {
    .stats {
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .trajectory-detail {
        grid-template-columns: 1fr;
    }
    
    .trajectory-list {
        grid-template-columns: 1fr;
    }
}
EOF

echo "Templates and CSS files have been set up successfully!"
echo "Now making directories executable..."

# Make the directory accessible
chmod -R 755 $TEMPLATE_DIR
chmod -R 755 $STATIC_DIR

echo "Setup complete!"