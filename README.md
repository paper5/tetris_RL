# Tetris Reinforcement Learning Agent

This project implements a reinforcement learning agent to play Tetris using the Stable-Baselines3 library. The agent is trained using the Deep Q-Network (DQN) algorithm and can record gameplay videos for evaluation.

## Features
- **Training**: The agent is trained using DQN with custom hyperparameters.
- **Evaluation**: The agent's performance is evaluated periodically during training.
- **Video Recording**: Gameplay videos are recorded to visually assess the agent's performance.

## Requirements
- Python 3.8+
- Libraries:
  - `gymnasium`
  - `stable-baselines3`
  - `opencv-python`
  - `numpy`

## Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd tetris_RL
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### Train the Agent
Run the `main.py` script to train the agent:
```bash
python main.py
```
This will train the agent using DQN and save the best model in the `best_model/` directory.
- Check version history for PPO usage
### Record Gameplay Videos
Run the `record_video.py` script to record gameplay videos:
```bash
python record_video.py
```
Videos will be saved in the `videos/` directory.

## Further Improvements
1. Add more sophisticated reward shaping to improve learning efficiency.
2. Change reward function to fix rotation issue (the model does not rotate blocks often
3. Implement other advanced algorithms like Proximal Policy Optimization (PPO) or A2C.
## Notes
- The agent's current performance may be limited; further tuning and experimentation are recommended.
- Ensure the environment is correctly installed (`tetris_gymnasium`) before running the scripts.

Happy coding!
