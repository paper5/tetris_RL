from tetris_gymnasium.envs.tetris import TetrisGymnasium, RewardsMapping
from dataclasses import dataclass

class EnhancedTetrisEnv(TetrisGymnasium):
    def __init__(self, **kwargs):
        super().__init__(
            reward_mapping=CustomRewardMapping(),
            **kwargs
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Get game state metrics
        board = self.engine.board
        lines = info.get("lines_cleared", 0)

        # Custom reward calculation
        custom_reward = 0
        custom_reward += self.reward_mapping.ALIVE
        custom_reward += lines * self.reward_mapping.LINE_CLEAR

        if lines == 4:
            custom_reward += self.reward_mapping.TETRIS

        # Add penalties
        custom_reward += self._count_holes() * self.reward_mapping.HOLE_PENALTY
        custom_reward += self._get_height() * self.reward_mapping.HEIGHT_PENALTY
        custom_reward += self._get_bumpiness() * self.reward_mapping.BUMPINESS

        if terminated:
            custom_reward += self.reward_mapping.GAME_OVER

        return obs, custom_reward, terminated, truncated, info

    def _count_holes(self):
        """Count covered empty cells"""
        board = self.engine.board
        return sum(
            any(col[:row]) and not col[row]
            for col in board.T
            for row in range(len(col))

    def _get_height(self):
        """Maximum column height"""
        return max(sum(col) for col in self.engine.board.T)

    def _get_bumpiness(self):
        """Difference between adjacent columns"""
        heights = [sum(col) for col in self.engine.board.T]
        return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))


@dataclass
class CustomRewardMapping(RewardsMapping):
    ALIVE: int = 0.01          # Small positive reward per step
    LINE_CLEAR: int = 1.0      # Base reward per line
    TETRIS: int = 4.0          # Bonus for clearing 4 lines
    HOLE_PENALTY: int = -0.3   # Penalty for creating holes
    HEIGHT_PENALTY: int = -0.1 # Penalty for stack height
    BUMPINESS: int = -0.05     # Penalty for uneven surface
    GAME_OVER: int = -5.0      # Large penalty for losing