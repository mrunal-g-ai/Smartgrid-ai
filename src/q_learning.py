import numpy as np

# States and Actions
STATES = ["LOW", "NORMAL", "HIGH", "CRITICAL"]
ACTIONS = ["LOW TARIFF", "NORMAL TARIFF", "PEAK TARIFF"]

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((len(STATES), len(ACTIONS)))
        
    def get_state_index(self, state):
        return STATES.index(state)
        
    def get_action_index(self, action):
        return ACTIONS.index(action)
        
    def choose_action(self, state, explore=True):
        state_idx = self.get_state_index(state)
        if explore and np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(ACTIONS)
        else:
            action_idx = np.argmax(self.q_table[state_idx])
            return ACTIONS[action_idx]
            
    def get_reward(self, state, action):
        """
        Reward function designing the grid behavior:
        - We want normal demand.
        - High tariffs during high demand reduces demand (good for grid).
        - High tariffs during low demand is bad (penalize).
        - Low tariffs during high demand causes overload (heavy penalty).
        """
        if state == "CRITICAL":
            if action == "PEAK TARIFF":
                return 10  # Correctly applied peak pricing
            elif action == "NORMAL TARIFF":
                return -5
            else:
                return -20 # Catastrophic grid overload
                
        elif state == "HIGH":
            if action == "PEAK TARIFF":
                return 5
            elif action == "NORMAL TARIFF":
                return -2
            else:
                return -10
                
        elif state == "NORMAL":
            if action == "NORMAL TARIFF":
                return 5
            elif action == "PEAK TARIFF":
                return -5 # Unnecessary burden on consumers
            else:
                return -2 # Might cause spike
                
        elif state == "LOW":
            if action == "LOW TARIFF":
                return 5 # Encourages demand shifting to valley
            elif action == "NORMAL TARIFF":
                return -2
            else:
                return -10 # Exploitative pricing
        
        return 0
        
    def get_next_state(self, state, action):
        """
        Simulate environment dynamics. 
        Higher tariffs reduce demand, lower tariffs increase demand.
        """
        state_idx = self.get_state_index(state)
        
        if action == "PEAK TARIFF":
            # Likely reduces demand level
            next_idx = max(0, state_idx - 1)
        elif action == "LOW TARIFF":
            # Likely increases demand level
            next_idx = min(len(STATES) - 1, state_idx + 1)
        else:
            # Stays roughly same
            next_idx = state_idx
            
        return STATES[next_idx]
        
    def train(self, episodes=1000):
        """Train the Q-Learning agent."""
        for _ in range(episodes):
            state = np.random.choice(STATES)
            
            # Simple episode: 1 step for contextual bandit-like simplicity
            action = self.choose_action(state, explore=True)
            reward = self.get_reward(state, action)
            next_state = self.get_next_state(state, action)
            
            # Update Q-table
            state_idx = self.get_state_index(state)
            action_idx = self.get_action_index(action)
            next_state_idx = self.get_state_index(next_state)
            
            best_next_action = np.argmax(self.q_table[next_state_idx])
            
            self.q_table[state_idx, action_idx] = self.q_table[state_idx, action_idx] + \
                self.alpha * (reward + self.gamma * self.q_table[next_state_idx, best_next_action] - self.q_table[state_idx, action_idx])
                
    def get_recommendation(self, state):
        """Get the best action for a given state after training."""
        action = self.choose_action(state, explore=False)
        return action
