import numpy as np

class ReplayBuffer():
    def __init__(self, max_buffer_size, n_agents, n_states, n_actions):
        self.max_buffer_size = max_buffer_size
        self.mem_ptr = 0
        self.state_memory = np.zeros((self.max_buffer_size, n_agents, n_states), dtype=np.float32)
        self.next_state_memory = np.zeros((self.max_buffer_size, n_agents, n_states), dtype=np.float32)
        self.action_memory = np.zeros((self.max_buffer_size, n_agents, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros((self.max_buffer_size, n_agents), dtype=np.float32)
        self.terminal_memory = np.zeros((self.max_buffer_size, n_agents), dtype=np.bool)
        
    def store_transition(self, state, action, reward, state_, done):
    # def store_transition(self, state, action, reward, state_):
        index = self.mem_ptr % self.max_buffer_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_ptr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ptr, self.max_buffer_size)

        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.terminal_memory[batch]
        # print(rewards)

        return states, actions, rewards, next_states, dones
        # return states, actions, rewards, next_states