import gymnasium as gym
import time

# Create environment
env = gym.make("CartPole-v1", render_mode="human")

# Reset environment
state, info = env.reset()

for step in range(200):
    # Add delay (in seconds)
    time.sleep(0.5)  # 0.1 seconds = 100 milliseconds
    
    # Random action (0 = left, 1 = right)
    action = env.action_space.sample()
    
    # Take action
    next_state, reward, terminated, truncated, info = env.step(action)
    
    # Print info
    print(f"Step: {step}, Action: {action}, Reward: {reward}")
    
    # End episode
    if terminated or truncated:
        print("Episode finished!")
        break

# Close environment
env.close()