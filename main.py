import time
import pandas as pd
from model import save_model, load_model, create_env, ExperienceReplayBuffer
from utils import download_stock_data, preprocess_data, add_technical_indicators

def train_model(env, timesteps=10000):
    """
    Train a model using PPO (Proximal Policy Optimization).
    """
    from stable_baselines3 import PPO
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    return model

def evaluate_model(env, model):
    """
    Evaluate the model performance.
    """
    obs = env.reset()
    rewards = []
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
    return rewards

def live_training_loop(ticker, initial_start_date, window_size):
    """
    Continuously train the model with live data.
    """
    start_date = initial_start_date
    model = None

    while True:
        # Define the end date as today
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

        # Download new data
        data = download_stock_data(ticker, start_date, end_date)
        data = preprocess_data(data)
        data = add_technical_indicators(data)

        # Create or update the environment
        env = create_env(data, window_size)

        # Train or update the model
        if model is None:
            model = train_model(env, timesteps=10000)  # Train a new model
        else:
            model.learn(total_timesteps=10000)  # Continue training

        # Save the updated model
        save_model(model, "live_trading_model.zip")

        # Update the start date to fetch only new data next time
        start_date = end_date

        # Evaluate the model
        rewards = evaluate_model(env, model)
        print(f"Evaluation rewards: {sum(rewards) / len(rewards)}")

        # Wait (e.g., 1 day) before fetching new data
        print("Training completed for this session. Waiting for the next update...")
        time.sleep(86400)  # Sleep for 1 day

def start():
    ticker = 'AAPL'  # Example: Apple Inc. stock
    initial_start_date = '2019-01-01'  # Historical data starting date
    window_size = 20  # Define window size for technical indicators and trading

    # Start the live training loop with continuous model improvement
    live_training_loop(ticker, initial_start_date, window_size)

if __name__ == '__main__':
    start()
