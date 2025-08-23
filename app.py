import streamlit as st
from face_grid_rl import FaceGridEnv, QLearningAgent
import numpy as np
import matplotlib.pyplot as plt

st.title('RL Face Grid Navigation')

grid_size = st.sidebar.slider('Grid Size', 5, 20, 10)
num_episodes = st.sidebar.slider('Episodes', 50, 600, 200)
target_zones = [(grid_size//3, grid_size//3), (grid_size-3,2), (grid_size//2,grid_size-2)]

if st.button('Run Q-Learning Agent'):
    env = FaceGridEnv(grid_size=(grid_size,grid_size), target_zones=target_zones)
    agent = QLearningAgent(env)
    reward_curve = []
    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            ep_reward += reward
        reward_curve.append(ep_reward)
    st.subheader('Reward Curve')
    fig, ax = plt.subplots()
    ax.plot(reward_curve)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Q-Learning Agent Reward Curve')
    st.pyplot(fig)

    st.subheader('Agent Final Position in Grid')
    env.render()
    st.pyplot(fig)