import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from face_grid_rl import FaceGridEnv, QLearningAgent

st.title('RL Face Grid Navigation & Real-Time Face Landmarks')

tab1, tab2 = st.tabs(["RL Grid Navigation", "Webcam Facial Landmarks"])

with tab1:
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

with tab2:
    st.write("Enable webcam and detect facial landmarks in real time!")
    run = st.button("Start Webcam (Press Q to Quit)")
    if run:
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        cap = cv2.VideoCapture(0)
        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            stframe = st.empty()
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    st.write("Ignoring empty camera frame.")
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                stframe.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
            cap.release()
