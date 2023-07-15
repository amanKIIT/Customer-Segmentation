# app.py
import streamlit as st
import pandas as pd
import pickle

def load_model():
    # Load the pre-trained model information from the 'model_info.pkl' file
    with open('model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)

    return model_info['model'], model_info['selected_features'], model_info['cluster_centers']

def main():
    st.title('Customer Segmentsation')
    st.write("This app predicts the cluster of a user based on given features.")

    # Load the pre-trained model
    model, selected_features, cluster_centers = load_model()

    # Create input widgets to get user's feature data
    user_data = {}
    st.subheader('Enter Customer Details:')
    for feature in selected_features:
        user_data[feature] = st.number_input(f'{feature}:', step=1)

    if st.button('Predict'):
        # Prepare the user data for clustering
        user_df = pd.DataFrame([user_data])
        X_user = user_df[selected_features]

        # Predict the cluster for the user data using the pre-trained model
        cluster_pred = model.predict(X_user)

        # Display the predicted cluster and the cluster center information
        st.subheader('Prediction Result:')
        st.write(f"The predicted cluster for the user is Cluster {cluster_pred[0]}")

if __name__ == '__main__':
    main()
