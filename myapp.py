# app.py
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

def load_data():
    # Replace 'https://raw.githubusercontent.com/your_username/your_repo/main/data.csv'
    # with the GitHub URL of your data
    url = 'https://raw.githubusercontent.com/amanKIIT/Customer-Segmentation/main/train_data.csv'
    data = pd.read_csv(url)
    return data

def preprocess_data(data):
    # Select the features for clustering
    selected_features = ['Income', 'NumDealsPurchases', 'NumWebVisitsMonth', 'Age', 'TotalSpending', 'Children']
    X = data[selected_features].values
    return X

def main():
    st.title('Customer Cluster Prediction App')
    st.write("This app predicts the cluster of a user based on given features.")

    # Load your customer data from the GitHub URL
    data = load_data()

    # Preprocess the data for clustering
    X = preprocess_data(data)

    # Train the K-means clustering model and predict clusters
    n_clusters = 4  # You can change the number of clusters here
    model = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = model.fit_predict(X)

    # Create input widgets to get user's feature data
    user_data = {}
    st.subheader('Enter User Features:')
    for feature in ['Income', 'NumDealsPurchases', 'NumWebVisitsMonth', 'Age', 'TotalSpending', 'Children']:
        user_data[feature] = st.number_input(f'{feature}:', step=1)

    if st.button('Predict'):
        # Prepare the user data for clustering
        user_df = pd.DataFrame([user_data])
        X_user = user_df.values

        # Predict the cluster for the user data using the trained model
        user_cluster = model.predict(X_user)

        # Display the predicted cluster
        st.subheader('Prediction Result:')
        st.write(f"The predicted cluster for the user is Cluster {user_cluster[0]}")

if __name__ == '__main__':
    main()
