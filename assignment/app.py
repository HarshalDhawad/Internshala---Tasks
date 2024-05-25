from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

def preprocess_data():
    # Load the raw data
    raw_data_path = 'rawdata.xlsx'
    raw_data = pd.read_excel(raw_data_path)

    # Ensure that 'date' and 'time' columns are in string format
    raw_data['date'] = raw_data['date'].astype(str)
    raw_data['time'] = raw_data['time'].astype(str)

    # Concatenate 'date' and 'time' columns and convert to datetime
    raw_data['datetime'] = pd.to_datetime(raw_data['date'] + ' ' + raw_data['time'])

    # Extract the date part for grouping purposes
    raw_data['date'] = raw_data['datetime'].dt.date

    # Calculate duration (assuming the dataset is ordered by time)
    raw_data['duration'] = raw_data['datetime'].diff().dt.total_seconds().fillna(0)

    # Classify the activities as 'picked' or 'placed'
    raw_data['activity'] = raw_data.apply(lambda row: 'picked' if row['sensor'] == 1 else 'placed', axis=1)

    # Classify locations as 'inside' or 'outside'
    raw_data['location_type'] = raw_data['location'].apply(lambda location: 'inside' if location.startswith('A') else 'outside')

    return raw_data

def load_and_preprocess_data():
    # Load the raw data
    training_file_path = 'train.xlsx'
    testing_file_path = 'test.xlsx'

    training_data = pd.read_excel(training_file_path)
    testing_data = pd.read_excel(testing_file_path)

    training_features = training_data.drop('target', axis=1)
    scaler = StandardScaler()
    normalized_training_features = scaler.fit_transform(training_features)

    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(normalized_training_features)
    training_data['Cluster'] = kmeans.labels_

    normalized_testing_features = scaler.transform(testing_data)
    testing_data['Predicted Cluster'] = kmeans.predict(normalized_testing_features)

    silhouette = silhouette_score(normalized_testing_features, testing_data['Predicted Cluster'])

    new_data_point = np.array([-65, -55, -60, -50, -48, -58, -75, -55, -70, -73, -60, -62, -55, -56, -68, -70, -58, -72])
    normalized_new_data_point = scaler.transform(new_data_point.reshape(1, -1))
    predicted_cluster = kmeans.predict(normalized_new_data_point)[0]

    # Plot the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(normalized_training_features[:, 0], normalized_training_features[:, 1], c=training_data['Cluster'], cmap='viridis', s=50, alpha=0.5)
    plt.scatter(normalized_new_data_point[:, 0], normalized_new_data_point[:, 1], c='red', marker='*', s=200, label='New Data Point')
    plt.title('Cluster Plot')
    plt.legend()
    plt.colorbar(label='Cluster')
    plt.grid(True)
    
    # Save the plot to a PNG image in a bytes buffer
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return testing_data.head(10), silhouette, predicted_cluster, plot_url

@app.route('/')
def index():

    
    testing_data, silhouette, predicted_cluster, plot_url = load_and_preprocess_data()
    # return render_template('index.html', testing_data=testing_data.to_html(), silhouette=silhouette, predicted_cluster=predicted_cluster, plot_url=plot_url)

    training_file_path = 'train.xlsx'
    training_data = pd.read_excel(training_file_path)

    # Process data
    X = training_data.drop('target', axis=1)
    y = training_data['target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_scaled, y_train)

    # Calculate accuracy
    predicted_val = rf_classifier.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, predicted_val)

    # Get top 10 rows
    top_10_rows = training_data.head(10).to_html()

    raw_data = preprocess_data()

    # Calculate date-wise total duration for each location type
    duration_summary = raw_data.groupby(['date', 'location_type'])['duration'].sum().reset_index()
    duration_summary_pivot = duration_summary.pivot(index='date', columns='location_type', values='duration').fillna(0).reset_index()

    # Calculate date-wise number of picking and placing activities
    activity_summary = raw_data.groupby(['date', 'activity'])['activity'].count().unstack().fillna(0).reset_index()

    # Merge the duration and activity summaries
    final_summary = pd.merge(duration_summary_pivot, activity_summary, on='date', how='outer').fillna(0)

    # Render the results in an HTML page
    return render_template('index.html',top_10_rows=top_10_rows, accuracy=accuracy, testing_data=testing_data.to_html(), silhouette=silhouette, predicted_cluster=predicted_cluster, plot_url=plot_url, duration_summary=duration_summary_pivot, activity_summary=activity_summary, final_summary=final_summary)

if __name__ == '__main__':
    app.run(debug=True)
