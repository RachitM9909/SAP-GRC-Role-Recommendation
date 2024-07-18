import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.cluster import KMeans,DBSCAN
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import ast
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import pickle

df = pd.read_csv("C:/Users/mistr/jupyter notebook codes/AMNS AIML Internship/SAP GRC Roles Recommendation Project/Sample_SAP_GRC_DATA.csv")

df.head()

data = df
data['Roles'] = data['Roles'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

department_encoder = OneHotEncoder()
department_encoded = department_encoder.fit_transform(data[['Department']]).toarray()

mlb = MultiLabelBinarizer()
roles_encoded = mlb.fit_transform(data['Roles'])

encoded_features = np.hstack((department_encoded, roles_encoded))

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(encoded_features)


data['Cluster'] = clusters


silhouette_avg = silhouette_score(encoded_features, clusters)
print(f'Silhouette Score: {silhouette_avg}')

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(encoded_features, clusters)

# Find the optimal number of clusters using PCA
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(encoded_features)

best_k = 2
best_score = -1

for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(reduced_features)
    score = silhouette_score(reduced_features, clusters)
    if score > best_score:
        best_k = k
        best_score = score

kmeans = KMeans(n_clusters=best_k, random_state=42)
clusters = kmeans.fit_predict(reduced_features)
data['Cluster'] = clusters

X_train, X_test, y_train, y_test = train_test_split(encoded_features, clusters, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)


predictions = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy on test data:", accuracy)
print("Classification Report on test data:\n", classification_report(y_test, predictions))
print("RMSE on test data:", np.sqrt(mean_squared_error(y_test, predictions)))


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(encoded_features, clusters)

best_rf_model = grid_search.best_estimator_

# Print the best parameters found by GridSearchCV
print("Best Parameters:", grid_search.best_params_)
best_rf_model.fit(X_train,y_train)
predictions = best_rf_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy on test data:", accuracy)

# Classification report
print("Classification Report on test data:")
print(classification_report(y_test, predictions))

rmse = mean_squared_error(y_test, predictions, squared=False)
print("RMSE on test data:", rmse)

with open('best_rf_model.pkl', 'wb') as file:
    pickle.dump(best_rf_model, file)