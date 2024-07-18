import pickle
import pandas as pd
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.cluster import KMeans,DBSCAN
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import ast
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity



with open("C:/Users/mistr/jupyter notebook codes/AMNS AIML Internship/SAP GRC Roles Recommendation Project/best_rf_model.pkl", 'rb') as file:
    model = pickle.load(file)

data = pd.read_csv("C:/Users/mistr/jupyter notebook codes/AMNS AIML Internship/SAP GRC Roles Recommendation Project/Sample_SAP_GRC_DATA.csv")

data['Roles'] = data['Roles'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

department_encoder = OneHotEncoder()
department_encoded = department_encoder.fit_transform(data[['Department']]).toarray()

mlb = MultiLabelBinarizer()
roles_encoded = mlb.fit_transform(data['Roles'])

encoded_features = np.hstack((department_encoded, roles_encoded))

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



def recommend_roles(new_user, df, model,  top_n=5, min_role_occurrence=2, role_similarity_threshold=0.4):

    new_user_df = pd.DataFrame([new_user])

    department_encoded = department_encoder.transform(new_user_df[['Department']]).toarray()
    roles_encoded = mlb.transform([new_user['Roles']])

    user_features = np.hstack((department_encoded, roles_encoded))

    predicted_cluster = model.predict(user_features)[0]

    cluster_users = df[df['Cluster'] == predicted_cluster]
    department_users = cluster_users[cluster_users['Department'] == new_user['Department']]
 
    if department_users.empty:
        print("No users found in the same department within the predicted cluster.")
        return []

    cluster_roles = department_users['Roles'].explode()

    role_counts = cluster_roles.value_counts()

    similar_roles = role_counts[role_counts >= min_role_occurrence].index.tolist()

    if not similar_roles:
        print("No similar roles found in the predicted cluster.")
        return []

    similar_roles_encoded = mlb.transform([similar_roles])
    new_user_roles_encoded = roles_encoded

    role_similarity = cosine_similarity(similar_roles_encoded, new_user_roles_encoded).mean()
    print(role_similarity)
    if role_similarity < role_similarity_threshold:
        print(f"Insufficient similarity to provide recommendations for {new_user['User']}.")
        return []
    
    top_roles = role_counts.head(top_n + len(new_user['Roles'])).index.tolist()
    top_roles = [role for role in top_roles if role not in new_user['Roles']][:top_n]
    
    return top_roles


new_user = {'User': 'User5004', 'Department': 'Finance', 'Roles': ['SAP_FI','SAP_CO']}
recommended_roles = recommend_roles(new_user, data, model, top_n=3)
print(f"Recommended roles for {new_user['User']}: {recommended_roles}")


