import joblib
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("Loading extracted features...")
data = joblib.load("extracted_features.joblib")
X, y = data['X'], data['y']

# Optional: Split data to check performance before using it live
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Building PCA + SVM Pipeline...")
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('pca', PCA(n_components=0.95)), # <-- Back to 95% variance!
    ('svm', SVC(kernel='rbf', class_weight='balanced'))
])

print("Training model (This might take a few minutes)...")
pipeline.fit(X_train, y_train)

print("Evaluating model...")
predictions = pipeline.predict(X_test)
print(classification_report(y_test, predictions))

print("Saving final trained model...")
joblib.dump(pipeline, "asl_svm_model.joblib", compress=9)
print("Done! Proceed to Step 3.")