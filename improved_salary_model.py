import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class SalaryPredictor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the data"""
        print("Loading and preprocessing data...")
        
        # Load data
        data = pd.read_csv(filepath)
        print(f"Initial data shape: {data.shape}")
        
        # Handle missing values (? marks)
        data.replace({'?': 'Unknown'}, inplace=True)
        
        # Clean workclass - remove problematic categories
        data = data[~data['workclass'].isin(['Without-pay', 'Never-worked'])]
        
        # Remove very rare education categories
        rare_education = ['1st-4th', '5th-6th', 'Preschool']
        if 'education' in data.columns:
            data = data[~data['education'].isin(rare_education)]
        
        # Handle outliers
        data = data[(data['age'] >= 17) & (data['age'] <= 75)]
        data = data[(data['educational-num'] >= 5) & (data['educational-num'] <= 16)]
        
        # Remove redundant features (education is redundant with educational-num)
        if 'education' in data.columns:
            data = data.drop(columns=['education'])
        
        print(f"Data shape after cleaning: {data.shape}")
        return data
    
    def encode_features(self, data, fit=True):
        """Encode categorical features"""
        categorical_features = ['workclass', 'marital-status', 'occupation', 
                              'relationship', 'race', 'gender', 'native-country']
        
        data_encoded = data.copy()
        
        for feature in categorical_features:
            if feature in data.columns:
                if fit:
                    self.label_encoders[feature] = LabelEncoder()
                    data_encoded[feature] = self.label_encoders[feature].fit_transform(data[feature])
                else:
                    if feature in self.label_encoders:
                        # Handle unknown categories
                        known_categories = set(self.label_encoders[feature].classes_)
                        data[feature] = data[feature].apply(
                            lambda x: x if x in known_categories else 'Unknown'
                        )
                        
                        # Add 'Unknown' to encoder if not present
                        if 'Unknown' not in known_categories:
                            self.label_encoders[feature].classes_ = np.append(
                                self.label_encoders[feature].classes_, 'Unknown'
                            )
                        
                        data_encoded[feature] = self.label_encoders[feature].transform(data[feature])
        
        return data_encoded
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models and select the best one"""
        print("Training multiple models...")
        
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True),
            'KNN': KNeighborsClassifier(),
            'DecisionTree': DecisionTreeClassifier(random_state=42)
        }
        
        results = {}
        trained_models = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            trained_models[name] = model
            
            print(f"{name} Accuracy: {accuracy:.4f}")
        
        # Select best model
        best_model_name = max(results, key=results.get)
        self.model = trained_models[best_model_name]
        
        print(f"\nBest model: {best_model_name} with accuracy: {results[best_model_name]:.4f}")
        
        # Create ensemble of top 3 models
        top_3_models = sorted(results.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"Top 3 models: {[name for name, _ in top_3_models]}")
        
        ensemble_models = [(name, trained_models[name]) for name, _ in top_3_models]
        ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
        ensemble.fit(X_train, y_train)
        
        ensemble_pred = ensemble.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        
        print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        
        # Use ensemble if it's better than the best individual model
        if ensemble_accuracy > results[best_model_name]:
            self.model = ensemble
            print("Using ensemble model as final model")
        else:
            print(f"Using {best_model_name} as final model")
        
        return results
    
    def fit(self, filepath):
        """Complete training pipeline"""
        # Load and preprocess data
        data = self.load_and_preprocess_data(filepath)
        
        # Separate features and target
        X = data.drop(columns=['income'])
        y = data['income']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Encode categorical features
        X_encoded = self.encode_features(X, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        results = self.train_models(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Print detailed results for best model
        y_pred = self.model.predict(X_test_scaled)
        print("\nDetailed Results:")
        print("="*50)
        print(classification_report(y_test, y_pred))
        
        return results
    
    def predict(self, X):
        """Make predictions on new data"""
        # Encode categorical features
        X_encoded = self.encode_features(X, fit=False)
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in X_encoded.columns:
                X_encoded[feature] = 0  # Default value for missing features
        
        # Select and reorder features
        X_encoded = X_encoded[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X_encoded)
        
        # Predict
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        # Encode categorical features
        X_encoded = self.encode_features(X, fit=False)
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in X_encoded.columns:
                X_encoded[feature] = 0
        
        # Select and reorder features
        X_encoded = X_encoded[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X_encoded)
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            return None
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filepath}")

def main():
    # Initialize predictor
    predictor = SalaryPredictor()
    
    # Train model
    results = predictor.fit("adult 3.csv")
    
    # Save model
    predictor.save_model("improved_salary_model.pkl")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values(), color='skyblue')
    plt.title('Model Comparison - Improved Pipeline')
    plt.ylabel('Accuracy Score')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
