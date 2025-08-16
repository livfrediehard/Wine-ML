#!/usr/bin/env python3
"""
Basic Machine Learning with scikit-learn
This script introduces traditional ML algorithms using scikit-learn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)

def main():
    print("🤖 Basic Machine Learning with scikit-learn")
    print("=" * 60)
    
    # 1. Data Preparation
    print("\n1. Data Preparation...")
    red_wine = pd.read_csv('wine+quality/winequality-red.csv', sep=';')
    white_wine = pd.read_csv('wine+quality/winequality-white.csv', sep=';')
    
    # Add wine type and combine
    red_wine['wine_type'] = 'red'
    white_wine['wine_type'] = 'white'
    wine_data = pd.concat([red_wine, white_wine], ignore_index=True)
    
    # Encode wine type
    le = LabelEncoder()
    wine_data['wine_type_encoded'] = le.fit_transform(wine_data['wine_type'])
    
    # Prepare features and target
    feature_columns = [col for col in wine_data.columns if col not in ['quality', 'wine_type']]
    X = wine_data[feature_columns]
    y = wine_data['quality']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {feature_columns}")
    
    # 2. Train-Test Split and Scaling
    print("\n2. Train-Test Split and Scaling...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # 3. Linear Models
    print("\n3. Linear Models...")
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    
    print("Linear Regression Results:")
    print(f"  Mean Squared Error: {mse_lr:.4f}")
    print(f"  R² Score: {r2_lr:.4f}")
    print(f"  Mean Absolute Error: {mae_lr:.4f}")
    
    # Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    
    print("\nRidge Regression Results:")
    print(f"  Mean Squared Error: {mse_ridge:.4f}")
    print(f"  R² Score: {r2_ridge:.4f}")
    
    # 4. Tree-Based Models
    print("\n4. Tree-Based Models...")
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)  # No scaling needed for tree-based models
    y_pred_rf = rf.predict(X_test)
    
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    
    print("Random Forest Results:")
    print(f"  Mean Squared Error: {mse_rf:.4f}")
    print(f"  R² Score: {r2_rf:.4f}")
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    
    mse_gb = mean_squared_error(y_test, y_pred_gb)
    r2_gb = r2_score(y_test, y_pred_gb)
    
    print("\nGradient Boosting Results:")
    print(f"  Mean Squared Error: {mse_gb:.4f}")
    print(f"  R² Score: {r2_gb:.4f}")
    
    # 5. Support Vector Regression
    print("\n5. Support Vector Regression...")
    svr = SVR(kernel='rbf', C=1.0, gamma='scale')
    svr.fit(X_train_scaled, y_train)
    y_pred_svr = svr.predict(X_test_scaled)
    
    mse_svr = mean_squared_error(y_test, y_pred_svr)
    r2_svr = r2_score(y_test, y_pred_svr)
    
    print("SVR Results:")
    print(f"  Mean Squared Error: {mse_svr:.4f}")
    print(f"  R² Score: {r2_svr:.4f}")
    
    # 6. Model Comparison
    print("\n6. Model Comparison...")
    models = {
        'Linear Regression': (y_pred_lr, mse_lr, r2_lr),
        'Ridge Regression': (y_pred_ridge, mse_ridge, r2_ridge),
        'Random Forest': (y_pred_rf, mse_rf, r2_rf),
        'Gradient Boosting': (y_pred_gb, mse_gb, r2_gb),
        'SVR': (y_pred_svr, mse_svr, r2_svr)
    }
    
    # Print comparison table
    print("\nModel Performance Summary:")
    print("-" * 60)
    print(f"{'Model':<20} {'MSE':<10} {'R²':<10}")
    print("-" * 60)
    for name, (_, mse, r2) in models.items():
        print(f"{name:<20} {mse:<10.4f} {r2:<10.4f}")
    
    # 7. Feature Importance Analysis
    print("\n7. Feature Importance Analysis...")
    feature_importance = rf.feature_importances_
    feature_names = feature_columns
    
    # Create feature importance plot
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.title('Feature Importance (Random Forest)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTop 5 Most Important Features:")
    for i, (feature, importance) in enumerate(zip(importance_df['feature'][-5:], importance_df['importance'][-5:])):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    # 8. Cross-Validation
    print("\n8. Cross-Validation...")
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
    
    print("Cross-Validation Results (Random Forest):")
    print(f"CV R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 9. Hyperparameter Tuning
    print("\n9. Hyperparameter Tuning...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Test the tuned model
    best_rf = grid_search.best_estimator_
    y_pred_best = best_rf.predict(X_test)
    r2_best = r2_score(y_test, y_pred_best)
    print(f"Test R² score: {r2_best:.4f}")
    
    # 10. Key Takeaways
    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    
    print("\nWhat We Learned:")
    print("1. Data Preprocessing: Scaling is crucial for some algorithms (SVR, linear models)")
    print("2. Model Performance: Tree-based models (Random Forest, Gradient Boosting) performed best")
    print("3. Feature Importance: Alcohol content and volatile acidity are key predictors")
    print("4. Cross-Validation: Essential for reliable performance estimation")
    print("5. Hyperparameter Tuning: Can improve model performance significantly")
    
    print("\nFor Legal Document Analysis:")
    print("- Text preprocessing will be similar to feature scaling")
    print("- Feature engineering will involve extracting text features (TF-IDF, word embeddings)")
    print("- Model selection will depend on your specific task (classification vs. regression)")
    print("- Interpretability is crucial for legal applications")
    
    print("\nNext: We'll explore neural networks with PyTorch!")

if __name__ == "__main__":
    main()
