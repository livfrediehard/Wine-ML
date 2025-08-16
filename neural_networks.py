#!/usr/bin/env python3
"""
Neural Networks with PyTorch
This script introduces deep learning concepts using PyTorch for wine quality prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class WineDataset(Dataset):
    """Custom Dataset for wine data"""
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets).reshape(-1, 1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class WineQualityNN(nn.Module):
    """Neural Network for wine quality prediction"""
    def __init__(self, input_size, hidden_sizes=[64, 32, 16], dropout_rate=0.2):
        super(WineQualityNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, device='cpu'):
    """Train the neural network"""
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
    
    return train_losses, val_losses

def main():
    print("🧠 Neural Networks with PyTorch")
    print("=" * 60)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # 2. Data Preprocessing
    print("\n2. Data Preprocessing...")
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # 3. Create PyTorch Datasets and DataLoaders
    print("\n3. Creating PyTorch Datasets...")
    train_dataset = WineDataset(X_train_scaled, y_train.values)
    val_dataset = WineDataset(X_val_scaled, y_val.values)
    test_dataset = WineDataset(X_test_scaled, y_test.values)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. Build Neural Network
    print("\n4. Building Neural Network...")
    input_size = X_train.shape[1]
    model = WineQualityNN(input_size=input_size, hidden_sizes=[64, 32, 16], dropout_rate=0.2)
    model.to(device)
    
    print(f"Model architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 5. Training Setup
    print("\n5. Training Setup...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 6. Train the Model
    print("\n6. Training the Model...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=100, device=device
    )
    
    # 7. Plot Training Curves
    print("\n7. Plotting Training Curves...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot learning curves in log scale
    plt.subplot(1, 2, 2)
    plt.semilogy(train_losses, label='Training Loss', color='blue')
    plt.semilogy(val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. Model Evaluation
    print("\n8. Model Evaluation...")
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            test_predictions.extend(outputs.cpu().numpy().flatten())
            test_targets.extend(targets.cpu().numpy().flatten())
    
    # Calculate metrics
    mse = mean_squared_error(test_targets, test_predictions)
    r2 = r2_score(test_targets, test_predictions)
    rmse = np.sqrt(mse)
    
    print(f"Test Results:")
    print(f"  Mean Squared Error: {mse:.4f}")
    print(f"  Root Mean Squared Error: {rmse:.4f}")
    print(f"  R² Score: {r2:.4f}")
    
    # 9. Prediction vs Actual Plot
    print("\n9. Creating Prediction vs Actual Plot...")
    plt.figure(figsize=(10, 8))
    
    plt.scatter(test_targets, test_predictions, alpha=0.6, color='blue')
    plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], 'r--', lw=2)
    plt.xlabel('Actual Quality')
    plt.ylabel('Predicted Quality')
    plt.title('Predicted vs Actual Wine Quality')
    plt.grid(True, alpha=0.3)
    
    # Add R² text
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 10. Feature Importance Analysis
    print("\n10. Feature Importance Analysis...")
    # Use gradient-based feature importance
    model.eval()
    feature_importance = np.zeros(input_size)
    
    for features, targets in train_loader:
        features = features.to(device)
        features.requires_grad_(True)
        
        outputs = model(features)
        loss = criterion(outputs, torch.ones_like(outputs))
        loss.backward()
        
        feature_importance += np.abs(features.grad.cpu().numpy()).mean(axis=0)
    
    feature_importance /= len(train_loader)
    
    # Create feature importance plot
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.title('Feature Importance (Neural Network)')
    plt.xlabel('Importance (Gradient-based)')
    plt.tight_layout()
    plt.savefig('nn_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTop 5 Most Important Features (Neural Network):")
    for i, (feature, importance) in enumerate(zip(importance_df['feature'][-5:], importance_df['importance'][-5:])):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    # 11. Model Comparison with Traditional ML
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    print("\nNeural Network Results:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    print("\nTraditional ML Results (from basic_ml.py):")
    print("  Random Forest:     R² = 0.499")
    print("  Gradient Boosting: R² = 0.376")
    print("  Linear Regression: R² = 0.267")
    
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    print("\nWhat You Learned:")
    print("1. Neural Networks: Can capture complex non-linear relationships")
    print("2. Overfitting: Validation loss helps detect overfitting")
    print("3. Hyperparameters: Learning rate, batch size, architecture matter")
    print("4. Feature Importance: Gradient-based methods vs tree-based methods")
    print("5. Model Complexity: More complex doesn't always mean better")
    
    print("\nFor Legal Document Analysis:")
    print("- Neural networks excel at text classification and sentiment analysis")
    print("- Word embeddings can capture semantic relationships in legal text")
    print("- Attention mechanisms can highlight important parts of documents")
    print("- Transfer learning (BERT, GPT) can leverage pre-trained legal models")
    
    print("\nNext Steps:")
    print("1. Try different architectures (deeper, wider networks)")
    print("2. Experiment with different optimizers and learning rates")
    print("3. Apply these concepts to text data")
    print("4. Learn about transformers for legal document analysis")

if __name__ == "__main__":
    main()
