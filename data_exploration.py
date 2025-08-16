#!/usr/bin/env python3
"""
Wine Quality Dataset Exploration
This script explores the wine dataset to understand the data before building ML models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

def main():
    print("🍷 Wine Quality Dataset Exploration")
    print("=" * 50)
    
    # 1. Loading the Data
    print("\n1. Loading the Data...")
    red_wine = pd.read_csv('wine+quality/winequality-red.csv', sep=';')
    white_wine = pd.read_csv('wine+quality/winequality-white.csv', sep=';')
    
    # Add wine type column for identification
    red_wine['wine_type'] = 'red'
    white_wine['wine_type'] = 'white'
    
    # Combine datasets
    wine_data = pd.concat([red_wine, white_wine], ignore_index=True)
    
    print(f"Red wine samples: {len(red_wine)}")
    print(f"White wine samples: {len(white_wine)}")
    print(f"Total samples: {len(wine_data)}")
    print(f"Dataset shape: {wine_data.shape}")
    
    # 2. Understanding the Data Structure
    print("\n2. Data Structure:")
    print("Dataset Info:")
    print(wine_data.info())
    
    print("\nFirst few rows:")
    print(wine_data.head())
    
    print("\nColumn names:")
    print(wine_data.columns.tolist())
    
    # 3. Basic Statistics
    print("\n3. Basic Statistics:")
    print("Descriptive Statistics:")
    print(wine_data.describe())
    
    print("\nMissing Values:")
    print(wine_data.isnull().sum())
    
    # 4. Target Variable Analysis (Quality)
    print("\n4. Quality Analysis:")
    print("Quality Score Distribution:")
    print(wine_data['quality'].value_counts().sort_index())
    
    # Create quality distribution plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall quality distribution
    wine_data['quality'].hist(bins=11, edgecolor='black', ax=ax1)
    ax1.set_title('Distribution of Wine Quality Scores')
    ax1.set_xlabel('Quality Score')
    ax1.set_ylabel('Frequency')
    
    # Quality by wine type
    wine_data.groupby('wine_type')['quality'].hist(alpha=0.7, bins=11, edgecolor='black', ax=ax2)
    ax2.set_title('Quality Distribution by Wine Type')
    ax2.set_xlabel('Quality Score')
    ax2.set_ylabel('Frequency')
    ax2.legend(['Red', 'White'])
    
    plt.tight_layout()
    plt.savefig('quality_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Quality statistics by wine type
    print("\nQuality Statistics by Wine Type:")
    print(wine_data.groupby('wine_type')['quality'].describe())
    
    # 5. Feature Analysis
    print("\n5. Feature Analysis:")
    feature_columns = [col for col in wine_data.columns if col not in ['quality', 'wine_type']]
    
    # Create correlation matrix
    correlation_matrix = wine_data[feature_columns + ['quality']].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Show correlations with quality
    quality_correlations = correlation_matrix['quality'].sort_values(ascending=False)
    print("\nCorrelations with Quality:")
    print(quality_correlations)
    
    # 6. Key Insights
    print("\n" + "="*50)
    print("KEY INSIGHTS")
    print("="*50)
    
    print("\n1. Dataset Balance:")
    print(f"   - Red wines: {len(red_wine)} samples")
    print(f"   - White wines: {len(white_wine)} samples")
    print(f"   - Total: {len(wine_data)} samples")
    
    print("\n2. Quality Distribution:")
    print(f"   - Quality range: {wine_data['quality'].min()} to {wine_data['quality'].max()}")
    print(f"   - Most common quality: {wine_data['quality'].mode().iloc[0]}")
    print(f"   - Average quality: {wine_data['quality'].mean():.2f}")
    
    print("\n3. Top Features Correlated with Quality:")
    top_features = quality_correlations[1:6]  # Exclude quality itself
    for feature, corr in top_features.items():
        print(f"   - {feature}: {corr:.3f}")
    
    print("\n4. Data Quality:")
    print(f"   - Missing values: {wine_data.isnull().sum().sum()}")
    print(f"   - Duplicate rows: {wine_data.duplicated().sum()}")
    
    print("\n5. Wine Type Differences:")
    red_avg = red_wine['quality'].mean()
    white_avg = white_wine['quality'].mean()
    print(f"   - Red wine average quality: {red_avg:.2f}")
    print(f"   - White wine average quality: {white_avg:.2f}")
    print(f"   - Difference: {abs(red_avg - white_avg):.2f}")
    
    print("\n" + "="*50)
    print("Next Steps:")
    print("1. The dataset is clean with no missing values")
    print("2. Quality scores are concentrated around 5-6 (imbalanced)")
    print("3. Alcohol content and volatile acidity are most correlated with quality")
    print("4. Red and white wines have different characteristics")
    print("5. Ready to build ML models!")

if __name__ == "__main__":
    main()
