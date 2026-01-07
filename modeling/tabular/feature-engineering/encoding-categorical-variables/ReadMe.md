# Encoding Categorical Variables

## Summary

Encoding categorical variables transforms non-numeric data into numerical representations that machine learning algorithms can process. The choice of encoding method significantly impacts model performance and should consider the nature of the categorical variable (nominal vs ordinal), cardinality, relationship with target, and the algorithm being used. Different encodings capture different aspects of categorical information.

Key points to remember:

- One-hot encoding: Creates binary columns, standard for low cardinality
- Label encoding: Assigns integers, suitable for ordinal or tree-based models
- Target encoding: Uses target statistics, powerful but risk of leakage
- Frequency encoding: Uses category frequency, captures popularity
- Binary encoding: Compromise between one-hot and label encoding
- Embedding: Learned representations, best for high cardinality in neural networks
- Choose based on: Cardinality, model type, interpretability needs, target relationship

## Encoding Methods Overview

### Comparison Matrix

| Method | Best For | Cardinality | Leakage Risk | Interpretable |
|--------|----------|-------------|--------------|---------------|
| One-hot | Low cardinality, linear models | Low (<20) | None | High |
| Label | Ordinal, tree models | Any | None | Medium |
| Target | High cardinality, any model | High | High | Low |
| Frequency | High cardinality | High | None | Medium |
| Binary | Medium cardinality | Medium | None | Low |
| Embedding | Very high, neural networks | Very high | None | Low |

## One-Hot Encoding

Creates binary columns for each category.

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Pandas get_dummies (simple approach)
df_encoded = pd.get_dummies(df, columns=['city', 'color'])

# Sklearn OneHotEncoder (production-ready)
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[['city', 'color']])
feature_names = encoder.get_feature_names_out(['city', 'color'])

# Handle unseen categories
encoder = OneHotEncoder(
    sparse=False,
    handle_unknown='infrequent_if_exist',
    min_frequency=0.01  # Group rare categories
)
```

### Pros and Cons

```
Pros:
 No ordinal assumption
 Works with linear models
 Interpretable features
 No target leakage

Cons:
 Curse of dimensionality (high cardinality)
 Sparse matrices
 Cannot handle unseen categories (without special handling)
 Ignores relationships between categories
```

## Label Encoding

Assigns integer labels to categories.

```python
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Single column
le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])

# Multiple columns with OrdinalEncoder
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df[['city_enc', 'color_enc']] = oe.fit_transform(df[['city', 'color']])

# Ordinal with explicit ordering
categories = [['small', 'medium', 'large']]  # Ordered
oe = OrdinalEncoder(categories=categories)
df['size_encoded'] = oe.fit_transform(df[['size']])
```

### When to Use

```
Use Label Encoding when:
 Feature is ordinal (has natural ordering)
 Using tree-based models (XGBoost, LightGBM)
 High cardinality + tree model
 Memory constrained

Avoid when:
 Feature is nominal (no ordering)
 Using linear models (creates false relationships)
 Need interpretable coefficients
```

## Target Encoding

Replaces categories with target statistics.

```python
from category_encoders import TargetEncoder
import numpy as np

# Basic target encoding
encoder = TargetEncoder(cols=['city'])
df['city_target'] = encoder.fit_transform(df['city'], df['target'])

# Manual implementation with regularization
def target_encode(train_df, col, target, smoothing=10):
    """Target encoding with smoothing to prevent overfitting."""
    global_mean = train_df[target].mean()

    agg = train_df.groupby(col)[target].agg(['mean', 'count'])

    # Smoothed mean
    smoothing_factor = 1 / (1 + np.exp(-(agg['count'] - smoothing)))
    agg['smoothed_mean'] = (
        smoothing_factor * agg['mean'] +
        (1 - smoothing_factor) * global_mean
    )

    return train_df[col].map(agg['smoothed_mean'])

# K-fold target encoding (prevents leakage)
from sklearn.model_selection import KFold

def kfold_target_encode(df, col, target, n_splits=5):
    """Target encoding with out-of-fold predictions."""
    df['encoded'] = np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(df):
        # Calculate target mean on training fold
        means = df.iloc[train_idx].groupby(col)[target].mean()

        # Apply to validation fold
        df.loc[df.index[val_idx], 'encoded'] = df.iloc[val_idx][col].map(means)

    # Fill NaN with global mean
    df['encoded'].fillna(df[target].mean(), inplace=True)

    return df['encoded']
```

### Handling Target Leakage

```python
# Problem: Using test data target information
# Solution 1: K-fold encoding (as above)

# Solution 2: Leave-one-out encoding
from category_encoders import LeaveOneOutEncoder
encoder = LeaveOneOutEncoder(cols=['city'])
df['city_loo'] = encoder.fit_transform(df['city'], df['target'])

# Solution 3: Regularization with prior
# Blend category mean with global mean based on sample size
```

## Frequency Encoding

Replace categories with their frequency.

```python
# Frequency (count) encoding
freq_map = df['city'].value_counts().to_dict()
df['city_freq'] = df['city'].map(freq_map)

# Percentage encoding
freq_pct_map = df['city'].value_counts(normalize=True).to_dict()
df['city_freq_pct'] = df['city'].map(freq_pct_map)

# Rank encoding
rank_map = df['city'].value_counts().rank().to_dict()
df['city_rank'] = df['city'].map(rank_map)
```

### When to Use

```
Frequency encoding works well when:
 Frequency is predictive (popular items)
 High cardinality
 Want to avoid target leakage
 E-commerce, recommendation systems

Limitation:
 Categories with same frequency get same encoding
 Doesn't capture target relationship directly
```

## Binary Encoding

Represent category labels as binary numbers.

```python
from category_encoders import BinaryEncoder

# Binary encoding
encoder = BinaryEncoder(cols=['city'])
df_encoded = encoder.fit_transform(df)

# Manual implementation
def binary_encode(df, col):
    """Convert labels to binary representation."""
    labels = df[col].astype('category').cat.codes
    n_bits = int(np.ceil(np.log2(df[col].nunique())))

    for i in range(n_bits):
        df[f'{col}_bin_{i}'] = ((labels >> i) & 1).astype(int)

    return df
```

### Comparison with One-Hot

```
Category: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

One-hot (8 columns):
A: [1,0,0,0,0,0,0,0]
B: [0,1,0,0,0,0,0,0]
...
H: [0,0,0,0,0,0,0,1]

Binary (3 columns):
A: [0,0,0]
B: [0,0,1]
C: [0,1,0]
D: [0,1,1]
E: [1,0,0]
F: [1,0,1]
G: [1,1,0]
H: [1,1,1]

Columns needed: log‚(n_categories) vs n_categories
```

## Embedding Encoding

Learned representations for neural networks.

```python
import torch
import torch.nn as nn

class CategoricalEmbedding(nn.Module):
    def __init__(self, num_categories, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

# Rule of thumb for embedding dimension
def get_embedding_dim(cardinality):
    return min(50, (cardinality + 1) // 2)

# Multiple categorical embeddings
class MultiCatEmbedding(nn.Module):
    def __init__(self, category_dims, embedding_dims):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cat, emb_dim)
            for n_cat, emb_dim in zip(category_dims, embedding_dims)
        ])

    def forward(self, x):
        # x shape: [batch_size, num_cat_features]
        embeds = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(embeds, dim=1)
```

## Choosing the Right Encoding

```
                                                         
                  Categorical Variable                    
                         “                                
              Is it ordinal (ordered)?                    
              “                    “                      
            Yes                   No                      
             “                    “                       
     OrdinalEncoder        What's the cardinality?        
                           “         “         “          
                        Low      Medium      High         
                       (<20)    (20-100)   (>100)        
                         “         “         “           
                     One-hot    Binary    Target/        
                                        Frequency        
                                           “             
                                    Using neural net?    
                                    “           “        
                                  Yes          No        
                                   “            “        
                             Embedding    Target Enc     
                                                         
```

## Handling Unknown Categories

```python
from sklearn.preprocessing import OneHotEncoder

# Strategy 1: Ignore unknown categories
encoder = OneHotEncoder(handle_unknown='ignore')

# Strategy 2: Map to infrequent category
encoder = OneHotEncoder(
    handle_unknown='infrequent_if_exist',
    min_frequency=5
)

# Strategy 3: Explicit unknown value
encoder = OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=-1
)

# Strategy 4: Fill with most frequent or mode
def handle_unknown(train_df, test_df, col):
    known_cats = set(train_df[col].unique())
    most_frequent = train_df[col].mode()[0]
    test_df[col] = test_df[col].apply(
        lambda x: x if x in known_cats else most_frequent
    )
```

## Best Practices

### Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define column types
numerical_cols = ['age', 'income']
categorical_cols = ['city', 'occupation']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Full pipeline with model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
```

### Multiple Encodings

```python
# Sometimes combining encodings helps
df['city_freq'] = df['city'].map(df['city'].value_counts())
df['city_target'] = target_encode(df, 'city', 'target')
# Use both as features
```

## See Also

- [Feature Scaling](../feature-scaling/ReadMe.md) - Numerical feature preprocessing
- [Feature Selection](../feature-selection/ReadMe.md) - Selecting best features
- [Feature Engineering](../ReadMe.md) - Parent topic
