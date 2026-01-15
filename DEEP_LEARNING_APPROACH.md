# Deep Learning Approach for PIV → Bubble Prediction

## Why Deep Learning Makes Sense

### Current Approach (XGBoost with Aggregated Metrics)
- **Input**: 10 aggregated PIV metrics (mean velocity per quadrant, TKE, etc.)
- **Limitation**: Loses rich spatial and temporal structure
- **Information Loss**: 
  - Spatial patterns (vortices, flow structures, recirculation zones)
  - Temporal dynamics (how flow evolves over time)
  - Fine-grained spatial relationships

### Deep Learning Advantages

1. **Preserves Spatial Structure**
   - Can learn from raw velocity fields (u, v components)
   - Captures spatial patterns: vortices, jets, recirculation zones
   - Learns which spatial regions are most predictive

2. **Preserves Temporal Dynamics**
   - Models how flow evolves over time
   - Can capture bubble formation events (transient phenomena)
   - Learns temporal dependencies

3. **End-to-End Learning**
   - No manual feature engineering required
   - Model learns optimal features from data
   - Can discover complex relationships

4. **Rich Representation**
   - PIV data: 999 frames × 22 height × 30 width = rich 3D structure
   - Deep learning excels at high-dimensional structured data

## Model Architecture Options

### Option 1: 3D Convolutional Neural Network (3D CNN)

**Architecture:**
- Input: 3D volume (time × height × width) for u and v components
- 3D convolutions to capture spatiotemporal patterns
- Pooling layers to reduce dimensionality
- Fully connected layers for regression

**Pros:**
- Natural fit for 3D spatiotemporal data
- Captures both spatial and temporal patterns simultaneously
- Relatively simple architecture

**Cons:**
- Computationally expensive
- Requires significant data augmentation
- Less interpretable

**Example:**
```python
Input: (batch, 2, 999, 22, 30)  # 2 channels (u, v), 999 frames, 22×30 spatial
3D Conv layers → Feature maps
Global pooling → Fixed-size features
Dense layers → Bubble count prediction
```

### Option 2: CNN-LSTM Hybrid

**Architecture:**
- **CNN branch**: Extract spatial features from each frame
- **LSTM branch**: Model temporal evolution of spatial features
- Combine for final prediction

**Pros:**
- Separates spatial and temporal modeling (more interpretable)
- Can visualize spatial features learned by CNN
- LSTM explicitly models temporal dependencies

**Cons:**
- More complex architecture
- Two-stage training may be needed

**Example:**
```python
For each frame:
  CNN(u, v) → spatial_features (e.g., 256-dim vector)
  
Sequence of spatial_features → LSTM → temporal_features
temporal_features → Dense → Bubble count
```

### Option 3: Vision Transformer (ViT) for Spatiotemporal Data

**Architecture:**
- Treat PIV frames as image sequences
- Use transformer attention to model spatial and temporal relationships
- Self-attention learns which regions/time points are important

**Pros:**
- State-of-the-art for many vision tasks
- Attention mechanism provides interpretability
- Can handle variable-length sequences

**Cons:**
- Requires large amounts of data
- Computationally expensive
- Less proven for this specific domain

### Option 4: Graph Neural Network (GNN)

**Architecture:**
- Model flow field as a graph (nodes = spatial points, edges = flow connections)
- GNN learns node features and relationships
- Aggregate to predict bubble counts

**Pros:**
- Natural representation of flow physics
- Can incorporate physical constraints
- Interpretable (which nodes/regions matter)

**Cons:**
- More complex to implement
- Need to define graph structure
- Less common, fewer examples

## Recommended Approach: CNN-LSTM Hybrid

### Rationale

1. **Interpretability**: Can visualize what spatial features CNN learns
2. **Temporal Modeling**: LSTM explicitly handles time series
3. **Proven**: Well-established architecture for spatiotemporal data
4. **Flexibility**: Can start simple, add complexity

### Architecture Design

```
Input: PIV velocity fields
  Shape: (batch, time_steps, height, width, channels)
  Channels: u, v (or u, v, |V|, vorticity, etc.)

For each time step:
  CNN Encoder:
    - 2D Conv layers (spatial feature extraction)
    - Extract features: (batch, features_dim)
  
Sequence of features → LSTM:
  - Process temporal sequence
  - Output: (batch, lstm_hidden_dim)

Final layers:
  - Dense layers
  - Output: bubble_count_primary, bubble_count_secondary
```

### Implementation Considerations

1. **Data Preparation**
   - Load PIV frames (u, v) aligned to bubble count timestamps
   - Create sequences (e.g., 10-20 frames per sample)
   - Normalize velocity fields

2. **Architecture Details**
   - CNN: 2-3 conv layers with pooling
   - LSTM: 1-2 layers, bidirectional optional
   - Dense: 1-2 layers before output

3. **Training**
   - Loss: MSE or MAE for regression
   - Optimizer: Adam with learning rate scheduling
   - Regularization: Dropout, batch normalization
   - Data augmentation: Temporal shifts, slight spatial transforms

4. **Evaluation**
   - Time-based cross-validation
   - Compare to XGBoost baseline
   - Visualize learned features

## Comparison: Deep Learning vs XGBoost

| Aspect | XGBoost (Aggregated Metrics) | Deep Learning (Raw PIV) |
|--------|------------------------------|-------------------------|
| **Input** | 10 aggregated metrics | Raw velocity fields (999×22×30) |
| **Spatial Info** | Lost (aggregated) | Preserved |
| **Temporal Info** | Limited (per-timestamp) | Full sequence |
| **Feature Engineering** | Manual (metrics) | Automatic |
| **Interpretability** | High (feature importance) | Medium (attention/visualization) |
| **Data Requirements** | Lower | Higher |
| **Training Time** | Fast | Slower |
| **Inference** | Very fast | Fast |
| **Complexity** | Low | Higher |

## Hybrid Approach (Best of Both Worlds?)

### Option A: Two-Stage
1. **Deep Learning**: Extract rich features from PIV data
2. **XGBoost**: Predict bubbles from deep learning features
   - More interpretable than pure deep learning
   - Leverages XGBoost's strengths

### Option B: Ensemble
1. Train both XGBoost (on metrics) and Deep Learning (on raw data)
2. Combine predictions
   - XGBoost: Interpretable, fast
   - Deep Learning: Rich features
   - Ensemble: Best performance

## Recommendation

### Phase 1: Start with CNN-LSTM
- Implement CNN-LSTM hybrid model
- Use raw PIV velocity fields (u, v) as input
- Create sequences aligned to bubble count timestamps
- Compare to XGBoost baseline

### Phase 2: Enhance if Needed
- Add more input channels (velocity magnitude, vorticity)
- Try attention mechanisms
- Experiment with architecture variations

### Phase 3: Hybrid/Ensemble
- If deep learning performs well, use it
- If XGBoost is competitive, consider ensemble
- Use deep learning features with XGBoost for interpretability

## Implementation Plan

1. **Data Pipeline**
   - Load PIV frames aligned to bubble counts
   - Create sequence datasets
   - Normalize and preprocess

2. **Model Architecture**
   - Implement CNN-LSTM in PyTorch or TensorFlow
   - Start with simple architecture
   - Add complexity iteratively

3. **Training**
   - Set up training loop
   - Implement evaluation metrics
   - Add visualization tools

4. **Evaluation**
   - Compare to XGBoost baseline
   - Analyze learned features
   - Interpret model predictions

## Key Questions to Answer

1. **Sequence Length**: How many frames per sample? (10? 20? 50?)
2. **Spatial Resolution**: Use full 22×30 or downsample?
3. **Input Channels**: Just u, v? Or add |V|, vorticity, etc.?
4. **Temporal Alignment**: How to align PIV frames to bubble counts?
5. **Data Augmentation**: What augmentations make sense?

## Next Steps

1. Implement data loading for PIV sequences
2. Create CNN-LSTM architecture
3. Train baseline model
4. Compare to XGBoost approach
5. Iterate and improve
