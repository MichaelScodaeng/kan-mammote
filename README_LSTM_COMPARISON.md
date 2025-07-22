# 🔬 LSTM Time Embedding Comparison Framework

This framework provides a comprehensive comparison of different LSTM variants with various time embeddings on the Event-Based MNIST dataset.

## 🎯 Overview

The framework compares four different approaches:

1. **Baseline LSTM**: Simple learnable position embedding
2. **SinCos LSTM**: Sinusoidal/Cosine position encoding (Transformer-style)
3. **LETE LSTM**: Learning Time Embedding (LeTE) with Fourier and Spline components
4. **KAN-MAMMOTE LSTM**: Advanced KAN-MAMMOTE time embedding with mixture of experts

## 📊 What Gets Compared

- **Accuracy**: Classification performance on MNIST digits
- **Training Speed**: Time per epoch and convergence speed
- **Parameter Count**: Model complexity
- **Training Stability**: Variance in performance
- **Temporal Modeling**: How well each method captures temporal patterns
- **Statistical Significance**: Rigorous comparison of performances

## 🚀 Quick Start

### Option 1: Run Complete Pipeline

```bash
# Run everything with default settings
bash run_lstm_comparison.sh
```

### Option 2: Run Components Separately

```bash
# 1. Train all models
python lstm_embedding_comparison.py

# 2. Run detailed analysis
python lstm_embedding_analysis.py
```

### Option 3: Custom Configuration

```bash
# Edit configuration file
nano config/lstm_comparison_config.json

# Run with custom config
python lstm_embedding_comparison.py --config config/lstm_comparison_config.json
```

## 📁 Output Structure

```
results/lstm_comparison/
├── training_histories.json      # Training curves for all models
├── summary.json                 # Overall results summary
├── results_summary.csv          # Results in CSV format
├── detailed_analysis.json       # Comprehensive analysis
├── analysis_report.md           # Human-readable report
├── training_curves.png          # Training/test curves
├── convergence_analysis.png     # Convergence properties
├── temporal_patterns.png        # Temporal modeling analysis
├── confusion_matrices.png       # Confusion matrices
├── statistical_significance.png # Statistical significance tests
├── *_best.pth                   # Best model checkpoints
└── logs/                        # Detailed training logs
```

## 🔧 Configuration Options

### Training Parameters
- `batch_size`: Training batch size (default: 128)
- `learning_rate`: Learning rate (default: 0.001)
- `num_epochs`: Training epochs (default: 50)
- `dropout_rate`: Dropout rate (default: 0.2)

### Model Architecture
- `lstm_hidden_dim`: LSTM hidden dimension (default: 128)
- `lstm_num_layers`: Number of LSTM layers (default: 2)
- `time_embedding_dim`: Time embedding dimension (default: 32)

### Data Processing
- `threshold`: MNIST pixel threshold for events (default: 0.9)
- `normalize`: Whether to normalize inputs (default: true)

### KAN-MAMMOTE Specific
- `num_experts`: Number of K-MOTE experts (default: 4)
- `hidden_dim_mamba`: Mamba hidden dimension (default: 32)
- `kan_grid_size`: KAN grid size (default: 5)

### LETE Specific
- `p`: Fourier vs Spline ratio (default: 0.5)
- `layer_norm`: Use layer normalization (default: true)
- `scale`: Use learnable scaling (default: true)

## 📊 Understanding the Results

### Key Metrics

1. **Best Accuracy**: Highest test accuracy achieved during training
2. **Final Accuracy**: Test accuracy at the end of training
3. **Convergence Epoch**: When the model stops improving significantly
4. **Training Stability**: Standard deviation in later epochs (lower is better)
5. **Parameter Count**: Total trainable parameters

### Analysis Components

1. **Training Curves**: Loss and accuracy over time
2. **Convergence Analysis**: When and how models converge
3. **Temporal Patterns**: How well models handle different sequence lengths
4. **Statistical Significance**: Whether differences are statistically meaningful
5. **Error Analysis**: Detailed breakdown of model failures

### Interpreting Results

- **Higher accuracy**: Better model performance
- **Lower convergence epoch**: Faster training
- **Lower stability value**: More consistent training
- **Better length-accuracy correlation**: Better temporal modeling
- **p-value < 0.05**: Statistically significant difference

## 🎨 Visualization Guide

### Training Curves
- **Solid lines**: Training performance
- **Dashed lines**: Test performance
- **Gap between lines**: Indication of overfitting

### Convergence Analysis
- **Convergence Epochs**: How quickly models reach stable performance
- **Overfitting Analysis**: Training vs test accuracy gap
- **Training Stability**: Consistency in final epochs

### Temporal Patterns
- **Accuracy by Length**: How performance varies with sequence length
- **Length-Accuracy Correlation**: Linear relationship strength
- **Sample Distribution**: How many samples per length

### Statistical Significance
- **Dark cells**: Significant differences (p < 0.05)
- **Light cells**: Non-significant differences
- **Diagonal**: Always non-significant (model vs itself)

## 🔍 Advanced Usage

### Running Ablation Studies

```python
# Modify config for ablation study
config = {
    "time_embedding_dim": [16, 32, 64],  # Test different dimensions
    "lstm_hidden_dim": [64, 128, 256],   # Test different LSTM sizes
    "num_experts": [2, 4, 8]             # Test different expert counts
}

# Run multiple experiments
python run_ablation_study.py --config config.json
```

### Adding New Models

```python
# In lstm_embedding_comparison.py
class MyCustomLSTM(nn.Module):
    def __init__(self, ...):
        # Your custom architecture
        pass
    
    def forward(self, events, features, lengths):
        # Your forward pass
        return logits

# Add to models dictionary
models = {
    'Baseline_LSTM': BaselineLSTM(),
    'SinCos_LSTM': SinCosLSTM(),
    'LETE_LSTM': LETE_LSTM(),
    'KAN_MAMMOTE_LSTM': KAN_MAMMOTE_LSTM(),
    'MyCustom_LSTM': MyCustomLSTM()  # Add your model
}
```

### Custom Metrics

```python
# In lstm_embedding_analysis.py
def my_custom_analysis(model, test_loader):
    # Your custom analysis logic
    return analysis_results

# Add to analysis pipeline
analyses = {
    'convergence': analyze_convergence,
    'temporal': analyze_temporal_patterns,
    'custom': my_custom_analysis  # Add your analysis
}
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config
   - Reduce model dimensions
   - Use gradient accumulation

2. **Slow Training**
   - Increase batch size if memory allows
   - Reduce sequence length
   - Use mixed precision training

3. **Poor Convergence**
   - Reduce learning rate
   - Add learning rate scheduler
   - Increase regularization

4. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path configuration
   - Verify model imports

### Performance Tips

- **GPU Usage**: Ensure CUDA is available and models are on GPU
- **Data Loading**: Use multiple workers for faster data loading
- **Memory Management**: Use gradient checkpointing for large models
- **Reproducibility**: Set random seeds for consistent results

## 📚 References

- **LETE**: Learning Time Embedding paper
- **KAN-MAMMOTE**: KAN-MAMMOTE architecture paper
- **Event-Based MNIST**: Event-based vision processing
- **Temporal Modeling**: LSTM and RNN temporal processing

## 🤝 Contributing

To add new time embedding methods:

1. Create a new model class inheriting from `nn.Module`
2. Implement the time embedding in the `forward` method
3. Add the model to the comparison dictionary
4. Update the configuration file
5. Add any new analysis functions

## 📄 License

This code is part of the KAN-MAMMOTE project and follows the same license terms.

---

**Happy Experimenting!** 🚀
