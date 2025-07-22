#!/bin/bash
"""
🚀 LSTM Time Embedding Comparison Pipeline
==========================================

This script runs the complete pipeline for comparing LSTM variants with different time embeddings.
It includes training, evaluation, and detailed analysis.

Usage:
    bash run_lstm_comparison.sh

The pipeline includes:
1. Training all LSTM variants
2. Comprehensive analysis
3. Visualization generation
4. Results summary

Author: Generated for KAN-MAMMOTE Project
Date: July 2025
"""

echo "🚀 LSTM Time Embedding Comparison Pipeline"
echo "=========================================="

# Set up environment
export PYTHONPATH="${PYTHONPATH}:/home/s2516027/kan-mammote"
cd /home/s2516027/kan-mammote

# Create results directory
mkdir -p results/lstm_comparison

echo "📊 Step 1: Training all LSTM variants..."
echo "This may take 30-60 minutes depending on your hardware."
python lstm_embedding_comparison.py

if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed. Please check the logs."
    exit 1
fi

echo ""
echo "📊 Step 2: Running comprehensive analysis..."
python lstm_embedding_analysis.py

if [ $? -eq 0 ]; then
    echo "✅ Analysis completed successfully!"
else
    echo "❌ Analysis failed. Please check the logs."
    exit 1
fi

echo ""
echo "📊 Step 3: Results Summary"
echo "=========================="

# Display key results
if [ -f "results/lstm_comparison/summary.json" ]; then
    echo "🏆 Best performing models:"
    python -c "
import json
with open('results/lstm_comparison/summary.json', 'r') as f:
    data = json.load(f)
    
best_accs = data['best_accuracies']
sorted_models = sorted(best_accs.items(), key=lambda x: x[1], reverse=True)

for i, (model, acc) in enumerate(sorted_models, 1):
    print(f'{i}. {model}: {acc:.4f}')
"
else
    echo "❌ Summary file not found"
fi

echo ""
echo "📁 All results saved to: results/lstm_comparison/"
echo "📋 Key files created:"
echo "   - training_histories.json: Training curves data"
echo "   - summary.json: Overall results summary"
echo "   - results_summary.csv: Results in CSV format"
echo "   - detailed_analysis.json: Comprehensive analysis"
echo "   - analysis_report.md: Human-readable report"
echo "   - *.png: Visualization plots"
echo ""
echo "🎉 LSTM Time Embedding Comparison Complete!"
echo "Check the results directory for detailed analysis and visualizations."
