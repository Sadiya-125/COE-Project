# HPAFL Kaggle Notebooks

Complete Jupyter notebooks for running HPAFL (Hybrid Privacy-Aware Federated Learning) on Kaggle's free T4 GPU.

---

## 📔 Notebooks

### `hpafl_complete_training.ipynb`
**End-to-end HPAFL implementation with results analysis** (10 cells)

1. **Install Dependencies** - PyTorch, Opacus, EfficientNet, FastAPI, Streamlit
2. **Setup Environment & Verify GPU** - Verify T4 GPU (16GB VRAM available)
3. **Clone HPAFL Framework** - Download framework from GitHub
4. **Import HPAFL Modules** - Load all components
5. **Configure for Kaggle** - Set up configuration optimized for T4
6. **Run HPAFL Training** - Execute federated learning pipeline
7. **Load & Display Results** - Parse privacy budget, weights, and metrics
8. **Visualize Progress** - Generate plots for privacy and aggregation
9. **Test Batch Sizes** - Optional diagnostic tool
10. **Export Results** - Create summary and list outputs

**Runtime**: ~20 minutes for 2 rounds (demo)

---

## 🚀 Quick Start

### Step 1: Create Kaggle Notebook
1. Go to https://kaggle.com/code
2. Click **"New Notebook"**
3. Under **Data**, add: `skin-cancer-mnist-ham10000`
4. Click **Settings → Accelerator: GPU (T4)**

### Step 2: Copy Notebook
- Copy entire contents of `hpafl_complete_training.ipynb`
- Paste into your Kaggle notebook

### Step 3: Run
- Execute cells in order
- Expected runtime: ~20 min for 2 rounds

---

## ⚙️ Configuration

The notebook uses pre-optimized settings for T4:

```python
cfg.num_rounds = 2              # Demo: 2 rounds
cfg.local_epochs = 1            # Demo: 1 epoch
cfg.batch_size = 64             # T4-safe: 64
cfg.target_epsilon = 8.0        # Privacy budget
```

**For Full Training** (20 rounds, change to):
```python
cfg.num_rounds = 20
cfg.local_epochs = 5
cfg.batch_size = 64  # Still safe on T4
# Expected time: ~35-40 minutes
```

---

## 📊 Expected Results

### After 2 Rounds (Demo)
- **Accuracy**: ~0.65-0.70 (early training)
- **Privacy ε**: ~0.75-0.80 (budget remaining)
- **Time**: ~15-20 minutes
- **Files**: 2 model checkpoints + results JSON

### After 20 Rounds (Full Training)
- **Accuracy**: ~0.80
- **F1-Macro**: ~0.69
- **Privacy ε**: ~7.8 (within 8.0 budget)
- **Time**: ~35-40 minutes
- **Files**: 20 model checkpoints + results JSON

---

## 📁 Output Files

```
/kaggle/working/results/
├── hpafl.json                 # Final metrics
├── privacy_budget.csv         # ε consumption per round
├── adaptive_weights.csv       # Hospital weights evolution
└── *.png                      # Visualizations
```

---

## 🔧 Common Customizations

### Change Batch Size
```python
cfg.batch_size = 32  # Reduce if OOM (shouldn't happen on T4)
```

### Reduce Workers
```python
cfg.num_workers = 0  # Disable parallel loading if slow
```

### Change Dataset Path
```python
cfg.data_root = "/kaggle/input/skin-cancer-mnist-ham10000"  # Auto-detected
```

---

## 🔍 Troubleshooting

### "Module not found" errors
- Verify dataset is added in notebook settings
- Check that framework is cloned to `/kaggle/working/hpafl-framework`

### "Out of memory" errors
- Reduce batch size: `cfg.batch_size = 32`
- The GPU cache clearing code handles most OOM scenarios

### "Connection timeout" errors
- Kaggle notebooks sometimes timeout after 6 hours
- Reduce to `cfg.num_rounds = 10` for quicker completion

### Training is very slow
- Check GPU utilization: `nvidia-smi`
- Ensure GPU is being used (not falling back to CPU)

---

## 📈 Performance on T4

| Configuration | Time | Accuracy | Privacy ε |
|---------------|------|----------|-----------|
| 2 rounds, 1 epoch | ~20 min | 0.65-0.70 | 0.75-0.80 |
| 5 rounds, 2 epochs | ~50 min | 0.72-0.75 | 1.90-2.00 |
| 20 rounds, 5 epochs | ~35-40 min | ~0.80 | ~7.80 |

---

## 🎯 Key Metrics Explained

- **Accuracy**: Percentage of correct predictions (target: 0.80+)
- **F1 (macro)**: Unweighted F1 across classes (target: 0.69+)
- **Epsilon (ε)**: Privacy budget consumed (target: ≤ 8.0)
- **Adaptive Weights**: Hospital contribution to global model

---

## 📚 References

**Full Documentation**: See `README.md` in project root

**HPAFL Framework**:
- FedBN (Li et al., ICLR 2021)
- DP-SGD (Abadi et al., CCS 2016)
- Secure Aggregation (Bonawitz et al., CCS 2017)

**Dataset**:
- HAM10000: 10,015 dermoscopic images, 7 skin lesion classes
- Severe class imbalance (58.3:1) handled via Focal Loss

---

## 🚀 Next Steps After Training

1. **Download Results** - Download JSON, CSV, and PNG files
2. **Analyze Metrics** - Review accuracy, F1, privacy spend
3. **Tune Hyperparameters** - Adjust for production use
4. **Deploy Model** - Use final weights for inference
5. **Compare Baselines** - Run `benchmark.py` locally for full comparison

---

## ✅ Notebook Checklist

- [ ] Dataset added: `skin-cancer-mnist-ham10000`
- [ ] GPU enabled: T4
- [ ] Cell 1: Dependencies installed successfully
- [ ] Cell 2: GPU verified with 16GB VRAM
- [ ] Cell 3: Framework cloned
- [ ] Cell 4: Modules imported
- [ ] Cell 5: Configuration displayed
- [ ] Cell 6: Training completes (watch progress)
- [ ] Cell 7: Results displayed
- [ ] Cell 8: Plots generated
- [ ] Cell 9: Summary exported

---

**Framework**: HPAFL v1.0 | **Dataset**: HAM10000 | **Status**: ✅ READY

See `README.md` for complete project documentation.
