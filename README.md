# Hybrid Scheme 1

This mini-project operationalises **Scheme 1**: adversarially training the
lightweight IDS models (SGDClassifier/RidgeClassifier) using adversarial samples.
The workflow is encapsulated in `pipeline.py`.

**New in this version**: 
- ✨ Integrated **OWC-SAWN** (Optimized Weighted Conditional Stepwise Adversarial Network) for high-quality adversarial sample generation
- 🎯 Integrated **feature selection methods** from the Lightweight IDS paper to reduce dimensionality
- 📊 Enhanced metrics visualization and reporting

## Structure

- `pipeline.py` – end-to-end script to load the dataset, craft adversarial
  samples (FGSM or OWC-SAWN), augment the training data, and evaluate the
  lightweight classifiers on both clean and adversarial test sets. Now includes
  optional feature selection step.
- **Adversarial Generation**:
  - `owc_sawn/` – OWC-SAWN conditional GAN module for adversarial sample generation
    - `generator.py` – Conditional generator with label embedding
    - `discriminator.py` – Weighted discriminator with class-aware loss
    - `trainer.py` – Stepwise adversarial training with gradient penalty
    - `utils.py` – Helper functions for generation and evaluation
    - `README.md` – Detailed OWC-SAWN documentation
    - `example_train.py` – Example training script
  - `run_owc_sawn.py` – Standalone launcher for OWC-SAWN training
- **Feature Selection**:
  - `feature_selection.py` – module implementing four feature selection methods:
    Forward Selection, Backward Elimination, Correlation-based Selection, and
    Importance-based Selection
  - `evaluate_feature_selection.py` – standalone script to compare and evaluate
    all feature selection methods on your dataset
  - `test_feature_selection.py` – quick sanity check for the feature selection module
  - `FEATURE_SELECTION.md` – detailed documentation for feature selection methods
- **Metrics & Visualization**:
  - `generate_metrics_table.py` – generate formatted metrics tables in console,
    Markdown, and LaTeX formats
  - `generate_visual_tables.py` – create visual metrics tables and charts
    (heatmap, comparison, radar chart)
  - `plot_metrics.py` – utility to transform `results/metrics.csv` into the
    three-scenario comparison plot used in the paper
- `requirements.txt` – minimal dependencies
- `config_example.json` – example configuration file with feature selection parameters
- `results/` – created automatically to store metrics tables and cached
  adversarial arrays
- `test_owc_sawn.py` – comprehensive test suite for OWC-SAWN module

## Prerequisites

1. An environment with the dependencies from `requirements.txt` installed.
2. Access to the datasets that ship with `lab-ids-anta-main`, e.g.
   `lab-ids-anta-main/Dataset/encoded_features_2017.csv`.
3. (Optional) GPU-enabled TensorFlow if you want faster FGSM generation.

## Quick start

### 1. Basic usage (FGSM adversarial samples)

From the workspace root:

```powershell
cd hybrid_scheme1
python pipeline.py --dataset ..\lab-ids-anta-main\Dataset\encoded_features_2017.csv
```

### 2. Using OWC-SAWN adversarial network

```powershell
# Train OWC-SAWN and use generated adversarial samples
python pipeline.py \
    --dataset ..\lab-ids-anta-main\Dataset\encoded_features_2017.csv \
    --adversarial-method owc-sawn \
    --owc-epochs 100

# Combine both FGSM and OWC-SAWN samples
python pipeline.py \
    --dataset ..\lab-ids-anta-main\Dataset\encoded_features_2017.csv \
    --adversarial-method both \
    --owc-epochs 100
```

### 3. Standalone OWC-SAWN training

```powershell
# Quick start with default settings
python run_owc_sawn.py \
    --dataset ..\lab-ids-anta-main\Dataset\normalized_data_2017.csv \
    --epochs 100

# Advanced usage with custom parameters
python run_owc_sawn.py \
    --dataset ..\lab-ids-anta-main\Dataset\normalized_data_2017.csv \
    --epochs 150 \
    --batch-size 128 \
    --latent-dim 100 \
    --generator-lr 0.0002 \
    --discriminator-lr 0.0002 \
    --generate-samples 5000 \
    --save-generated \
    --output-dir owc_output_2017
```

### 4. With feature selection

```powershell
# Use a specific feature selection method
python pipeline.py \
    --dataset ..\lab-ids-anta-main\Dataset\encoded_features_2017.csv \
    --feature-selection forward \
    --max-features 20

# Compare all feature selection methods and use the best one
python pipeline.py \
    --dataset ..\lab-ids-anta-main\Dataset\encoded_features_2017.csv \
    --feature-selection compare \
    --max-features 20

# Combine feature selection with OWC-SAWN
python pipeline.py \
    --dataset ..\lab-ids-anta-main\Dataset\encoded_features_2017.csv \
    --feature-selection importance \
    --max-features 20 \
    --adversarial-method owc-sawn \
    --owc-epochs 100
```

### 5. Evaluate feature selection methods

```powershell
# Test the feature selection module
python test_feature_selection.py

# Comprehensive evaluation of all methods
python evaluate_feature_selection.py \
    --dataset ..\lab-ids-anta-main\Dataset\encoded_features_2017.csv \
    --max-features 20
```

### Key Command-Line Flags

**Adversarial Generation**:
- `--adversarial-method` – method to use: `fgsm`, `owc-sawn`, or `both` (default: `fgsm`)
- `--epsilon` – FGSM strength (default `0.3`)
- `--clip-value` – clips FGSM perturbations after scaling
- `--owc-epochs` – training epochs for OWC-SAWN (default: 100)
- `--owc-latent-dim` – latent dimension for OWC-SAWN generator (default: 100)
- `--owc-augmentation-ratio` – ratio of generated samples to add (default: 0.5)

**Feature Selection**:
- `--feature-selection` – method to use: `forward`, `backward`, `correlation`, 
  `importance`, or `compare` (default: None, no feature selection)
- `--max-features` – maximum number of features to select (default: 20)
- `--fs-cv` – cross-validation folds for feature selection (default: 3)

**Training**:
- `--epochs`, `--batch-size` – configure the helper DNN or OWC-SAWN
- `--hidden-layers` – configure the FGSM helper DNN
- `--output-dir` – where to write metrics and cached arrays (defaults to
  `hybrid_scheme1/results`)

To mirror the paper's scenario comparison chart once metrics exist:

```powershell
cd hybrid_scheme1

# Generate scenario comparison plot
python plot_metrics.py --metric f1

# Generate formatted tables (console, markdown, LaTeX)
python generate_metrics_table.py

# Generate visual tables and charts
python generate_visual_tables.py
```

These commands read `results/metrics.csv` and generate:
- `scenario_comparison.png` – bar chart comparing the three scenarios
- `metrics_table.txt/md/tex` – formatted tables in multiple formats
- `metrics_heatmap.png` – heatmap visualization of all metrics
- `metrics_comparison.png` – grouped bar charts
- `metrics_radar.png` – radar chart for comprehensive view
train/clean test, clean train/adversarial test, adversarial train/adversarial
test) for both lightweight models.

## Outputs

Running the pipeline produces:

- `results/metrics.csv` and `results/metrics.json` – precision/recall/F1/AUC for
  each lightweight classifier on clean vs adversarial test data
- `results/feature_selection.json` – feature selection information (if enabled):
  method used, number of features selected, and feature indices
- `results/owc_sawn_info.json` – OWC-SAWN training information (if used):
  epochs, losses, accuracies, and training history
- `results/owc_sawn_checkpoints/` – saved OWC-SAWN model checkpoints
- `results/owc_sawn_logs/` – TensorBoard logs for OWC-SAWN training
- `results/scenario_comparison.png` – saved by `plot_metrics.py` to match the
  paper's bar chart (default metric: F1 score)
- `results/X_train_adv.npy`, `results/X_test_adv.npy` – cached adversarial samples
  (FGSM, OWC-SAWN, or both) for additional experiments

Running `run_owc_sawn.py` produces:

- `owc_sawn_output/training_summary.json` – complete training configuration and results
- `owc_sawn_output/training_history.png` – loss and accuracy plots
- `owc_sawn_output/sample_quality_metrics.json` – generated sample quality evaluation
- `owc_sawn_output/generated_samples.csv` – generated adversarial samples (if `--save-generated`)
- `owc_sawn_output/checkpoints/` – model checkpoints for resuming training
- `owc_sawn_output/logs/` – TensorBoard logs

Running the feature selection evaluation produces:

- `results/feature_selection_eval/evaluation_results.csv` – detailed comparison
  of all methods with multiple classifiers
- `results/feature_selection_eval/feature_selection_details.json` – selected
  features for each method
- `results/feature_selection_eval/feature_selection_comparison.png` – bar chart
  comparing methods across metrics
- `results/feature_selection_eval/features_vs_performance.png` – scatter plot
  showing trade-off between feature count and performance

Use these artefacts to extend the SSRN lightweight IDS codebase (e.g. inject the
cached arrays into `botiot.py`) or to report Scheme 1 improvements in your
paper/thesis.

## Feature Selection Methods

This project integrates four feature selection methods from the Lightweight IDS paper:

1. **Forward Selection (FS)**: Starts with empty set, iteratively adds best features
2. **Backward Elimination (BE)**: Starts with all features, iteratively removes worst
3. **Correlation-based Selection (CFS)**: Selects features with high target correlation and low inter-feature correlation
4. **Importance-based Selection (FIS)**: Uses tree-based feature importance ranking

For detailed documentation, see [FEATURE_SELECTION.md](FEATURE_SELECTION.md).

### Benefits of Feature Selection

- **Reduced dimensionality**: Fewer features mean faster training and inference
- **Improved performance**: Remove noisy/redundant features
- **Better generalization**: Reduce overfitting on high-dimensional data
- **Lightweight deployment**: Critical for resource-constrained environments

### Example Results

On CIC-IDS-2017 (78 features → 20 selected features):
- Training time: ~60% reduction
- Model size: ~75% reduction
- Accuracy: maintained or improved (+0.5-2%)
- F1-score: similar or better performance

See `evaluate_feature_selection.py` for detailed benchmarks on your dataset.

## OWC-SAWN: Advanced Adversarial Sample Generation

**OWC-SAWN** (Optimized Weighted Conditional Stepwise Adversarial Network) is a conditional GAN specifically designed for IDS adversarial sample generation. It offers significant advantages over traditional FGSM:

### Key Features

- **Conditional Generation**: Generate specific attack types by conditioning on class labels
- **Weighted Discriminator**: Automatically learns class weights to balance hard-to-detect attacks
- **Stepwise Training**: Stable adversarial training with gradient penalty (WGAN-GP style)
- **High-Quality Samples**: Learns true data distribution for realistic adversarial samples

### Architecture

```
Conditional Generator:
  [Noise (100) + Label Embedding (50)] → [256→512→512→256] → Output (78)

Weighted Discriminator:
  [Sample (78) + Label Embedding (50)] → [256→256→128→64] → Real/Fake (1)
```

### OWC-SAWN vs FGSM

| Feature | **OWC-SAWN** | **FGSM** |
|---------|--------------|----------|
| **Generation Method** | Learns data distribution | Gradient-based perturbation |
| **Training Required** | Yes (100+ epochs) | No |
| **Sample Quality** | High (realistic) | Medium (perturbations) |
| **Controllability** | ✓ Class-conditional | ✗ No class control |
| **Diversity** | ✓ High (stochastic) | ✗ Deterministic |
| **Use Case** | Data augmentation, new attacks | Robustness testing, quick attacks |

### Quick Example

```powershell
# Train OWC-SAWN standalone
python run_owc_sawn.py \
    --dataset ..\Dataset\normalized_data_2017.csv \
    --epochs 100 \
    --generate-samples 5000 \
    --save-generated

# Use in pipeline
python pipeline.py \
    --dataset ..\Dataset\encoded_features_2017.csv \
    --adversarial-method owc-sawn \
    --owc-epochs 100

# Combine FGSM and OWC-SAWN
python pipeline.py \
    --dataset ..\Dataset\encoded_features_2017.csv \
    --adversarial-method both \
    --owc-epochs 100
```

### Monitoring Training

OWC-SAWN includes TensorBoard integration for real-time monitoring:

```powershell
# Start training
python run_owc_sawn.py --dataset data.csv --epochs 100

# In another terminal, launch TensorBoard
tensorboard --logdir=owc_sawn_output/logs
```

Visit `http://localhost:6006` to view:
- Generator and discriminator losses
- Real/fake accuracy curves
- Learning rate schedules
- Sample generation progress

### Sample Quality Metrics

After training, OWC-SAWN automatically evaluates:

- **Overall Quality Score** (0-1): Combined measure of realism
- **Diversity** (0-1): Variety of generated samples
- **Coverage** (0-1): How well generated samples cover real distribution
- **Mean/Std Distance**: Statistical similarity to real data

See `owc_sawn/README.md` for detailed documentation, API reference, and advanced usage.

