"""
Feature Selection Evaluation Script

This script evaluates and compares different feature selection methods
on the IDS dataset. It helps determine which method works best for your
specific dataset and classifier combination.

Usage:
    python evaluate_feature_selection.py --dataset ../lab-ids-anta-main/Dataset/encoded_features_2017.csv
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from feature_selection import (
    ForwardSelection,
    BackwardElimination,
    CorrelationBasedSelection,
    ImportanceBasedSelection,
    compare_feature_selection_methods,
)


def evaluate_on_classifiers(X_train, X_test, y_train, y_test, 
                           selected_features, method_name):
    """Evaluate selected features on multiple classifiers."""
    results = []
    
    # Select features
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    
    # Test with SGDClassifier
    sgd = SGDClassifier(loss='log_loss', max_iter=2000, random_state=42)
    sgd.fit(X_train_selected, y_train)
    y_pred = sgd.predict(X_test_selected)
    
    results.append({
        'method': method_name,
        'classifier': 'SGDClassifier',
        'n_features': len(selected_features),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    })
    
    # Test with RidgeClassifier
    ridge = RidgeClassifier()
    ridge.fit(X_train_selected, y_train)
    y_pred = ridge.predict(X_test_selected)
    
    results.append({
        'method': method_name,
        'classifier': 'RidgeClassifier',
        'n_features': len(selected_features),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    })
    
    return results


def plot_comparison(results_df, output_dir):
    """Create comparison plots for feature selection methods."""
    methods = results_df['method'].unique()
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Plot 1: Metrics comparison by method
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data = results_df.pivot_table(
            values=metric, 
            index='method', 
            columns='classifier'
        )
        data.plot(kind='bar', ax=ax)
        ax.set_title(f'{metric.capitalize()} by Method', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.capitalize())
        ax.set_xlabel('Feature Selection Method')
        ax.legend(title='Classifier')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_selection_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_dir / 'feature_selection_comparison.png'}")
    plt.close()
    
    # Plot 2: Feature count vs Performance
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in methods:
        method_data = results_df[results_df['method'] == method]
        # Average across classifiers
        avg_acc = method_data.groupby('n_features')['accuracy'].mean()
        n_features = method_data['n_features'].unique()[0]
        ax.scatter(n_features, avg_acc.values[0], s=150, label=method, alpha=0.7)
    
    ax.set_xlabel('Number of Selected Features', fontsize=12)
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title('Feature Count vs Performance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'features_vs_performance.png', dpi=300, bbox_inches='tight')
    print(f"Saved features vs performance plot to {output_dir / 'features_vs_performance.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate feature selection methods')
    parser.add_argument(
        '--dataset',
        type=Path,
        required=True,
        help='Path to CSV dataset'
    )
    parser.add_argument(
        '--label-column',
        default='Label',
        help='Name of the label column'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test split size'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--max-features',
        type=int,
        default=20,
        help='Maximum features to select'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./results/feature_selection_eval'),
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}")
    df = pd.read_csv(args.dataset)
    
    if args.label_column not in df.columns:
        raise ValueError(f"Column '{args.label_column}' not found in dataset")
    
    X = df.drop(columns=[args.label_column]).values.astype(np.float32)
    y = df[args.label_column].values.astype(np.int32)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    # Create classifier for feature selection
    clf = SGDClassifier(loss='log_loss', max_iter=1000, random_state=args.random_state)
    
    # Compare all methods
    print("\n" + "="*80)
    print("COMPARING FEATURE SELECTION METHODS")
    print("="*80)
    
    fs_results = compare_feature_selection_methods(
        X_train_scaled,
        y_train,
        clf,
        cv=3,
        verbose=True
    )
    
    # Evaluate each method on test set with multiple classifiers
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    all_results = []
    
    for method_name, fs_result in fs_results.items():
        print(f"\n{method_name.upper()}")
        print("-" * 80)
        selected_features = fs_result['features']
        
        results = evaluate_on_classifiers(
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            selected_features,
            method_name
        )
        
        all_results.extend(results)
        
        for result in results:
            print(f"  {result['classifier']}: "
                  f"Acc={result['accuracy']:.4f}, "
                  f"Prec={result['precision']:.4f}, "
                  f"Rec={result['recall']:.4f}, "
                  f"F1={result['f1']:.4f}")
    
    # Also evaluate baseline (no feature selection)
    print(f"\nBASELINE (All {X_train_scaled.shape[1]} features)")
    print("-" * 80)
    baseline_results = evaluate_on_classifiers(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        list(range(X_train_scaled.shape[1])),
        'baseline'
    )
    all_results.extend(baseline_results)
    
    for result in baseline_results:
        print(f"  {result['classifier']}: "
              f"Acc={result['accuracy']:.4f}, "
              f"Prec={result['precision']:.4f}, "
              f"Rec={result['recall']:.4f}, "
              f"F1={result['f1']:.4f}")
    
    # Create DataFrame and save results
    results_df = pd.DataFrame(all_results)
    results_csv = output_dir / 'evaluation_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved results to {results_csv}")
    
    # Save detailed feature selection info
    fs_info = {}
    for method_name, fs_result in fs_results.items():
        fs_info[method_name] = {
            'n_features': fs_result['n_features'],
            'selected_features': [int(f) for f in fs_result['features']],
            'cv_score': float(fs_result['score'])
        }
    
    fs_json = output_dir / 'feature_selection_details.json'
    fs_json.write_text(json.dumps(fs_info, indent=2))
    print(f"Saved feature selection details to {fs_json}")
    
    # Create visualizations
    plot_comparison(results_df, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nAverage Performance by Method:")
    summary = results_df.groupby('method').agg({
        'n_features': 'first',
        'accuracy': 'mean',
        'f1': 'mean'
    }).sort_values('accuracy', ascending=False)
    print(summary.to_string())
    
    best_method = summary.index[0]
    print(f"\nBest method: {best_method}")
    print(f"Features selected: {fs_info.get(best_method, {}).get('n_features', 'N/A')}")
    print(f"Average accuracy: {summary.loc[best_method, 'accuracy']:.4f}")


if __name__ == "__main__":
    main()
