"""
Generate visual metrics comparison tables and charts.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_heatmap_table(df, output_path):
    """Create a heatmap-style table visualization."""
    
    # Prepare data for heatmap
    scenarios = ['baseline_clean_train_clean_test', 
                 'baseline_clean_train_adv_test',
                 'adv_train_clean_test',
                 'adv_train_adv_test']
    
    scenario_labels = ['Clean Train\nClean Test', 
                       'Clean Train\nAdv Test',
                       'Adv Train\nClean Test',
                       'Adv Train\nAdv Test']
    
    models = ['SGDClassifier', 'RidgeClassifier']
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        
        # Prepare data matrix
        data_matrix = np.zeros((len(scenarios), len(metrics)))
        
        for i, scenario in enumerate(scenarios):
            scenario_df = df[(df['model'] == model) & (df['scenario'] == scenario)]
            if not scenario_df.empty:
                row = scenario_df.iloc[0]
                for j, metric in enumerate(metrics):
                    data_matrix[i, j] = row[metric]
        
        # Create heatmap
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0.6, vmax=1.0)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(scenarios)))
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.set_yticklabels(scenario_labels, fontsize=10)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        
        # Add text annotations
        for i in range(len(scenarios)):
            for j in range(len(metrics)):
                value = data_matrix[i, j]
                text = ax.text(j, i, f'{value:.3f}',
                             ha="center", va="center", color="black",
                             fontsize=10, fontweight='bold')
        
        ax.set_title(f'{model}', fontsize=14, fontweight='bold', pad=15)
        
        # Add grid
        ax.set_xticks(np.arange(len(metrics))-.5, minor=True)
        ax.set_yticks(np.arange(len(scenarios))-.5, minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', 
                       fraction=0.046, pad=0.15)
    cbar.set_label('Performance Score', fontsize=12, fontweight='bold')
    
    plt.suptitle('IDS Performance Metrics Comparison (with Feature Selection)',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap table to {output_path}")
    plt.close()


def create_metric_comparison_chart(df, output_path):
    """Create grouped bar chart for metric comparison."""
    
    scenarios = ['baseline_clean_train_clean_test', 
                 'baseline_clean_train_adv_test',
                 'adv_train_adv_test']
    
    scenario_labels = ['Clean→Clean', 'Clean→Adv', 'Adv→Adv']
    
    models = ['SGDClassifier', 'RidgeClassifier']
    metrics = ['accuracy', 'f1', 'auc']
    metric_labels = ['Accuracy', 'F1-Score', 'AUC']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    for idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        sgd_values = []
        ridge_values = []
        
        for scenario in scenarios:
            sgd_df = df[(df['model'] == 'SGDClassifier') & (df['scenario'] == scenario)]
            ridge_df = df[(df['model'] == 'RidgeClassifier') & (df['scenario'] == scenario)]
            
            if scenario == 'baseline_clean_train_adv_test':
                sgd_df = sgd_df[sgd_df['split'] == 'adv_test']
                ridge_df = ridge_df[ridge_df['split'] == 'adv_test']
            else:
                sgd_df = sgd_df[sgd_df['split'].isin(['clean_test', 'adv_test'])]
                ridge_df = ridge_df[ridge_df['split'].isin(['clean_test', 'adv_test'])]
            
            sgd_values.append(sgd_df[metric].values[0] if not sgd_df.empty else 0)
            ridge_values.append(ridge_df[metric].values[0] if not ridge_df.empty else 0)
        
        bars1 = ax.bar(x - width/2, sgd_values, width, label='SGDClassifier',
                      color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, ridge_values, width, label='RidgeClassifier',
                      color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Scenario', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_label} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0.5, 1.0])
    
    plt.suptitle('Scenario Performance Comparison (with Feature Selection)',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved metric comparison chart to {output_path}")
    plt.close()


def create_radar_chart(df, output_path):
    """Create radar chart for comprehensive metric view."""
    
    models = ['SGDClassifier', 'RidgeClassifier']
    scenarios = ['baseline_clean_train_clean_test', 
                 'baseline_clean_train_adv_test',
                 'adv_train_adv_test']
    scenario_labels = ['Clean→Clean', 'Clean→Adv', 'Adv→Adv']
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        
        for scenario, scenario_label, color in zip(scenarios, scenario_labels, colors):
            scenario_df = df[(df['model'] == model) & (df['scenario'] == scenario)]
            
            if scenario == 'baseline_clean_train_adv_test':
                scenario_df = scenario_df[scenario_df['split'] == 'adv_test']
            
            if not scenario_df.empty:
                values = [scenario_df[metric].values[0] for metric in metrics]
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=scenario_label, 
                       color=color, markersize=6)
                ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_ylim(0.5, 1.0)
        ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticklabels(['0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_title(f'{model}', fontsize=13, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    
    plt.suptitle('Comprehensive Metrics View (with Feature Selection)',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved radar chart to {output_path}")
    plt.close()


def main():
    # Load data
    metrics_file = Path('./results/metrics.csv')
    if not metrics_file.exists():
        print(f"Error: {metrics_file} not found")
        return
    
    df = pd.read_csv(metrics_file)
    output_dir = Path('./results')
    
    print("Generating visual metrics tables and charts...")
    print("=" * 80)
    
    # Generate visualizations
    create_heatmap_table(df, output_dir / 'metrics_heatmap.png')
    create_metric_comparison_chart(df, output_dir / 'metrics_comparison.png')
    create_radar_chart(df, output_dir / 'metrics_radar.png')
    
    print("=" * 80)
    print("✓ All visualizations generated successfully!")
    print("\nGenerated files:")
    print(f"  - {output_dir / 'metrics_heatmap.png'}")
    print(f"  - {output_dir / 'metrics_comparison.png'}")
    print(f"  - {output_dir / 'metrics_radar.png'}")


if __name__ == "__main__":
    main()
