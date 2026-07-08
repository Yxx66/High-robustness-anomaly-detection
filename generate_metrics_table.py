"""
Generate formatted metrics table from pipeline results.

This script reads metrics.csv and generates formatted tables in multiple formats:
- Console output (pretty table)
- Markdown table
- LaTeX table
- Excel file (optional)
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def format_percentage(value, decimals=2):
    """Format value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def format_metrics_table(df, format_type='console'):
    """
    Format metrics table in different styles.
    
    Args:
        df: DataFrame with metrics
        format_type: 'console', 'markdown', 'latex'
    """
    # Pivot table for better presentation
    pivot_data = []
    
    for scenario in df['scenario'].unique():
        scenario_df = df[df['scenario'] == scenario]
        for model in scenario_df['model'].unique():
            model_df = scenario_df[scenario_df['model'] == model]
            for _, row in model_df.iterrows():
                pivot_data.append({
                    'Scenario': scenario,
                    'Model': model,
                    'Split': row['split'],
                    'Accuracy': row['accuracy'],
                    'Precision': row['precision'],
                    'Recall': row['recall'],
                    'F1-Score': row['f1'],
                    'AUC': row['auc']
                })
    
    pivot_df = pd.DataFrame(pivot_data)
    
    if format_type == 'console':
        return format_console_table(pivot_df)
    elif format_type == 'markdown':
        return format_markdown_table(pivot_df)
    elif format_type == 'latex':
        return format_latex_table(pivot_df)
    else:
        return str(pivot_df)


def format_console_table(df):
    """Format as console-friendly table."""
    output = []
    output.append("\n" + "="*120)
    output.append("METRICS SUMMARY")
    output.append("="*120)
    
    # Group by scenario
    for scenario in df['Scenario'].unique():
        scenario_df = df[df['Scenario'] == scenario]
        
        output.append(f"\n{scenario.upper()}")
        output.append("-"*120)
        output.append(f"{'Model':<20} {'Split':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC':<12}")
        output.append("-"*120)
        
        for _, row in scenario_df.iterrows():
            output.append(
                f"{row['Model']:<20} "
                f"{row['Split']:<15} "
                f"{format_percentage(row['Accuracy']):<12} "
                f"{format_percentage(row['Precision']):<12} "
                f"{format_percentage(row['Recall']):<12} "
                f"{format_percentage(row['F1-Score']):<12} "
                f"{format_percentage(row['AUC']):<12}"
            )
    
    output.append("="*120)
    return "\n".join(output)


def format_markdown_table(df):
    """Format as Markdown table."""
    output = []
    output.append("\n# Metrics Summary\n")
    
    for scenario in df['Scenario'].unique():
        scenario_df = df[df['Scenario'] == scenario]
        
        output.append(f"\n## {scenario}\n")
        output.append("| Model | Split | Accuracy | Precision | Recall | F1-Score | AUC |")
        output.append("|-------|-------|----------|-----------|--------|----------|-----|")
        
        for _, row in scenario_df.iterrows():
            output.append(
                f"| {row['Model']} "
                f"| {row['Split']} "
                f"| {format_percentage(row['Accuracy'])} "
                f"| {format_percentage(row['Precision'])} "
                f"| {format_percentage(row['Recall'])} "
                f"| {format_percentage(row['F1-Score'])} "
                f"| {format_percentage(row['AUC'])} |"
            )
    
    return "\n".join(output)


def format_latex_table(df):
    """Format as LaTeX table."""
    # Create pivot data first
    pivot_data = []
    for scenario in df['scenario'].unique():
        scenario_df = df[df['scenario'] == scenario]
        for model in scenario_df['model'].unique():
            model_df = scenario_df[scenario_df['model'] == model]
            for _, row in model_df.iterrows():
                pivot_data.append({
                    'Scenario': scenario,
                    'Model': model,
                    'Split': row['split'],
                    'Accuracy': row['accuracy'],
                    'Precision': row['precision'],
                    'Recall': row['recall'],
                    'F1-Score': row['f1'],
                    'AUC': row['auc']
                })
    
    pivot_df = pd.DataFrame(pivot_data)
    
    output = []
    output.append("\\begin{table}[htbp]")
    output.append("\\centering")
    output.append("\\caption{IDS Performance Metrics with Feature Selection}")
    output.append("\\label{tab:metrics}")
    output.append("\\begin{tabular}{llcccccc}")
    output.append("\\hline")
    output.append("\\textbf{Scenario} & \\textbf{Model} & \\textbf{Split} & \\textbf{Acc.} & \\textbf{Prec.} & \\textbf{Rec.} & \\textbf{F1} & \\textbf{AUC} \\\\")
    output.append("\\hline")
    
    for scenario in pivot_df['Scenario'].unique():
        scenario_df = pivot_df[pivot_df['Scenario'] == scenario]
        first = True
        
        for _, row in scenario_df.iterrows():
            scenario_text = scenario if first else ""
            first = False
            
            output.append(
                f"{scenario_text} & "
                f"{row['Model']} & "
                f"{row['Split']} & "
                f"{row['Accuracy']:.3f} & "
                f"{row['Precision']:.3f} & "
                f"{row['Recall']:.3f} & "
                f"{row['F1-Score']:.3f} & "
                f"{row['AUC']:.3f} \\\\"
            )
        output.append("\\hline")
    
    output.append("\\end{tabular}")
    output.append("\\end{table}")
    
    return "\n".join(output)


def generate_comparison_table(df):
    """Generate scenario comparison table."""
    output = []
    output.append("\n" + "="*100)
    output.append("SCENARIO COMPARISON (F1-Score)")
    output.append("="*100)
    
    # Create comparison matrix
    scenarios = ['baseline_clean_train_clean_test', 
                 'baseline_clean_train_adv_test', 
                 'adv_train_adv_test']
    scenario_names = ['Clean→Clean', 'Clean→Adversarial', 'Adversarial→Adversarial']
    
    models = df['model'].unique()
    
    output.append(f"\n{'Model':<20} {'Clean→Clean':<25} {'Clean→Adversarial':<25} {'Adv→Adversarial':<25}")
    output.append("-"*100)
    
    for model in models:
        row_data = [model]
        for scenario in scenarios:
            mask = (df['model'] == model) & (df['scenario'] == scenario)
            if scenario == 'baseline_clean_train_adv_test':
                mask = mask & (df['split'] == 'adv_test')
            else:
                mask = mask & (df['split'].isin(['clean_test', 'adv_test']))
            
            if mask.any():
                f1 = df[mask]['f1'].values[0]
                row_data.append(format_percentage(f1, 2))
            else:
                row_data.append("N/A")
        
        output.append(f"{row_data[0]:<20} {row_data[1]:<25} {row_data[2]:<25} {row_data[3]:<25}")
    
    output.append("="*100)
    return "\n".join(output)


def generate_summary_statistics(df):
    """Generate summary statistics."""
    output = []
    output.append("\n" + "="*80)
    output.append("SUMMARY STATISTICS")
    output.append("="*80)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    output.append(f"\n{'Metric':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    output.append("-"*80)
    
    for metric in metrics:
        values = df[metric]
        output.append(
            f"{metric.capitalize():<15} "
            f"{format_percentage(values.mean()):<12} "
            f"{format_percentage(values.std()):<12} "
            f"{format_percentage(values.min()):<12} "
            f"{format_percentage(values.max()):<12}"
        )
    
    output.append("="*80)
    
    # Best performance
    output.append("\nBEST PERFORMANCE:")
    output.append("-"*80)
    
    for metric in ['accuracy', 'f1', 'auc']:
        best_idx = df[metric].idxmax()
        best_row = df.loc[best_idx]
        output.append(
            f"Best {metric.upper()}: {format_percentage(best_row[metric])} "
            f"({best_row['model']}, {best_row['scenario']}, {best_row['split']})"
        )
    
    output.append("="*80)
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description='Generate formatted metrics tables')
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('./results/metrics.csv'),
        help='Input metrics CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./results'),
        help='Output directory for formatted tables'
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        default=['console', 'markdown', 'latex'],
        choices=['console', 'markdown', 'latex', 'all'],
        help='Output formats to generate'
    )
    
    args = parser.parse_args()
    
    # Load metrics
    if not args.input.exists():
        print(f"Error: Metrics file not found: {args.input}")
        return
    
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} records from {args.input}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine formats
    formats = args.formats
    if 'all' in formats:
        formats = ['console', 'markdown', 'latex']
    
    # Generate tables
    for fmt in formats:
        print(f"\nGenerating {fmt.upper()} table...")
        
        if fmt == 'console':
            table = format_metrics_table(df, fmt)
            print(table)
            
            comparison = generate_comparison_table(df)
            print(comparison)
            
            summary = generate_summary_statistics(df)
            print(summary)
            
            # Save to file
            output_file = args.output_dir / 'metrics_table.txt'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(table)
                f.write("\n\n")
                f.write(comparison)
                f.write("\n\n")
                f.write(summary)
            print(f"Saved console table to {output_file}")
        
        elif fmt == 'markdown':
            table = format_metrics_table(df, fmt)
            output_file = args.output_dir / 'metrics_table.md'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(table)
            print(f"Saved markdown table to {output_file}")
        
        elif fmt == 'latex':
            table = format_latex_table(df)
            output_file = args.output_dir / 'metrics_table.tex'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(table)
            print(f"Saved LaTeX table to {output_file}")
    
    print("\n✓ All tables generated successfully!")


if __name__ == "__main__":
    main()
