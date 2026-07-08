"""
为每个数据集生成独立的指标报告
"""
import json
from pathlib import Path

# 手动输入六个数据集的结果数据
datasets_results = {
    "CICIDS-2017": {
        "info": {
            "samples": 53135,
            "features": 78,
            "labels": "10种 (BENIGN + 9种攻击)"
        },
        "results": [
            {"model": "SGDClassifier", "split": "clean_test", "scenario": "baseline_clean_train_clean_test", 
             "accuracy": 0.9594, "precision": 0.9647, "recall": 0.9907, "f1": 0.9775, "auc": 0.9580},
            {"model": "RidgeClassifier", "split": "clean_test", "scenario": "baseline_clean_train_clean_test",
             "accuracy": 0.9541, "precision": 0.9591, "recall": 0.9907, "f1": 0.9746, "auc": 0.9471},
            {"model": "SGDClassifier", "split": "adv_test", "scenario": "baseline_clean_train_adv_test",
             "accuracy": 0.0760, "precision": 0.4110, "recall": 0.0853, "f1": 0.1413, "auc": 0.0297},
            {"model": "RidgeClassifier", "split": "adv_test", "scenario": "baseline_clean_train_adv_test",
             "accuracy": 0.2330, "precision": 0.7332, "recall": 0.2188, "f1": 0.3370, "auc": 0.1911},
            {"model": "SGDClassifier", "split": "clean_test", "scenario": "adv_train_clean_test",
             "accuracy": 0.9548, "precision": 0.9607, "recall": 0.9898, "f1": 0.9750, "auc": 0.9499},
            {"model": "RidgeClassifier", "split": "clean_test", "scenario": "adv_train_clean_test",
             "accuracy": 0.9466, "precision": 0.9532, "recall": 0.9887, "f1": 0.9706, "auc": 0.9235},
            {"model": "SGDClassifier", "split": "adv_test", "scenario": "adv_train_adv_test",
             "accuracy": 0.9976, "precision": 0.9973, "recall": 1.0000, "f1": 0.9986, "auc": 0.9995},
            {"model": "RidgeClassifier", "split": "adv_test", "scenario": "adv_train_adv_test",
             "accuracy": 0.9788, "precision": 0.9774, "recall": 0.9994, "f1": 0.9883, "auc": 0.9923},
        ]
    },
    "CICIDS-2018": {
        "info": {
            "samples": 59565,
            "features": 78,
            "labels": "3种 (Benign, DoS GoldenEye, DoS Slowloris)"
        },
        "results": [
            {"model": "SGDClassifier", "split": "clean_test", "scenario": "baseline_clean_train_clean_test",
             "accuracy": 0.9950, "precision": 0.9847, "recall": 0.9856, "f1": 0.9825, "auc": 0.9994},
            {"model": "RidgeClassifier", "split": "clean_test", "scenario": "baseline_clean_train_clean_test",
             "accuracy": 0.9549, "precision": 0.8815, "recall": 0.8815, "f1": 0.8815, "auc": 0.9965},
            {"model": "SGDClassifier", "split": "adv_test", "scenario": "baseline_clean_train_adv_test",
             "accuracy": 0.9945, "precision": 0.9805, "recall": 0.9805, "f1": 0.9805, "auc": 0.9393},
            {"model": "RidgeClassifier", "split": "adv_test", "scenario": "baseline_clean_train_adv_test",
             "accuracy": 0.9535, "precision": 0.3457, "recall": 0.3457, "f1": 0.3457, "auc": 0.7261},
            {"model": "SGDClassifier", "split": "clean_test", "scenario": "adv_train_clean_test",
             "accuracy": 0.9946, "precision": 0.9790, "recall": 0.9790, "f1": 0.9790, "auc": 0.9992},
            {"model": "RidgeClassifier", "split": "clean_test", "scenario": "adv_train_clean_test",
             "accuracy": 0.9148, "precision": 0.8047, "recall": 0.8047, "f1": 0.8047, "auc": 0.9909},
            {"model": "SGDClassifier", "split": "adv_test", "scenario": "adv_train_adv_test",
             "accuracy": 0.9948, "precision": 0.9630, "recall": 0.9630, "f1": 0.9630, "auc": 0.9996},
            {"model": "RidgeClassifier", "split": "adv_test", "scenario": "adv_train_adv_test",
             "accuracy": 0.9975, "precision": 0.9950, "recall": 0.9950, "f1": 0.9950, "auc": 0.9999},
        ]
    },
    "CICIDS-2019": {
        "info": {
            "samples": 56903,
            "features": 82,
            "labels": "2种 (BENIGN, Portmap)"
        },
        "results": [
            {"model": "SGDClassifier", "split": "clean_test", "scenario": "baseline_clean_train_clean_test",
             "accuracy": 0.9768, "precision": 0.9984, "recall": 0.9984, "f1": 0.9984, "auc": 0.9963},
            {"model": "RidgeClassifier", "split": "clean_test", "scenario": "baseline_clean_train_clean_test",
             "accuracy": 0.9751, "precision": 0.9993, "recall": 0.9993, "f1": 0.9993, "auc": 0.9996},
            {"model": "SGDClassifier", "split": "adv_test", "scenario": "baseline_clean_train_adv_test",
             "accuracy": 0.9749, "precision": 0.9871, "recall": 0.9871, "f1": 0.9871, "auc": 0.5888},
            {"model": "RidgeClassifier", "split": "adv_test", "scenario": "baseline_clean_train_adv_test",
             "accuracy": 0.9744, "precision": 0.9944, "recall": 0.9944, "f1": 0.9944, "auc": 0.9609},
            {"model": "SGDClassifier", "split": "clean_test", "scenario": "adv_train_clean_test",
             "accuracy": 0.9772, "precision": 0.9988, "recall": 0.9988, "f1": 0.9988, "auc": 0.9944},
            {"model": "RidgeClassifier", "split": "clean_test", "scenario": "adv_train_clean_test",
             "accuracy": 0.9755, "precision": 0.9991, "recall": 0.9991, "f1": 0.9991, "auc": 0.9995},
            {"model": "SGDClassifier", "split": "adv_test", "scenario": "adv_train_adv_test",
             "accuracy": 0.9765, "precision": 0.9982, "recall": 0.9982, "f1": 0.9982, "auc": 0.9722},
            {"model": "RidgeClassifier", "split": "adv_test", "scenario": "adv_train_adv_test",
             "accuracy": 0.9756, "precision": 0.9996, "recall": 0.9996, "f1": 0.9996, "auc": 0.9999},
        ]
    },
    "BoTIoT-2018": {
        "info": {
            "samples": 5477,
            "features": 16,
            "labels": "6种 (Normal, TCP, UDP, Service_Scan, OS_Fingerprint, HTTP)"
        },
        "results": [
            {"model": "SGDClassifier", "split": "clean_test", "scenario": "baseline_clean_train_clean_test",
             "accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1": 1.0000, "auc": 1.0000},
            {"model": "RidgeClassifier", "split": "clean_test", "scenario": "baseline_clean_train_clean_test",
             "accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1": 1.0000, "auc": 1.0000},
            {"model": "SGDClassifier", "split": "adv_test", "scenario": "baseline_clean_train_adv_test",
             "accuracy": 0.9662, "precision": 0.9662, "recall": 0.9662, "f1": 0.9662, "auc": 1.0000},
            {"model": "RidgeClassifier", "split": "adv_test", "scenario": "baseline_clean_train_adv_test",
             "accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1": 1.0000, "auc": 1.0000},
            {"model": "SGDClassifier", "split": "clean_test", "scenario": "adv_train_clean_test",
             "accuracy": 0.9995, "precision": 0.9995, "recall": 0.9995, "f1": 0.9995, "auc": 0.9990},
            {"model": "RidgeClassifier", "split": "clean_test", "scenario": "adv_train_clean_test",
             "accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1": 1.0000, "auc": 1.0000},
            {"model": "SGDClassifier", "split": "adv_test", "scenario": "adv_train_adv_test",
             "accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1": 1.0000, "auc": 1.0000},
            {"model": "RidgeClassifier", "split": "adv_test", "scenario": "adv_train_adv_test",
             "accuracy": 1.0000, "precision": 1.0000, "recall": 1.0000, "f1": 1.0000, "auc": 1.0000},
        ]
    },
    "N-BaiIoT-2021": {
        "info": {
            "samples": 165507,
            "features": 115,
            "labels": "2种 (Normal, Attack)"
        },
        "results": [
            {"model": "SGDClassifier", "split": "clean_test", "scenario": "baseline_clean_train_clean_test",
             "accuracy": 0.9994, "precision": 0.9994, "recall": 0.9994, "f1": 0.9994, "auc": 0.9996},
            {"model": "RidgeClassifier", "split": "clean_test", "scenario": "baseline_clean_train_clean_test",
             "accuracy": 0.9969, "precision": 0.9938, "recall": 0.9938, "f1": 0.9938, "auc": 0.9994},
            {"model": "SGDClassifier", "split": "adv_test", "scenario": "baseline_clean_train_adv_test",
             "accuracy": 0.8048, "precision": 0.6687, "recall": 0.6687, "f1": 0.6687, "auc": 0.6904},
            {"model": "RidgeClassifier", "split": "adv_test", "scenario": "baseline_clean_train_adv_test",
             "accuracy": 0.6685, "precision": 0.4932, "recall": 0.4932, "f1": 0.4932, "auc": 0.3331},
            {"model": "SGDClassifier", "split": "clean_test", "scenario": "adv_train_clean_test",
             "accuracy": 0.9992, "precision": 0.9988, "recall": 0.9988, "f1": 0.9988, "auc": 0.9995},
            {"model": "RidgeClassifier", "split": "clean_test", "scenario": "adv_train_clean_test",
             "accuracy": 0.9951, "precision": 0.9902, "recall": 0.9902, "f1": 0.9902, "auc": 0.9991},
            {"model": "SGDClassifier", "split": "adv_test", "scenario": "adv_train_adv_test",
             "accuracy": 0.9995, "precision": 0.9991, "recall": 0.9991, "f1": 0.9991, "auc": 0.9995},
            {"model": "RidgeClassifier", "split": "adv_test", "scenario": "adv_train_adv_test",
             "accuracy": 0.9991, "precision": 0.9983, "recall": 0.9983, "f1": 0.9983, "auc": 0.9995},
        ]
    },
}

def generate_txt_table(dataset_name, data):
    """生成文本格式的表格"""
    info = data["info"]
    results = data["results"]
    
    lines = []
    lines.append("=" * 120)
    lines.append(f"{dataset_name.upper()} - 性能指标报告")
    lines.append("=" * 120)
    lines.append("")
    lines.append("数据集信息:")
    lines.append(f"  样本数: {info['samples']:,}")
    lines.append(f"  特征数: {info['features']}")
    lines.append(f"  类别: {info['labels']}")
    lines.append("")
    
    # 按场景分组
    scenarios = {
        "baseline_clean_train_clean_test": "场景1: 基线 - 干净训练 → 干净测试",
        "baseline_clean_train_adv_test": "场景2: 脆弱性 - 干净训练 → 对抗测试",
        "adv_train_clean_test": "场景3: 副作用 - 对抗训练 → 干净测试",
        "adv_train_adv_test": "场景4: 防御效果 - 对抗训练 → 对抗测试"
    }
    
    for scenario_key, scenario_name in scenarios.items():
        scenario_results = [r for r in results if r['scenario'] == scenario_key]
        if not scenario_results:
            continue
            
        lines.append("-" * 120)
        lines.append(scenario_name)
        lines.append("-" * 120)
        lines.append(f"{'模型':<20} {'测试集':<15} {'准确率':<12} {'精确率':<12} {'召回率':<12} {'F1分数':<12} {'AUC':<12}")
        lines.append("-" * 120)
        
        for r in scenario_results:
            lines.append(
                f"{r['model']:<20} {r['split']:<15} "
                f"{r['accuracy']*100:>10.2f}% {r['precision']*100:>10.2f}% "
                f"{r['recall']*100:>10.2f}% {r['f1']*100:>10.2f}% "
                f"{r['auc']*100:>10.2f}%"
            )
        lines.append("")
    
    # 关键指标对比
    lines.append("=" * 120)
    lines.append("关键指标对比")
    lines.append("=" * 120)
    lines.append("")
    
    for model in ["SGDClassifier", "RidgeClassifier"]:
        model_results = [r for r in results if r['model'] == model]
        baseline_clean = next((r for r in model_results if r['scenario'] == 'baseline_clean_train_clean_test'), None)
        baseline_adv = next((r for r in model_results if r['scenario'] == 'baseline_clean_train_adv_test'), None)
        adv_clean = next((r for r in model_results if r['scenario'] == 'adv_train_clean_test'), None)
        adv_adv = next((r for r in model_results if r['scenario'] == 'adv_train_adv_test'), None)
        
        lines.append(f"{model}:")
        if baseline_clean:
            lines.append(f"  基线性能 (干净→干净):        准确率 {baseline_clean['accuracy']*100:.2f}%, F1 {baseline_clean['f1']*100:.2f}%")
        if baseline_adv:
            lines.append(f"  对抗脆弱性 (干净→对抗):      准确率 {baseline_adv['accuracy']*100:.2f}%, F1 {baseline_adv['f1']*100:.2f}%")
        if adv_adv and baseline_adv:
            improvement = (adv_adv['accuracy'] - baseline_adv['accuracy']) * 100
            lines.append(f"  对抗训练效果 (对抗→对抗):    准确率 {adv_adv['accuracy']*100:.2f}%, F1 {adv_adv['f1']*100:.2f}%")
            lines.append(f"  对抗训练提升:                +{improvement:.2f}%")
        if adv_clean and baseline_clean:
            cost = (baseline_clean['accuracy'] - adv_clean['accuracy']) * 100
            lines.append(f"  对抗训练代价:                {cost:+.2f}%")
        lines.append("")
    
    # 最佳性能
    lines.append("-" * 120)
    lines.append("最佳性能:")
    best_acc = max(results, key=lambda x: x['accuracy'])
    best_f1 = max(results, key=lambda x: x['f1'])
    best_auc = max(results, key=lambda x: x['auc'])
    lines.append(f"  最高准确率: {best_acc['accuracy']*100:.2f}% ({best_acc['model']}, {best_acc['scenario']})")
    lines.append(f"  最高F1分数: {best_f1['f1']*100:.2f}% ({best_f1['model']}, {best_f1['scenario']})")
    lines.append(f"  最高AUC:    {best_auc['auc']*100:.2f}% ({best_auc['model']}, {best_auc['scenario']})")
    lines.append("=" * 120)
    lines.append("")
    
    return "\n".join(lines)

def generate_markdown_table(dataset_name, data):
    """生成Markdown格式的表格"""
    info = data["info"]
    results = data["results"]
    
    lines = []
    lines.append(f"# {dataset_name} - 性能指标报告\n")
    lines.append("## 数据集信息\n")
    lines.append(f"- **样本数**: {info['samples']:,}")
    lines.append(f"- **特征数**: {info['features']}")
    lines.append(f"- **类别**: {info['labels']}\n")
    
    # 按场景分组
    scenarios = {
        "baseline_clean_train_clean_test": "场景1: 基线 - 干净训练 → 干净测试",
        "baseline_clean_train_adv_test": "场景2: 脆弱性 - 干净训练 → 对抗测试",
        "adv_train_clean_test": "场景3: 副作用 - 对抗训练 → 干净测试",
        "adv_train_adv_test": "场景4: 防御效果 - 对抗训练 → 对抗测试"
    }
    
    lines.append("## 详细结果\n")
    
    for scenario_key, scenario_name in scenarios.items():
        scenario_results = [r for r in results if r['scenario'] == scenario_key]
        if not scenario_results:
            continue
            
        lines.append(f"### {scenario_name}\n")
        lines.append("| 模型 | 测试集 | 准确率 | 精确率 | 召回率 | F1分数 | AUC |")
        lines.append("|------|--------|--------|--------|--------|--------|-----|")
        
        for r in scenario_results:
            lines.append(
                f"| {r['model']} | {r['split']} | "
                f"{r['accuracy']*100:.2f}% | {r['precision']*100:.2f}% | "
                f"{r['recall']*100:.2f}% | {r['f1']*100:.2f}% | "
                f"{r['auc']*100:.2f}% |"
            )
        lines.append("")
    
    # 关键指标对比
    lines.append("## 关键指标对比\n")
    
    lines.append("| 模型 | 基线性能 | 对抗脆弱性 | 对抗训练后 | 提升幅度 | 训练代价 |")
    lines.append("|------|----------|-----------|-----------|---------|---------|")
    
    for model in ["SGDClassifier", "RidgeClassifier"]:
        model_results = [r for r in results if r['model'] == model]
        baseline_clean = next((r for r in model_results if r['scenario'] == 'baseline_clean_train_clean_test'), None)
        baseline_adv = next((r for r in model_results if r['scenario'] == 'baseline_clean_train_adv_test'), None)
        adv_clean = next((r for r in model_results if r['scenario'] == 'adv_train_clean_test'), None)
        adv_adv = next((r for r in model_results if r['scenario'] == 'adv_train_adv_test'), None)
        
        base_perf = f"{baseline_clean['accuracy']*100:.2f}%" if baseline_clean else "-"
        vuln = f"{baseline_adv['accuracy']*100:.2f}%" if baseline_adv else "-"
        after_def = f"{adv_adv['accuracy']*100:.2f}%" if adv_adv else "-"
        
        if adv_adv and baseline_adv:
            improvement = f"+{(adv_adv['accuracy'] - baseline_adv['accuracy'])*100:.2f}%"
        else:
            improvement = "-"
            
        if adv_clean and baseline_clean:
            cost = f"{(baseline_clean['accuracy'] - adv_clean['accuracy'])*100:+.2f}%"
        else:
            cost = "-"
        
        lines.append(f"| {model} | {base_perf} | {vuln} | {after_def} | {improvement} | {cost} |")
    
    lines.append("")
    
    # 最佳性能
    lines.append("## 最佳性能\n")
    best_acc = max(results, key=lambda x: x['accuracy'])
    best_f1 = max(results, key=lambda x: x['f1'])
    best_auc = max(results, key=lambda x: x['auc'])
    
    lines.append(f"- **最高准确率**: {best_acc['accuracy']*100:.2f}% ({best_acc['model']})")
    lines.append(f"- **最高F1分数**: {best_f1['f1']*100:.2f}% ({best_f1['model']})")
    lines.append(f"- **最高AUC**: {best_auc['auc']*100:.2f}% ({best_auc['model']})")
    lines.append("")
    
    return "\n".join(lines)

def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # 为每个数据集生成报告
    for dataset_name, data in datasets_results.items():
        # 生成TXT格式
        txt_content = generate_txt_table(dataset_name, data)
        txt_file = results_dir / f"{dataset_name.replace('-', '_').lower()}_metrics.txt"
        txt_file.write_text(txt_content, encoding='utf-8')
        print(f"✓ 生成 {txt_file}")
        
        # 生成Markdown格式
        md_content = generate_markdown_table(dataset_name, data)
        md_file = results_dir / f"{dataset_name.replace('-', '_').upper()}_METRICS.md"
        md_file.write_text(md_content, encoding='utf-8')
        print(f"✓ 生成 {md_file}")
    
    print(f"\n✓ 所有数据集报告生成完成！共 {len(datasets_results)} 个数据集")
    print(f"  - TXT格式: results/*_metrics.txt")
    print(f"  - Markdown格式: results/*_METRICS.md")

if __name__ == '__main__':
    main()
