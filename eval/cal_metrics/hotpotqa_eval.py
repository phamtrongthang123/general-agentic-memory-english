#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAM 框架 + HotpotQA 结果评估代码

基于 hotpotqa.py 的 F1 计算方法和 locomo_eval.py 的逻辑，
计算所有样本的评估指标。
"""

import json
import re
import csv
import os
from collections import defaultdict, Counter
from typing import List, Dict, Any

# ========== 文本归一化与 F1 计算：借鉴自 hotpotqa.py ==========

def normalize_text(s: str) -> str:
    """文本归一化：借鉴自 hotpotqa.py"""
    if s is None:
        return ""
    s = str(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def f1_score(prediction: str, gold: str) -> float:
    """F1分数计算：完全借鉴自 hotpotqa.py"""
    p = normalize_text(prediction).split()
    g = normalize_text(gold).split()
    if not p and not g: 
        return 1.0
    if not p or not g:  
        return 0.0
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:   
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)

# ========== 评估指标计算：只计算F1 ==========

def compute_metrics_by_type(items: List[Dict[str, Any]], pred_key: str = "summary_answer") -> tuple:
    """
    按问题类型计算指标：只计算F1分数
    HotpotQA 的问题类型包括：comparison, bridge, intersection
    """
    agg = defaultdict(list)
    rows = []
    
    for ex in items:
        q_type = ex.get("type", "unknown")
        level = ex.get("level", "unknown")
        gold = ex.get("gold_answer", "")
        pred = ex.get(pred_key, "")
        
        # 只计算F1指标
        f1 = f1_score(pred, gold)
        
        # 按类型聚合
        agg[q_type].append(f1)
        
        # 记录详细信息
        rows.append({
            "sample_id": ex.get("sample_id", ""),
            "question": ex.get("question", ""),
            "type": q_type,
            "level": level,
            "gold_answer": str(gold),
            "prediction": str(pred),
            "F1": f1,
            "supporting_facts": ex.get("supporting_facts", [])
        })
    
    # 计算汇总统计
    summary = []
    for q_type in sorted(agg.keys(), key=lambda x: str(x)):
        scores = agg[q_type]
        if scores:
            f1_avg = sum(scores) / len(scores)
            summary.append({
                "type": q_type, 
                "count": len(scores), 
                "F1_avg": f1_avg
            })
    
    return summary, rows

def compute_metrics_by_level(items: List[Dict[str, Any]], pred_key: str = "summary_answer") -> tuple:
    """按难度级别计算指标：只计算F1分数"""
    agg = defaultdict(list)
    rows = []
    
    for ex in items:
        q_type = ex.get("type", "unknown")
        level = ex.get("level", "unknown")
        gold = ex.get("gold_answer", "")
        pred = ex.get(pred_key, "")
        
        f1 = f1_score(pred, gold)
        
        agg[level].append(f1)
        
        rows.append({
            "sample_id": ex.get("sample_id", ""),
            "question": ex.get("question", ""),
            "type": q_type,
            "level": level,
            "gold_answer": str(gold),
            "prediction": str(pred),
            "F1": f1
        })
    
    summary = []
    for level in sorted(agg.keys(), key=lambda x: str(x)):
        scores = agg[level]
        if scores:
            f1_avg = sum(scores) / len(scores)
            summary.append({
                "level": level, 
                "count": len(scores), 
                "F1_avg": f1_avg
            })
    
    return summary, rows

def write_csv(path: str, rows: List[Dict[str, Any]], fields: List[str]):
    """写入CSV文件：借鉴自 locomo_eval.py"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def load_all_results(input_dir: str) -> List[Dict[str, Any]]:
    """
    加载所有结果：借鉴自 locomo_eval.py 的逻辑
    但适配 HotpotQA 的文件夹结构
    """
    all_data = []
    
    # 获取所有子文件夹（HotpotQA 的样本ID格式）
    subdirs = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path) and (item.startswith('sample_') or item.startswith('5')):
            subdirs.append(item)
    
    subdirs.sort()
    print(f"找到 {len(subdirs)} 个样本文件夹: {subdirs[:5]}..." if len(subdirs) > 5 else f"找到 {len(subdirs)} 个样本文件夹: {subdirs}")
    
    # 合并所有数据
    for subdir in subdirs:
        input_json = os.path.join(input_dir, subdir, "qa_results.json")
        
        if not os.path.exists(input_json):
            print(f"跳过 {subdir}: 找不到 {input_json}")
            continue
            
        print(f"读取文件夹: {subdir}")
        
        try:
            with open(input_json, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                
                if isinstance(data, list):
                    questions = data
                    # 为每个问题添加来源文件夹信息
                    for item in questions:
                        item['source_folder'] = subdir
                    all_data.extend(questions)
                    print(f"  添加了 {len(questions)} 条数据")
                else:
                    print(f"  警告: {subdir} 中数据格式不是数组")
                
        except Exception as e:
            print(f"读取 {subdir} 时出错: {e}")
            continue
    
    print(f"\n总共合并了 {len(all_data)} 条数据")
    return all_data

def main():
    # ========== 配置参数 ==========
    INPUT_DIR = "/share/project/bingyu/code/general-agentic-memory/results/hotpotqa_output"
    OUTPUT_DIR = "/share/project/bingyu/code/general-agentic-memory/results/cal_metrics/hotpotqa"
    PRED_KEYS = ["memory_answer", "summary_answer"]  # 评估两种答案类型
    
    print("=" * 60)
    print("GAM 框架 + HotpotQA 结果评估")
    print("=" * 60)
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"评估答案类型: {PRED_KEYS}")
    print("=" * 60)
    
    # 检查输入目录
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入目录不存在: {INPUT_DIR}")
        return
    
    # 加载所有结果
    all_data = load_all_results(INPUT_DIR)
    
    if not all_data:
        print("错误: 没有找到任何数据")
        return
    
    # 对所有合并的数据进行整体分析
    for pred_key in PRED_KEYS:
        print(f"\n# HotpotQA Metrics for pred_key='{pred_key}' (所有数据)")
        
        # 按问题类型计算指标
        type_summary, type_details = compute_metrics_by_type(all_data, pred_key=pred_key)
        
        # 按难度级别计算指标
        level_summary, level_details = compute_metrics_by_level(all_data, pred_key=pred_key)
        
        # 保存按类型的结果
        type_sum_csv = os.path.join(OUTPUT_DIR, f"hotpotqa_metrics_{pred_key}_by_type_summary.csv")
        type_det_csv = os.path.join(OUTPUT_DIR, f"hotpotqa_metrics_{pred_key}_by_type_details.csv")
        write_csv(type_sum_csv, type_summary, ["type", "count", "F1_avg"])
        write_csv(type_det_csv, type_details, ["sample_id", "question", "type", "level", "gold_answer", "prediction", "F1", "supporting_facts", "source_folder"])
        
        # 保存按级别的结果
        level_sum_csv = os.path.join(OUTPUT_DIR, f"hotpotqa_metrics_{pred_key}_by_level_summary.csv")
        level_det_csv = os.path.join(OUTPUT_DIR, f"hotpotqa_metrics_{pred_key}_by_level_details.csv")
        write_csv(level_sum_csv, level_summary, ["level", "count", "F1_avg"])
        write_csv(level_det_csv, level_details, ["sample_id", "question", "type", "level", "gold_answer", "prediction", "F1", "source_folder"])
        
        # 打印统计结果
        print(f"\n按问题类型统计 ({pred_key}):")
        for r in type_summary:
            print(f"  Type {r['type']}: n={r['count']}, F1_avg={r['F1_avg']:.4f}")
        
        print(f"\n按难度级别统计 ({pred_key}):")
        for r in level_summary:
            print(f"  Level {r['level']}: n={r['count']}, F1_avg={r['F1_avg']:.4f}")
        
        # 计算整体统计
        all_f1_scores = [row['F1'] for row in type_details]
        overall_f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0.0
        
        print(f"\n整体统计 ({pred_key}):")
        print(f"  总样本数: {len(all_data)}")
        print(f"  整体 F1: {overall_f1:.4f}")
        
        print(f"\n结果文件:")
        print(f"  按类型汇总: {type_sum_csv}")
        print(f"  按类型详情: {type_det_csv}")
        print(f"  按级别汇总: {level_sum_csv}")
        print(f"  按级别详情: {level_det_csv}")
    
    print(f"\n{'='*60}")
    print("评估完成!")
    print(f"处理了 {len(all_data)} 条数据")
    print(f"结果保存在: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
