#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
概念漂移验证脚本
验证生成的数据是否真的包含概念漂移
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

def validate_drift_detection(csv_path, drift_type, window_size=100):
    """
    验证概念漂移是否被成功注入
    通过分析数据分布和分类器性能变化来检测漂移
    """
    print(f"📊 验证 {drift_type} 漂移数据: {os.path.basename(csv_path)}")
    
    # 读取数据
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    print(f"   数据集形状: {X.shape}")
    print(f"   类别数: {len(np.unique(y))}")
    
    # 计算滑动窗口的统计信息
    n_samples = len(X)
    n_windows = n_samples // window_size
    
    # 存储每个窗口的统计信息
    window_means = []
    window_stds = []
    window_accuracies = []
    window_labels = []
    
    # 训练初始分类器
    scaler = StandardScaler()
    clf = LogisticRegression(random_state=42, max_iter=1000)
    
    # 使用前两个窗口训练分类器
    if n_windows >= 2:
        train_X = X[:2*window_size]
        train_y = y[:2*window_size]
        train_X_scaled = scaler.fit_transform(train_X)
        clf.fit(train_X_scaled, train_y)
    
    # 分析每个窗口
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, n_samples)
        
        window_X = X[start_idx:end_idx]
        window_y = y[start_idx:end_idx]
        
        # 计算特征统计
        mean_features = np.mean(window_X, axis=0)
        std_features = np.std(window_X, axis=0)
        
        window_means.append(np.mean(mean_features))
        window_stds.append(np.mean(std_features))
        
        # 计算分类准确率
        if len(window_X) > 0:
            window_X_scaled = scaler.transform(window_X)
            y_pred = clf.predict(window_X_scaled)
            accuracy = accuracy_score(window_y, y_pred)
            window_accuracies.append(accuracy)
        else:
            window_accuracies.append(0.0)
        
        # 计算标签分布
        label_dist = np.bincount(window_y.astype(int))
        window_labels.append(label_dist)
    
    # 分析结果
    print(f"   分析了 {n_windows} 个窗口")
    print(f"   平均特征均值变化: {np.std(window_means):.4f}")
    print(f"   平均特征标准差变化: {np.std(window_stds):.4f}")
    print(f"   准确率变化: {np.std(window_accuracies):.4f}")
    print(f"   最低准确率: {np.min(window_accuracies):.4f}")
    print(f"   最高准确率: {np.max(window_accuracies):.4f}")
    
    # 检测漂移点
    drift_detected = False
    if len(window_accuracies) > 2:
        # 检查准确率是否有显著下降
        accuracy_changes = np.diff(window_accuracies)
        significant_drops = np.where(accuracy_changes < -0.1)[0]
        
        if len(significant_drops) > 0:
            drift_detected = True
            print(f"   🔍 检测到漂移点: 窗口 {significant_drops}")
    
    # 特征变化检测
    if len(window_means) > 2:
        mean_changes = np.diff(window_means)
        significant_mean_changes = np.where(np.abs(mean_changes) > np.std(mean_changes) * 2)[0]
        
        if len(significant_mean_changes) > 0:
            drift_detected = True
            print(f"   🔍 检测到特征分布变化: 窗口 {significant_mean_changes}")
    
    if drift_detected:
        print(f"   ✅ 成功检测到概念漂移！")
    else:
        print(f"   ⚠️  未检测到明显的概念漂移")
    
    return {
        'drift_type': drift_type,
        'window_means': window_means,
        'window_stds': window_stds,
        'window_accuracies': window_accuracies,
        'drift_detected': drift_detected,
        'n_windows': n_windows
    }

def compare_with_original(original_dataset='Waveform', n_samples=3000):
    """
    与原始数据集进行比较
    """
    print(f"\n🔄 与原始 {original_dataset} 数据集比较...")
    
    # 生成原始数据
    from skmultiflow.data import WaveformGenerator
    original_stream = WaveformGenerator(random_state=42)
    X_original, y_original = original_stream.next_sample(n_samples)
    
    # 计算原始数据的统计信息
    original_mean = np.mean(X_original.flatten())
    original_std = np.std(X_original.flatten())
    
    print(f"   原始数据 - 均值: {original_mean:.4f}, 标准差: {original_std:.4f}")
    
    # 比较漂移数据
    drift_files = [
        'Waveform_sudden_severity0.3_pos0.5_width0.1.csv',
        'Waveform_gradual_severity0.5_pos0.5_width0.2.csv',
        'Waveform_incremental_severity0.4_pos0.3_width0.1.csv',
        'Waveform_recurring_severity0.3_pos0.5_width0.1.csv'
    ]
    
    for filename in drift_files:
        filepath = os.path.join('drift_datasets', filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            X_drift = df.iloc[:, :-1].values
            
            drift_mean = np.mean(X_drift.flatten())
            drift_std = np.std(X_drift.flatten())
            
            print(f"   {filename.split('_')[1]:12} - 均值: {drift_mean:.4f}, 标准差: {drift_std:.4f}")
            print(f"   {'':12}   均值差异: {abs(drift_mean - original_mean):.4f}, 标准差差异: {abs(drift_std - original_std):.4f}")

def main():
    """主验证函数"""
    print("🔍 概念漂移验证工具")
    print("=" * 50)
    
    # 验证所有生成的漂移数据
    drift_files = [
        ('Waveform_sudden_severity0.3_pos0.5_width0.1.csv', 'sudden'),
        ('Waveform_gradual_severity0.5_pos0.5_width0.2.csv', 'gradual'),
        ('Waveform_incremental_severity0.4_pos0.3_width0.1.csv', 'incremental'),
        ('Waveform_recurring_severity0.3_pos0.5_width0.1.csv', 'recurring')
    ]
    
    results = []
    
    for filename, drift_type in drift_files:
        filepath = os.path.join('drift_datasets', filename)
        if os.path.exists(filepath):
            result = validate_drift_detection(filepath, drift_type)
            results.append(result)
            print()
    
    # 与原始数据比较
    compare_with_original()
    
    # 总结报告
    print(f"\n🎉 验证总结:")
    successful_detections = sum(1 for r in results if r['drift_detected'])
    total_tests = len(results)
    
    print(f"   成功检测到漂移: {successful_detections}/{total_tests}")
    
    for result in results:
        status = "✅ 检测到" if result['drift_detected'] else "❌ 未检测到"
        print(f"   {result['drift_type']:12} | {status}")
    
    if successful_detections == total_tests:
        print("\n🏆 所有漂移都被成功检测到！漂移注入工具工作正常。")
    elif successful_detections > 0:
        print(f"\n⚠️  部分漂移被检测到。可能需要调整漂移参数。")
    else:
        print(f"\n❌ 未检测到任何漂移。请检查漂移注入逻辑。")

if __name__ == "__main__":
    main() 