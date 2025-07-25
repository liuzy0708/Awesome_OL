#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concept Drift Visualization Tool
Integrated visualization toolkit for analyzing concept drift datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
import warnings
warnings.filterwarnings('ignore')

# Set English font and style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class ConceptDriftVisualizer:
    """Comprehensive concept drift visualization toolkit"""
    
    def __init__(self, data_dir='./drift_datasets'):
        """Initialize the visualizer"""
        self.data_dir = data_dir
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        self.drift_types = ['sudden', 'gradual', 'incremental', 'recurring']
        self.drift_names = {
            'sudden': 'Sudden Drift',
            'gradual': 'Gradual Drift',
            'incremental': 'Incremental Drift',
            'recurring': 'Recurring Drift'
        }
        
    def load_drift_data(self, filename):
        """Load drift dataset"""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            return X, y
        else:
            print(f"File not found: {filepath}")
            return None, None
    
    def get_filename_pattern(self, dataset_name, drift_type):
        """Get actual filename for specific dataset and drift type"""
        if not os.path.exists(self.data_dir):
            return None
        
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        # 查找匹配的文件
        for file in files:
            if file.startswith(f"{dataset_name}_{drift_type}_") and file.endswith('.csv'):
                return file
        
        return None
    
    def list_available_datasets(self):
        """List all available datasets"""
        if not os.path.exists(self.data_dir):
            print(f"Data directory not found: {self.data_dir}")
            return {}
        
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        datasets = {}
        
        for file in files:
            # 解析文件名格式: dataset_drifttype_severityX_posY_widthZ.csv
            parts = file.replace('.csv', '').split('_')
            if len(parts) >= 3:
                dataset_name = parts[0]
                drift_type = parts[1]
                
                if drift_type in self.drift_types:
                    if dataset_name not in datasets:
                        datasets[dataset_name] = []
                    if drift_type not in datasets[dataset_name]:
                        datasets[dataset_name].append(drift_type)
        
        return datasets
    
    def plot_overview_comparison(self, dataset_name="Waveform", save_path=None):
        """Plot overview comparison of all drift types"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Feature mean changes
        ax1 = axes[0, 0]
        window_size = 100
        
        for i, drift_type in enumerate(self.drift_types):
            filename = self.get_filename_pattern(dataset_name, drift_type)
            if not filename:
                continue
                
            X, y = self.load_drift_data(filename)
            if X is None:
                continue
            
            n_windows = len(X) // window_size
            positions = []
            means = []
            
            for w in range(n_windows):
                start_idx = w * window_size
                end_idx = min((w + 1) * window_size, len(X))
                window_X = X[start_idx:end_idx]
                means.append(np.mean(window_X))
                positions.append(start_idx)
            
            ax1.plot(positions, means, 'o-', 
                    color=self.colors[i], linewidth=2, markersize=4,
                    label=self.drift_names[drift_type])
        
        ax1.set_title('Feature Mean Changes Over Time', fontsize=14)
        ax1.set_xlabel('Sample Position', fontsize=12)
        ax1.set_ylabel('Feature Mean', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Classification performance
        ax2 = axes[0, 1]
        
        for i, drift_type in enumerate(self.drift_types):
            filename = self.get_filename_pattern(dataset_name, drift_type)
            if not filename:
                continue
                
            X, y = self.load_drift_data(filename)
            if X is None:
                continue
            
            n_windows = len(X) // window_size
            scaler = StandardScaler()
            clf = LogisticRegression(random_state=42, max_iter=1000)
            
            if n_windows >= 2:
                train_X = X[:2*window_size]
                train_y = y[:2*window_size]
                train_X_scaled = scaler.fit_transform(train_X)
                clf.fit(train_X_scaled, train_y)
            
            accuracies = []
            positions = []
            
            for w in range(n_windows):
                start_idx = w * window_size
                end_idx = min((w + 1) * window_size, len(X))
                window_X = X[start_idx:end_idx]
                window_y = y[start_idx:end_idx]
                
                if len(window_X) > 0:
                    window_X_scaled = scaler.transform(window_X)
                    y_pred = clf.predict(window_X_scaled)
                    accuracy = accuracy_score(window_y, y_pred)
                    accuracies.append(accuracy)
                    positions.append(start_idx)
            
            ax2.plot(positions, accuracies, 'o-', 
                    color=self.colors[i], linewidth=2, markersize=4,
                    label=self.drift_names[drift_type])
        
        ax2.set_title('Classification Performance Over Time', fontsize=14)
        ax2.set_xlabel('Sample Position', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.5, 1.0)
        
        # Plot 3: 2D PCA visualization for sudden drift
        ax3 = axes[1, 0]
        filename = self.get_filename_pattern(dataset_name, 'sudden')
        if filename:
            X, y = self.load_drift_data(filename)
            if X is not None:
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(X)
                
                mid_point = len(X_2d) // 2
                ax3.scatter(X_2d[:mid_point, 0], X_2d[:mid_point, 1], 
                           c='blue', alpha=0.6, s=20, label='Before Drift')
                ax3.scatter(X_2d[mid_point:, 0], X_2d[mid_point:, 1], 
                           c='red', alpha=0.6, s=20, marker='^', label='After Drift')
                
                ax3.set_title('2D PCA - Sudden Drift Example', fontsize=14)
                ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
                ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        # Plot 4: Drift intensity comparison
        ax4 = axes[1, 1]
        drift_intensities = []
        drift_names = []
        
        for i, drift_type in enumerate(self.drift_types):
            filename = self.get_filename_pattern(dataset_name, drift_type)
            if not filename:
                continue
                
            X, y = self.load_drift_data(filename)
            if X is None:
                continue
            
            mid_point = len(X) // 2
            before_mean = np.mean(X[:mid_point])
            after_mean = np.mean(X[mid_point:])
            intensity = abs(after_mean - before_mean)
            
            drift_intensities.append(intensity)
            drift_names.append(self.drift_names[drift_type])
        
        bars = ax4.bar(drift_names, drift_intensities, color=self.colors[:len(drift_names)])
        ax4.set_title('Drift Intensity Comparison', fontsize=14)
        ax4.set_ylabel('Feature Change Magnitude', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, intensity in zip(bars, drift_intensities):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{intensity:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_detailed_analysis(self, dataset_name, drift_type, save_path=None):
        """Detailed analysis of a specific drift type"""
        filename = self.get_filename_pattern(dataset_name, drift_type)
        if not filename:
            print(f"Unsupported drift type: {drift_type}")
            return
        
        X, y = self.load_drift_data(filename)
        if X is None:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Feature distribution over time
        ax1 = axes[0, 0]
        window_size = 100
        n_windows = len(X) // window_size
        
        window_means = []
        window_stds = []
        positions = []
        
        for w in range(n_windows):
            start_idx = w * window_size
            end_idx = min((w + 1) * window_size, len(X))
            window_X = X[start_idx:end_idx]
            window_means.append(np.mean(window_X))
            window_stds.append(np.std(window_X))
            positions.append(start_idx)
        
        ax1.plot(positions, window_means, 'o-', color=self.colors[0], linewidth=2, label='Mean')
        ax1.fill_between(positions, 
                        np.array(window_means) - np.array(window_stds), 
                        np.array(window_means) + np.array(window_stds), 
                        alpha=0.3, color=self.colors[0])
        
        ax1.set_title('Feature Distribution Over Time', fontsize=12)
        ax1.set_xlabel('Sample Position')
        ax1.set_ylabel('Feature Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 2D PCA visualization
        ax2 = axes[0, 1]
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        mid_point = len(X_2d) // 2
        ax2.scatter(X_2d[:mid_point, 0], X_2d[:mid_point, 1], 
                   c='blue', alpha=0.6, s=15, label='Before Drift')
        ax2.scatter(X_2d[mid_point:, 0], X_2d[mid_point:, 1], 
                   c='red', alpha=0.6, s=15, marker='^', label='After Drift')
        
        ax2.set_title('2D PCA Visualization', fontsize=12)
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Classification performance
        ax3 = axes[0, 2]
        scaler = StandardScaler()
        clf = LogisticRegression(random_state=42, max_iter=1000)
        
        train_X = X[:200]
        train_y = y[:200]
        train_X_scaled = scaler.fit_transform(train_X)
        clf.fit(train_X_scaled, train_y)
        
        accuracies = []
        for w in range(n_windows):
            start_idx = w * window_size
            end_idx = min((w + 1) * window_size, len(X))
            window_X = X[start_idx:end_idx]
            window_y = y[start_idx:end_idx]
            
            if len(window_X) > 0:
                window_X_scaled = scaler.transform(window_X)
                y_pred = clf.predict(window_X_scaled)
                accuracy = accuracy_score(window_y, y_pred)
                accuracies.append(accuracy)
        
        ax3.plot(positions, accuracies, 'o-', color=self.colors[1], linewidth=2)
        ax3.set_title('Classification Performance', fontsize=12)
        ax3.set_xlabel('Sample Position')
        ax3.set_ylabel('Accuracy')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0.5, 1.0)
        
        # 4. Label distribution
        ax4 = axes[1, 0]
        n_classes = len(np.unique(y))
        label_distributions = []
        
        for w in range(n_windows):
            start_idx = w * window_size
            end_idx = min((w + 1) * window_size, len(X))
            window_y = y[start_idx:end_idx]
            
            label_counts = np.bincount(window_y.astype(int), minlength=n_classes)
            label_dist = label_counts / len(window_y)
            label_distributions.append(label_dist)
        
        label_distributions = np.array(label_distributions)
        
        for class_idx in range(n_classes):
            ax4.plot(positions, label_distributions[:, class_idx], 
                    'o-', linewidth=2, markersize=4, label=f'Class {class_idx}')
        
        ax4.set_title('Label Distribution Changes', fontsize=12)
        ax4.set_xlabel('Sample Position')
        ax4.set_ylabel('Label Proportion')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        # 5. Change intensity
        ax5 = axes[1, 1]
        change_intensity = []
        for i in range(1, len(window_means)):
            change_intensity.append(abs(window_means[i] - window_means[i-1]))
        
        ax5.bar(range(len(change_intensity)), change_intensity, 
               color=self.colors[2], alpha=0.7)
        ax5.set_title('Change Intensity Distribution', fontsize=12)
        ax5.set_xlabel('Time Window')
        ax5.set_ylabel('Change Magnitude')
        ax5.grid(True, alpha=0.3)
        
        # 6. Statistics summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        total_samples = len(X)
        n_features = X.shape[1]
        drift_start = int(total_samples * 0.5)
        
        before_stats = {
            'mean': np.mean(X[:drift_start]),
            'std': np.std(X[:drift_start]),
        }
        after_stats = {
            'mean': np.mean(X[drift_start:]),
            'std': np.std(X[drift_start:]),
        }
        
        stats_text = f"""
Dataset Statistics:
━━━━━━━━━━━━━━━━━━━━
Total Samples: {total_samples:,}
Features: {n_features}
Classes: {n_classes}
Drift Type: {self.drift_names[drift_type]}

Before Drift:
• Mean: {before_stats['mean']:.4f}
• Std: {before_stats['std']:.4f}

After Drift:
• Mean: {after_stats['mean']:.4f}
• Std: {after_stats['std']:.4f}

Changes:
• Mean Δ: {abs(after_stats['mean'] - before_stats['mean']):.4f}
• Std Δ: {abs(after_stats['std'] - before_stats['std']):.4f}
        """
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.suptitle(f'{dataset_name} - {self.drift_names[drift_type]} Detailed Analysis', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_all_visualizations(self, dataset_name="Waveform", output_dir='./drift_visualizations'):
        """Generate all visualization charts"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("🎨 Generating concept drift visualizations...")
        
        # 1. Overview comparison
        print("📊 Generating overview comparison...")
        self.plot_overview_comparison(
            dataset_name, save_path=os.path.join(output_dir, 'overview_comparison.png')
        )
        
        # 2. Detailed analysis for each drift type
        for drift_type in self.drift_types:
            filename = self.get_filename_pattern(dataset_name, drift_type)
            if filename and os.path.exists(os.path.join(self.data_dir, filename)):
                print(f"📈 Generating detailed analysis for {drift_type}...")
                self.plot_detailed_analysis(
                    dataset_name, drift_type, 
                    save_path=os.path.join(output_dir, f'detailed_{drift_type}.png')
                )
        
        print(f"✅ All visualizations saved to: {output_dir}")
    
    def print_available_datasets(self):
        """Print available datasets and their drift types"""
        datasets = self.list_available_datasets()
        
        if not datasets:
            print("❌ No datasets found")
            print("Please run drift injection first to generate datasets")
            return
        
        print("\n📋 Available Datasets and Drift Types:")
        print("=" * 50)
        for dataset, drifts in datasets.items():
            print(f"\n📊 {dataset}:")
            for drift in drifts:
                filename = self.get_filename_pattern(dataset, drift)
                print(f"   ✓ {drift:12} - {filename}") 