import numpy as np
import pandas as pd
import time
from skmultiflow.data import DataStream, WaveformGenerator, SEAGenerator, HyperplaneGenerator
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import os
import warnings
warnings.filterwarnings('ignore')

class BaseDriftManager:
    """概念漂移管理器基类"""
    def __init__(self, drift_type='sudden', drift_severity=0.5, drift_position=0.5, 
                 drift_width=0.1, random_seed=None):
        """
        参数:
        - drift_type: 漂移类型 ('sudden', 'gradual', 'incremental', 'recurring')
        - drift_severity: 漂移强度 (0-1之间, 0表示无漂移, 1表示完全漂移)
        - drift_position: 漂移位置 (0-1之间, 表示在整个数据流中的位置)
        - drift_width: 漂移宽度 (0-1之间, 仅对gradual drift有效)
        - random_seed: 随机种子
        """
        self.drift_type = drift_type
        self.drift_severity = drift_severity
        self.drift_position = drift_position
        self.drift_width = drift_width
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def apply_drift(self, X, y, sample_idx, total_samples):
        """应用概念漂移到数据"""
        raise NotImplementedError
    
    def get_drift_filename(self, dataset_name):
        """生成包含漂移信息的文件名"""
        return f"{dataset_name}_{self.drift_type}_severity{self.drift_severity}_pos{self.drift_position}_width{self.drift_width}.csv"


class SuddenDriftManager(BaseDriftManager):
    """突然漂移管理器"""
    def __init__(self, drift_type='sudden', drift_severity=0.5, drift_position=0.5, 
                 drift_width=0.1, random_seed=None):
        super().__init__(drift_type, drift_severity, drift_position, drift_width, random_seed)
        self.drift_occurred = False
    
    def apply_drift(self, X, y, sample_idx, total_samples):
        """应用突然漂移"""
        # 计算漂移点
        drift_point = int(total_samples * self.drift_position)
        
        if sample_idx >= drift_point and not self.drift_occurred:
            self.drift_occurred = True
            print(f"Sudden drift occurs in the sample {sample_idx}")
        
        if self.drift_occurred:
            # 应用特征空间漂移
            X_drifted = self._apply_feature_drift(X)
            # 应用标签漂移
            y_drifted = self._apply_label_drift(y)
            return X_drifted, y_drifted
        
        return X, y
    
    def _apply_feature_drift(self, X):
        """应用特征空间漂移"""
        X_drifted = X.copy()
        n_features = X.shape[1]
        
        # 如果没有特征，无法进行特征漂移
        if n_features == 0:
            return X_drifted
        
        # 大幅增强特征漂移：至少修改70%的特征
        min_features = max(1, int(n_features * 0.7))
        max_features = max(min_features, int(n_features * self.drift_severity))
        n_drift_features = np.random.randint(min_features, max_features + 1)
        drift_features = np.random.choice(n_features, n_drift_features, replace=False)
        
        for feature_idx in drift_features:
            # 大幅增强漂移效果
            if np.random.random() < 0.25:
                # 添加强高斯噪声（大幅增强噪声强度）
                noise_std = self.drift_severity * 5.0  # 增强5倍
                noise = np.random.normal(0, noise_std, X.shape[0])
                X_drifted[:, feature_idx] += noise
            elif np.random.random() < 0.5:
                # 添加强偏移（大幅增强偏移强度）
                feature_std = np.std(X[:, feature_idx])
                if feature_std > 0:
                    offset = self.drift_severity * feature_std * 4.0  # 增强4倍
                    X_drifted[:, feature_idx] += offset
            else:
                # 添加缩放变换（大幅增强：改变特征分布）
                feature_mean = np.mean(X[:, feature_idx])
                feature_std = np.std(X[:, feature_idx])
                if feature_std > 0:
                    scale_factor = 1.0 + self.drift_severity * 4.0  # 增强4倍
                    X_drifted[:, feature_idx] = (X_drifted[:, feature_idx] - feature_mean) * scale_factor + feature_mean
        
        return X_drifted
    
    def _apply_label_drift(self, y):
        """应用标签漂移"""
        y_drifted = y.copy()
        unique_labels = np.unique(y)
        n_classes = len(unique_labels)
        
        # 如果只有一个类别，无法进行标签漂移
        if n_classes <= 1:
            return y_drifted
        
        # 降低标签漂移强度
        base_flip_prob = self.drift_severity * 0.15  # 降低到15%
        # 根据类别数量调整翻转概率
        if n_classes == 2:
            flip_prob = base_flip_prob  # 二分类：直接翻转
        else:
            flip_prob = base_flip_prob * 0.7  # 多分类：进一步降低概率
        
        # 批量翻转标签以提高效率
        flip_mask = np.random.random(len(y)) < flip_prob
        flip_indices = np.where(flip_mask)[0]
        
        for idx in flip_indices:
            available_labels = [label for label in unique_labels if label != y[idx]]
            if len(available_labels) > 0:
                y_drifted[idx] = np.random.choice(available_labels)
        
        return y_drifted


class GradualDriftManager(BaseDriftManager):
    """渐进漂移管理器"""
    def __init__(self, drift_type='gradual', drift_severity=0.5, drift_position=0.5, 
                 drift_width=0.1, random_seed=None):
        super().__init__(drift_type, drift_severity, drift_position, drift_width, random_seed)
    
    def apply_drift(self, X, y, sample_idx, total_samples):
        """应用渐进漂移"""
        # 计算漂移开始和结束点
        drift_start = int(total_samples * (self.drift_position - self.drift_width/2))
        drift_end = int(total_samples * (self.drift_position + self.drift_width/2))
        
        if drift_start <= sample_idx <= drift_end:
            # 计算当前漂移程度
            progress = (sample_idx - drift_start) / (drift_end - drift_start)
            current_severity = self.drift_severity * progress
            
            # 应用漂移
            X_drifted = self._apply_gradual_feature_drift(X, current_severity)
            y_drifted = self._apply_gradual_label_drift(y, current_severity)
            return X_drifted, y_drifted
        
        return X, y
    
    def _apply_gradual_feature_drift(self, X, current_severity):
        """应用渐进特征漂移"""
        X_drifted = X.copy()
        n_features = X.shape[1]
        
        # 如果没有特征，无法进行特征漂移
        if n_features == 0:
            return X_drifted
        
        # 大幅增强渐进漂移：确保至少修改60%的特征
        min_features = max(1, int(n_features * 0.6))
        max_features = max(min_features, int(n_features * current_severity))
        n_drift_features = np.random.randint(min_features, max_features + 1)
        drift_features = np.random.choice(n_features, n_drift_features, replace=False)
        
        for feature_idx in drift_features:
            # 大幅增强渐进漂移效果
            if np.random.random() < 0.3:
                # 渐进噪声（大幅增强强度）
                noise_std = current_severity * 4.0
                noise = np.random.normal(0, noise_std, X.shape[0])
                X_drifted[:, feature_idx] += noise
            elif np.random.random() < 0.6:
                # 渐进偏移（大幅增强强度）
                feature_std = np.std(X[:, feature_idx])
                if feature_std > 0:
                    offset = current_severity * feature_std * 3.5
                    X_drifted[:, feature_idx] += offset
            else:
                # 渐进缩放（大幅增强）
                feature_mean = np.mean(X[:, feature_idx])
                feature_std = np.std(X[:, feature_idx])
                if feature_std > 0:
                    scale_factor = 1.0 + current_severity * 3.0
                    X_drifted[:, feature_idx] = (X_drifted[:, feature_idx] - feature_mean) * scale_factor + feature_mean
        
        return X_drifted
    
    def _apply_gradual_label_drift(self, y, current_severity):
        """应用渐进标签漂移"""
        y_drifted = y.copy()
        unique_labels = np.unique(y)
        n_classes = len(unique_labels)
        
        # 如果只有一个类别，无法进行标签漂移
        if n_classes <= 1:
            return y_drifted
        
        # 降低渐进标签漂移
        base_flip_prob = current_severity * 0.12  # 降低到12%
        if n_classes == 2:
            flip_prob = base_flip_prob
        else:
            flip_prob = base_flip_prob * 0.6
        
        # 批量翻转标签
        flip_mask = np.random.random(len(y)) < flip_prob
        flip_indices = np.where(flip_mask)[0]
        
        for idx in flip_indices:
            available_labels = [label for label in unique_labels if label != y[idx]]
            if len(available_labels) > 0:
                y_drifted[idx] = np.random.choice(available_labels)
        
        return y_drifted


class IncrementalDriftManager(BaseDriftManager):
    """增量漂移管理器"""
    def __init__(self, drift_type='incremental', drift_severity=0.5, drift_position=0.5, 
                 drift_width=0.1, random_seed=None):
        super().__init__(drift_type, drift_severity, drift_position, drift_width, random_seed)
        self.accumulated_drift = 0
    
    def apply_drift(self, X, y, sample_idx, total_samples):
        """应用增量漂移"""
        # 计算漂移开始点
        drift_start = int(total_samples * self.drift_position)
        
        if sample_idx >= drift_start:
            # 增量增加漂移强度
            increment = self.drift_severity / (total_samples - drift_start)
            self.accumulated_drift = min(self.drift_severity, self.accumulated_drift + increment)
            
            # 应用漂移
            X_drifted = self._apply_incremental_feature_drift(X, self.accumulated_drift)
            y_drifted = self._apply_incremental_label_drift(y, self.accumulated_drift)
            return X_drifted, y_drifted
        
        return X, y
    
    def _apply_incremental_feature_drift(self, X, current_severity):
        """应用增量特征漂移"""
        X_drifted = X.copy()
        n_features = X.shape[1]
        
        # 如果没有特征，无法进行特征漂移
        if n_features == 0:
            return X_drifted
        
        # 大幅增强增量漂移：确保至少修改80%的特征
        min_features = max(1, int(n_features * 0.8))
        max_features = max(min_features, int(n_features * current_severity))
        n_drift_features = np.random.randint(min_features, max_features + 1)
        drift_features = np.random.choice(n_features, n_drift_features, replace=False)
        
        for feature_idx in drift_features:
            # 大幅增强增量漂移效果
            if np.random.random() < 0.25:
                # 增量噪声（大幅增强强度）
                noise_std = current_severity * 6.0
                noise = np.random.normal(0, noise_std, X.shape[0])
                X_drifted[:, feature_idx] += noise
            elif np.random.random() < 0.55:
                # 增量偏移（大幅增强强度）
                feature_std = np.std(X[:, feature_idx])
                if feature_std > 0:
                    offset = current_severity * feature_std * 5.0
                    X_drifted[:, feature_idx] += offset
            else:
                # 增量缩放（大幅增强）
                feature_mean = np.mean(X[:, feature_idx])
                feature_std = np.std(X[:, feature_idx])
                if feature_std > 0:
                    scale_factor = 1.0 + current_severity * 4.5
                    X_drifted[:, feature_idx] = (X_drifted[:, feature_idx] - feature_mean) * scale_factor + feature_mean
        
        return X_drifted
    
    def _apply_incremental_label_drift(self, y, current_severity):
        """应用增量标签漂移"""
        y_drifted = y.copy()
        unique_labels = np.unique(y)
        n_classes = len(unique_labels)
        
        # 如果只有一个类别，无法进行标签漂移
        if n_classes <= 1:
            return y_drifted
        
        # 降低增量标签漂移
        base_flip_prob = current_severity * 0.10  # 降低到10%
        if n_classes == 2:
            flip_prob = base_flip_prob
        else:
            flip_prob = base_flip_prob * 0.5
        
        # 批量翻转标签
        flip_mask = np.random.random(len(y)) < flip_prob
        flip_indices = np.where(flip_mask)[0]
        
        for idx in flip_indices:
            available_labels = [label for label in unique_labels if label != y[idx]]
            if len(available_labels) > 0:
                y_drifted[idx] = np.random.choice(available_labels)
        
        return y_drifted


class RecurringDriftManager(BaseDriftManager):
    """循环漂移管理器"""
    def __init__(self, drift_type='recurring', drift_severity=0.5, drift_position=0.5, 
                 drift_width=0.1, n_cycles=3, random_seed=None):
        super().__init__(drift_type, drift_severity, drift_position, drift_width, random_seed)
        self.n_cycles = n_cycles
        self.cycle_length = 0
    
    def apply_drift(self, X, y, sample_idx, total_samples):
        """应用循环漂移"""
        # 计算循环长度
        if self.cycle_length == 0:
            self.cycle_length = int(total_samples / self.n_cycles)
        
        # 计算当前在第几个循环中
        current_cycle = (sample_idx // self.cycle_length) % 2  # 0或1，表示两种状态
        
        if current_cycle == 1:
            # 应用漂移
            X_drifted = self._apply_recurring_feature_drift(X)
            y_drifted = self._apply_recurring_label_drift(y)
            return X_drifted, y_drifted
        
        return X, y
    
    def _apply_recurring_feature_drift(self, X):
        """应用循环特征漂移"""
        X_drifted = X.copy()
        n_features = X.shape[1]
        
        # 如果没有特征，无法进行特征漂移
        if n_features == 0:
            return X_drifted
        
        # 大幅增强循环漂移：确保至少修改90%的特征
        min_features = max(1, int(n_features * 0.9))
        max_features = max(min_features, int(n_features * self.drift_severity))
        n_drift_features = np.random.randint(min_features, max_features + 1)
        drift_features = np.random.choice(n_features, n_drift_features, replace=False)
        
        for feature_idx in drift_features:
            # 大幅增强循环漂移效果
            if np.random.random() < 0.2:
                # 循环噪声（大幅增强强度）
                noise_std = self.drift_severity * 7.0
                noise = np.random.normal(0, noise_std, X.shape[0])
                X_drifted[:, feature_idx] += noise
            elif np.random.random() < 0.45:
                # 循环偏移（大幅增强强度）
                feature_std = np.std(X[:, feature_idx])
                if feature_std > 0:
                    offset = self.drift_severity * feature_std * 6.0
                    X_drifted[:, feature_idx] += offset
            else:
                # 循环缩放（大幅增强）
                feature_mean = np.mean(X[:, feature_idx])
                feature_std = np.std(X[:, feature_idx])
                if feature_std > 0:
                    scale_factor = 1.0 + self.drift_severity * 5.0
                    X_drifted[:, feature_idx] = (X_drifted[:, feature_idx] - feature_mean) * scale_factor + feature_mean
        
        return X_drifted
    
    def _apply_recurring_label_drift(self, y):
        """应用循环标签漂移"""
        y_drifted = y.copy()
        unique_labels = np.unique(y)
        n_classes = len(unique_labels)
        
        # 如果只有一个类别，无法进行标签漂移
        if n_classes <= 1:
            return y_drifted
        
        # 降低循环标签漂移
        base_flip_prob = self.drift_severity * 0.08  # 降低到8%
        if n_classes == 2:
            flip_prob = base_flip_prob
        else:
            flip_prob = base_flip_prob * 0.4
        
        # 批量翻转标签
        flip_mask = np.random.random(len(y)) < flip_prob
        flip_indices = np.where(flip_mask)[0]
        
        for idx in flip_indices:
            available_labels = [label for label in unique_labels if label != y[idx]]
            if len(available_labels) > 0:
                y_drifted[idx] = np.random.choice(available_labels)
        
        return y_drifted


class ConceptDriftInjector:
    """概念漂移注入器"""
    def __init__(self, drift_manager, chunk_size=100):
        self.drift_manager = drift_manager
        self.chunk_size = chunk_size
    
    def inject_drift_to_stream(self, stream, total_samples, save_path=None):
        """向数据流注入概念漂移"""
        X_all = []
        y_all = []
        
        sample_idx = 0
        print(f"Start injecting{self.drift_manager.drift_type}drift...")
        
        while sample_idx < total_samples and stream.has_more_samples():
            # 获取下一批数据
            X_chunk, y_chunk = stream.next_sample(self.chunk_size)
            
            # 应用漂移
            X_drifted, y_drifted = self.drift_manager.apply_drift(
                X_chunk, y_chunk, sample_idx, total_samples
            )
            
            X_all.append(X_drifted)
            y_all.append(y_drifted)
            
            sample_idx += self.chunk_size
            
            # 显示进度
            if sample_idx % 1000 == 0:
                print(f"Processed {sample_idx}/{total_samples} samples")
        
        # 合并所有数据
        X_final = np.vstack(X_all)
        y_final = np.hstack(y_all)
        
        # 创建DataFrame
        df = pd.DataFrame(X_final)
        df['label'] = y_final
        
        # 保存到文件
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Drift data has been saved to: {save_path}")
        
        return df
    
    def inject_drift_to_dataset(self, dataset_name, total_samples, save_path=None):
        """向数据集注入概念漂移"""
        # 加载数据集
        if dataset_name in ["Waveform", "SEA", "Hyperplane"]:
            # 使用skmultiflow生成器
            if dataset_name == "Waveform":
                stream = WaveformGenerator(random_state=self.drift_manager.random_seed)
            elif dataset_name == "SEA":
                stream = SEAGenerator(random_state=self.drift_manager.random_seed)
            elif dataset_name == "Hyperplane":
                stream = HyperplaneGenerator(random_state=self.drift_manager.random_seed)
        else:
            # 从文件加载
            data_path = f'./datasets/{dataset_name}.csv'
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
                data_values = data.values
                X = data_values[:, :-1]
                y = data_values[:, -1].astype(int)
                stream = DataStream(X, y)
            else:
                raise FileNotFoundError(f"The dataset file {data_path} does not exist")
        
        # 注入漂移
        return self.inject_drift_to_stream(stream, total_samples, save_path)


def create_drift_manager(drift_type, drift_severity=0.5, drift_position=0.5, 
                        drift_width=0.1, n_cycles=3, random_seed=None):
    """创建漂移管理器的工厂函数"""
    if drift_type == 'sudden':
        return SuddenDriftManager(drift_type, drift_severity, drift_position, 
                                drift_width, random_seed)
    elif drift_type == 'gradual':
        return GradualDriftManager(drift_type, drift_severity, drift_position, 
                                 drift_width, random_seed)
    elif drift_type == 'incremental':
        return IncrementalDriftManager(drift_type, drift_severity, drift_position, 
                                     drift_width, random_seed)
    elif drift_type == 'recurring':
        return RecurringDriftManager(drift_type, drift_severity, drift_position, 
                                   drift_width, n_cycles, random_seed)
    else:
        raise ValueError(f"Unsupported drift type: {drift_type}")


def main():
    """主函数，演示概念漂移注入"""
    print("=== 概念漂移注入演示 ===")
    
    # 配置参数
    dataset_name = "Waveform"  # 可选: Waveform, SEA, Hyperplane, covtype, electricity等
    total_samples = 5000
    chunk_size = 100
    random_seed = 42
    
    # 创建保存目录
    save_dir = "./drift_datasets"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 测试不同类型的漂移
    drift_configs = [
        {'drift_type': 'sudden', 'drift_severity': 0.3, 'drift_position': 0.5},
        {'drift_type': 'gradual', 'drift_severity': 0.5, 'drift_position': 0.5, 'drift_width': 0.2},
        {'drift_type': 'incremental', 'drift_severity': 0.4, 'drift_position': 0.3},
        {'drift_type': 'recurring', 'drift_severity': 0.3, 'drift_position': 0.5, 'n_cycles': 4}
    ]
    
    for config in drift_configs:
        print(f"\n--- 测试 {config['drift_type']} 漂移 ---")
        
        # 创建漂移管理器
        drift_manager = create_drift_manager(
            drift_type=config['drift_type'],
            drift_severity=config['drift_severity'],
            drift_position=config['drift_position'],
            drift_width=config.get('drift_width', 0.1),
            n_cycles=config.get('n_cycles', 3),
            random_seed=random_seed
        )
        
        # 创建漂移注入器
        injector = ConceptDriftInjector(drift_manager, chunk_size)
        
        # 生成文件名
        filename = drift_manager.get_drift_filename(dataset_name)
        save_path = os.path.join(save_dir, filename)
        
        # 注入漂移并保存
        df = injector.inject_drift_to_dataset(dataset_name, total_samples, save_path)
        
        print(f"数据集形状: {df.shape}")
        print(f"保存路径: {save_path}")


if __name__ == "__main__":
    main() 