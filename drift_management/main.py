# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
概念漂移注入主程序
使用方法：
1. 直接运行脚本查看演示
2. 导入模块并自定义参数
"""

import os
from .utils.drift_utils import DriftInjectionUtil


def demo_basic():
    """基本演示：使用预定义配置"""
    print("🌊 Basic demonstration: predefined drift configuration")
    print("=" * 50)

    # 初始化工具类
    util = DriftInjectionUtil()

    # 获取预定义配置
    configs = util.get_predefined_configs()

    # 演示参数
    dataset_name = "SEA"
    total_samples = 100000
    chunk_size = 1
    random_seed = 42

    # 处理每种漂移类型
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n🔄 [{i}/{len(configs)}] {config['description']}")

        df, save_path = util.inject_single_drift(
            dataset_name=dataset_name,
            drift_type=config['drift_type'],
            drift_severity=config['drift_severity'],
            drift_position=config['drift_position'],
            drift_width=config.get('drift_width', 0.5),
            n_cycles=config.get('n_cycles', 3),
            total_samples=total_samples,
            chunk_size=chunk_size,
            random_seed=random_seed
        )

        results.append({
            'dataset': dataset_name,
            'drift_type': config['drift_type'],
            'filename': os.path.basename(save_path),
            'shape': df.shape,
            'path': save_path,
            'status': 'success'
        })

    # 显示结果摘要
    util.print_results_summary(results)
    return results


def demo_custom(dataset_name="SEA", drift_type="sudden", drift_severity=0.8, drift_position=0.4, total_samples=100000, chunk_size=50, random_seed=123):
    """自定义演示：用户指定参数"""
    print("\n🎯 Custom presentation: user specified parameters")
    print("=" * 50)

    # 初始化工具类
    util = DriftInjectionUtil()

    # 自定义参数
    df, save_path = util.inject_single_drift(
        dataset_name=dataset_name,
        drift_type=drift_type,
        drift_severity=drift_severity,
        drift_position=drift_position,
        total_samples=total_samples,
        chunk_size=chunk_size,
        random_seed=random_seed
    )

    print(f"✨ Custom Demo Complete！")
    return df, save_path


def demo_batch(dataset_names=None, drift_configs=None, total_samples=500000, chunk_size=1, random_seed=456):
    """批量演示：多个数据集和配置"""
    print("\n🚀 Batch Demo: Multiple Datasets and Configurations")
    print("=" * 50)

    # 初始化工具类
    util = DriftInjectionUtil()

    # 批量处理
    results = util.batch_inject_drift(
        dataset_names=dataset_names,
        drift_configs=drift_configs,
        total_samples=total_samples,
        chunk_size=chunk_size,
        random_seed=random_seed
    )

    # 显示结果摘要
    util.print_results_summary(results)
    return results


def interactive_demo(choice):
    """交互式演示：让用户选择操作"""
    print("\n🎮 Interactive presentation")
    print("=" * 50)

    if choice == 1:
        demo_basic()
    elif choice == 2:
        demo_custom()
    elif choice == 3:
        demo_batch()
    elif choice == 0:
        print("👋 Bye！")
    else:
        print("❌ Invalid selection, please re-enter")


def main():
    """主函数"""
    print("🌟 概念漂移注入工具")
    print("=" * 50)

    # 运行所有演示
    try:
        # 1. 基本演示
        # demo_basic()

        # 2. 自定义演示
        # demo_custom()

        # 3. 批量演示
        demo_batch()

        print("\n✨ 所有演示完成！")

    except Exception as e:
        print(f"❌ 演示过程中出现错误: {str(e)}")

    # 使用说明
    print("\n📝 使用说明:")
    print("   1. 运行 main() 查看所有演示")
    print("   2. 运行 demo_basic() 查看基本演示")
    print("   3. 运行 demo_custom() 查看自定义演示")
    print("   4. 运行 demo_batch() 查看批量演示")
    print("   5. 运行 interactive_demo() 进入交互模式")


if __name__ == "__main__":
    # 可以选择运行所有演示或交互模式
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_demo()
    else:
        main()
