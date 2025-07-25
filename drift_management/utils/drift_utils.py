#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concept Drift Injection Utility
Provides core functionality for injecting concept drift
"""

import numpy as np
import pandas as pd
import os
from .concept_drift_injection import create_drift_manager, ConceptDriftInjector


class DriftInjectionUtil:
    """Concept Drift Injection Utility Class"""

    def __init__(self, save_dir="./drift_datasets"):
        """Initialize the utility class

        Args:
            save_dir: Path to save directory
        """
        self.save_dir = save_dir
        self._ensure_save_dir()

    def _ensure_save_dir(self):
        """Ensure save directory exists"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"📁 Created directory: {self.save_dir}")

    def inject_single_drift(self, dataset_name, drift_type, drift_severity=0.5,
                            drift_position=0.5, drift_width=0.1, n_cycles=3,
                            total_samples=5000, chunk_size=100, random_seed=42):
        """Inject single concept drift

        Args:
            dataset_name: Dataset name
            drift_type: Drift type ('sudden', 'gradual', 'incremental', 'recurring')
            drift_severity: Drift intensity (0-1)
            drift_position: Drift position (0-1)
            drift_width: Drift width (0-1, only for gradual drift)
            n_cycles: Number of cycles (only for recurring drift)
            total_samples: Total number of samples
            chunk_size: Batch processing size
            random_seed: Random seed

        Returns:
            tuple: (DataFrame, save path)
        """
        print(f"🎯 Injecting {drift_type} drift to {dataset_name}")

        # Create drift manager
        drift_manager = create_drift_manager(
            drift_type=drift_type,
            drift_severity=drift_severity,
            drift_position=drift_position,
            drift_width=drift_width,
            n_cycles=n_cycles,
            random_seed=random_seed
        )

        # Create drift injector
        injector = ConceptDriftInjector(drift_manager, chunk_size)

        # Generate filename and path
        filename = drift_manager.get_drift_filename(dataset_name)
        save_path = os.path.join(self.save_dir, filename)

        # Inject drift and save
        df = injector.inject_drift_to_dataset(dataset_name, total_samples, save_path)

        print(f"   ✅ Completed! Dataset shape: {df.shape}")
        print(f"   📄 Saved to: {filename}")

        return df, save_path

    def batch_inject_drift(self, dataset_names, drift_configs,
                           total_samples=5000, chunk_size=100, random_seed=42):
        """Batch inject concept drifts

        Args:
            dataset_names: List of dataset names
            drift_configs: List of drift configurations
            total_samples: Total number of samples
            chunk_size: Batch processing size
            random_seed: Random seed

        Returns:
            list: List of processing results
        """
        print(f"🚀 Batch injecting concept drifts")
        print(f"   Datasets: {len(dataset_names)}, Configurations: {len(drift_configs)}")
        print(f"   Total tasks: {len(dataset_names) * len(drift_configs)}")

        results = []
        task_num = 0
        total_tasks = len(dataset_names) * len(drift_configs)

        for dataset_name in dataset_names:
            for config in drift_configs:
                task_num += 1
                print(f"\n📋 [{task_num}/{total_tasks}] {dataset_name} + {config['drift_type']}")

                try:
                    df, save_path = self.inject_single_drift(
                        dataset_name=dataset_name,
                        drift_type=config['drift_type'],
                        drift_severity=config['drift_severity'],
                        drift_position=config['drift_position'],
                        drift_width=config.get('drift_width', 0.1),
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

                except Exception as e:
                    print(f"   ❌ Failed: {str(e)}")
                    results.append({
                        'dataset': dataset_name,
                        'drift_type': config['drift_type'],
                        'filename': f"FAILED_{dataset_name}_{config['drift_type']}",
                        'shape': None,
                        'path': None,
                        'status': 'failed',
                        'error': str(e)
                    })

        # Show summary
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'failed')
        print(f"\n🎉 Batch processing complete! Success: {successful}, Failed: {failed}")

        return results

    def get_predefined_configs(self):
        """Get predefined drift configurations

        Returns:
            list: List of drift configurations
        """
        return [
            {
                'drift_type': 'sudden',
                'drift_severity': 0.6,
                'drift_position': 0.5,
                'description': 'Sudden drift: abrupt change at 50% position, intensity 0.6'
            },
            {
                'drift_type': 'gradual',
                'drift_severity': 0.6,
                'drift_position': 0.5,
                'drift_width': 0.2,
                'description': 'Gradual drift: progressive change between 40%-60% position, intensity 0.6'
            },
            {
                'drift_type': 'incremental',
                'drift_severity': 0.6,
                'drift_position': 0.3,
                'description': 'Incremental drift: incremental change starting from 30% position, intensity 0.6'
            },
            {
                'drift_type': 'recurring',
                'drift_severity': 0.6,
                'drift_position': 0.5,
                'n_cycles': 4,
                'description': 'Recurring drift: 4 cycles, intensity 0.6'
            }
        ]

    def print_results_summary(self, results):
        """Print results summary

        Args:
            results: List of processing results
        """
        print(f"\n📊 Results Summary:")
        print("-" * 60)
        for result in results:
            status_icon = "✅" if result['status'] == 'success' else "❌"
            if result['status'] == 'success':
                print(
                    f"{status_icon} {result['dataset']:12} | {result['drift_type']:12} | {str(result['shape']):12} | {result['filename']}")
            else:
                print(
                    f"{status_icon} {result['dataset']:12} | {result['drift_type']:12} | Failed | {result.get('error', 'Unknown error')}")