#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils Package for Concept Drift Management
提供概念漂移注入和可视化的功能类
"""

# 导入核心功能类
from .concept_drift_injection import (
    ConceptDriftInjector,
    SuddenDriftManager,
    GradualDriftManager,
    IncrementalDriftManager,
    RecurringDriftManager,
    create_drift_manager
)

from .drift_utils import DriftInjectionUtil

from .visualizer import ConceptDriftVisualizer

# 定义包的公共接口
__all__ = [
    'ConceptDriftInjector',
    'SuddenDriftManager',
    'GradualDriftManager', 
    'IncrementalDriftManager',
    'RecurringDriftManager',
    'create_drift_manager',
    'DriftInjectionUtil',
    'ConceptDriftVisualizer'
]

# 版本信息
__version__ = '1.0.0'
__author__ = 'Concept Drift Management Team'
__description__ = 'Comprehensive toolkit for concept drift injection and visualization' 