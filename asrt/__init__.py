"""
ASRT EEG Analysis Module

包含 ASRT 實驗的所有分析功能：
- FFT 功率分析
- 地形圖繪製
- ERSP 時頻分析
- ASRT 工作流程（從 main.py 移植）
"""

# FFT 分析
from .fft_analysis import (
    asrt_visualization,
    asrt_wholebrain_fft_analysis
)

# 地形圖
from .topomap import (
    asrt_testing_phase_topomap,
    asrt_testing_phase_detailed_topomap
)

# ERSP 核心分析
from .ersp import (
    asrt_ersp_analysis,
    asrt_ersp_comparison,
    asrt_ersp_full_analysis
)

# ERSP 視覺化（通常不需要直接導出，但可以選擇性導出）
from .ersp_plots import (
    plot_ersp_lum2023_style,
    plot_ersp_comparison,
    plot_learning_comparison,
    plot_testing_comparison
)

# ASRT 工作流程（從 main.py 移植）
from .workflows import (
    asrt_complete_analysis,
    asrt_roi_spectral_analysis,
    asrt_block_comparison,
    asrt_artifact_rejection
)

__all__ = [
    # FFT
    'asrt_visualization',
    'asrt_wholebrain_fft_analysis',
    # Topomap
    'asrt_testing_phase_topomap',
    'asrt_testing_phase_detailed_topomap',
    # ERSP
    'asrt_ersp_analysis',
    'asrt_ersp_comparison',
    'asrt_ersp_full_analysis',
    # ERSP Plots
    'plot_ersp_lum2023_style',
    'plot_ersp_comparison',
    'plot_learning_comparison',
    'plot_testing_comparison',
    # Workflows
    'asrt_complete_analysis',
    'asrt_roi_spectral_analysis',
    'asrt_block_comparison',
    'asrt_artifact_rejection',
]

__version__ = '2.1'
