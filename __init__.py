"""
EEG Analysis Package

A package for EEG data analysis using MNE-Python.
"""

__version__ = '4.1.0'

# 導入主要模組
from . import data_io
from . import montage
from . import signal_processing
from . import ica_analysis
from . import preprocessing
from . import epochs
from . import spectral_analysis
from . import roi_analysis
from . import utils

# 視覺化獨立模組
try:
    from . import asrt_visualization
except ImportError:
    pass  # 視覺化模組可選

# 導入常用函數（方便直接使用）
from .data_io import load_bids_eeg, load_cnt_file
from .epochs import create_response_locked_epochs, create_stimulus_locked_epochs
from .spectral_analysis import (
    compute_roi_power_with_freq_baseline,
    compute_fft_power
)
from .roi_analysis import define_roi_channels

# 定義公開的 API
__all__ = [
    'data_io',
    'montage',
    'signal_processing',
    'ica_analysis',
    'preprocessing',
    'epochs',
    'spectral_analysis',
    'roi_analysis',
    'utils',
    # 常用函數
    'load_bids_eeg',
    'load_cnt_file',
    'create_response_locked_epochs',
    'create_stimulus_locked_epochs',
    'compute_roi_power_with_freq_baseline',
    'compute_fft_power',
    'define_roi_channels',
]

# 顯示版本信息
def get_version():
    """返回套件版本"""
    return __version__

# 套件初始化時的設定
import warnings
import matplotlib
matplotlib.use('TkAgg')  # 設定後端
warnings.filterwarnings('ignore', category=DeprecationWarning)

print(f"MNE-Python Analysis Package v{__version__} loaded successfully")