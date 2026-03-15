"""
User Interface Module for MNE-Python Analysis

提供選單系統、使用者互動和工作流程函數
"""

from .menu import (
    show_data_source_menu,
    show_main_menu,
    show_epochs_analysis_menu,
    display_welcome_message,
    display_subject_info,
    display_processing_history
)

from .prompts import (
    ask_save_confirmation,
    ask_continue,
    validate_channels,
    format_duration
)

from .workflows import (
    # 檢視資料
    display_raw_waveform,
    display_electrode_positions,
    display_psd_plot,
    # 前處理
    run_standard_preprocessing,
    mark_bad_segments_interactive,
    # 進階分析
    run_ica_analysis,
    prepare_microstate_analysis,
    # Epochs
    create_epochs_interactive,
    create_epochs_default,
    display_epochs_info,
    display_epochs_plot,
    compute_epochs_psd,
    compute_epochs_tfr,
    # 儲存
    save_raw_interactive,
    save_epochs_interactive
)

__all__ = [
    # Menu
    'show_data_source_menu',
    'show_main_menu',
    'show_epochs_analysis_menu',
    'display_welcome_message',
    'display_subject_info',
    'display_processing_history',
    # Prompts
    'ask_save_confirmation',
    'ask_continue',
    'validate_channels',
    'format_duration',
    # Workflows
    'display_raw_waveform',
    'display_electrode_positions',
    'display_psd_plot',
    'run_standard_preprocessing',
    'mark_bad_segments_interactive',
    'run_ica_analysis',
    'prepare_microstate_analysis',
    'create_epochs_interactive',
    'create_epochs_default',
    'display_epochs_info',
    'display_epochs_plot',
    'compute_epochs_psd',
    'compute_epochs_tfr',
    'save_raw_interactive',
    'save_epochs_interactive'
]

__version__ = '1.0'