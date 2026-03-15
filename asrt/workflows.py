"""
ASRT 工作流程模組 - ASRT Workflows Module

包含 ASRT 實驗專用的分析工作流程函數

所有函數均從 main.py 移植而來
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from scipy import stats
import matplotlib.pyplot as plt

# mne_python_analysis 基礎模組
from mne_python_analysis.epochs import (
    create_stimulus_locked_epochs,
    create_response_locked_epochs,
    separate_trial_types,
    extract_block_epochs,
    epoch_data_asrt
)
from mne_python_analysis.roi_analysis import (
    define_roi_channels,
    average_roi_epochs,
    create_virtual_channel_epochs,
    compare_roi_across_conditions
)
from mne_python_analysis.spectral_analysis import (
    compute_fft_power,
    compute_band_power_multiple_bands,
    compute_tfr_morlet,
    compute_power_with_freq_baseline,
    compute_roi_power_with_freq_baseline,
    exclude_non_eeg_channels
)
from mne_python_analysis.response_lock import (
    align_epochs_to_response,
    compute_response_locked_power,
    validate_response_times
)
from mne_python_analysis.statistical_analysis import (
    compare_regular_vs_random,
    compare_blocks,
    compute_learning_effect,
    aggregate_blocks
)

# ASRT 模組（同一資料夾）
try:
    from .fft_analysis import asrt_visualization, asrt_wholebrain_fft_analysis
    from .topomap import asrt_testing_phase_topomap, asrt_testing_phase_detailed_topomap
    from .ersp import asrt_ersp_analysis, asrt_ersp_comparison, asrt_ersp_full_analysis
    ASRT_MODULES_AVAILABLE = True
    print("✓ ASRT 模組載入成功")
except ImportError as e:
    print(f"⚠️  ASRT 模組導入失敗: {e}")
    ASRT_MODULES_AVAILABLE = False

# 群體分析（可選）
try:
    from mne_python_analysis.group_ersp_analysis import group_ersp_analysis
    HAS_GROUP_ANALYSIS = True
except ImportError:
    HAS_GROUP_ANALYSIS = False


# ============================================================
# ASRT 完整分析
# 來源：main.py 第 477-1059 行（583 行）
# ============================================================

def asrt_complete_analysis(current_raw, subject_id):
    """
    ASRT 完整分析：Stimulus Lock + Response Lock
    
    兩種分析完全獨立：
    - Stimulus Lock: 看刺激後 100-300ms 的 alpha（各自 baseline）
    - Response Lock: 看反應前 -300~50ms 的 theta（各自 baseline）
    
    全腦 FFT（所有電極）+ ROI 分析
    """
    print("\n" + "="*60)
    print("ASRT 完整分析")
    print("Stimulus Lock (Alpha) + Response Lock (Theta)")
    print("="*60)
    
    if not ASRT_MODULES_AVAILABLE:
        print("⚠️  ASRT 模組未安裝，無法執行此功能")
        return None
    
    # 步驟 1: 載入事件和反應時間
    print("\n步驟 1: 載入資料")
    events_file = input("請輸入事件檔案路徑 (留空則從 Raw 提取): ").strip()
    
    if events_file and os.path.exists(events_file):
        events = mne.read_events(events_file)
    else:
        events = mne.find_events(current_raw, stim_channel='STI')
    
    print(f"找到 {len(events)} 個事件")
    
    # 定義事件 ID
    event_id_str = input("請輸入事件ID (格式: regular=1,random=2): ").strip()
    if not event_id_str:
        event_id = {'regular': 1, 'random': 2}
    else:
        event_id = {}
        for pair in event_id_str.split(','):
            name, value = pair.split('=')
            event_id[name.strip()] = int(value.strip())
    
    # 載入反應時間
    rt_file = input("請輸入反應時間檔案路徑 (.txt/.csv): ").strip()
    if not os.path.exists(rt_file):
        print("⚠️  檔案不存在")
        return None
    
    response_times_ms = np.loadtxt(rt_file)
    response_times_s = response_times_ms / 1000.0
    
    # 驗證 RT
    valid_mask, invalid_idx = validate_response_times(response_times_ms, min_rt=100, max_rt=500)
    
    # 步驟 2: 創建 Epochs
    epochs_stim = create_stimulus_locked_epochs(
        current_raw, events, event_id,
        tmin=-0.5, tmax=0.6,
        baseline=(-0.5, -0.1)  # Time-domain baseline
    )
    
    epochs_resp = create_response_locked_epochs(
        current_raw, events, response_times_s, event_id,
        tmin=-0.75, tmax=0.3,
        baseline=(-0.75, -0.25)  # Time-domain baseline
    )
    
    # 分離條件
    epochs_stim_regular = epochs_stim['regular']
    epochs_stim_random = epochs_stim['random']
    epochs_resp_regular = epochs_resp['regular']
    epochs_resp_random = epochs_resp['random']
    
    # 步驟 3: 全腦 FFT 分析
    print("\n步驟 3: ROI 分析（含 frequency-domain baseline）")
    
    # --- Stimulus Lock - Alpha ROI ---
    print("\n【主分析 1】Stimulus Lock - Alpha ROI (100-300ms)")
    alpha_roi = define_roi_channels('alpha')  # O1, Oz, O2, P3, Pz, P4
    print(f"Alpha ROI: {alpha_roi}")
    
    # Regular trials
    alpha_roi_stim_reg_rel, alpha_roi_stim_reg_task, alpha_roi_stim_reg_base = \
        compute_roi_power_with_freq_baseline(
            epochs_stim_regular, alpha_roi,
            fmin=8, fmax=13,
            task_tmin=0.1, task_tmax=0.3,      # 分析窗口：100-300ms
            baseline_tmin=-0.5, baseline_tmax=-0.1  # Baseline：-400~-200ms
        )
    
    # Random trials
    alpha_roi_stim_ran_rel, alpha_roi_stim_ran_task, alpha_roi_stim_ran_base = \
        compute_roi_power_with_freq_baseline(
            epochs_stim_random, alpha_roi,
            fmin=8, fmax=13,
            task_tmin=0.1, task_tmax=0.3,
            baseline_tmin=-0.5, baseline_tmax=-0.1
        )
    
    print(f"  Regular: M(relative)={np.mean(alpha_roi_stim_reg_rel):.4f}")
    print(f"  Random:  M(relative)={np.mean(alpha_roi_stim_ran_rel):.4f}")
    
    # 統計比較（使用相對功率）
    from mne_python_analysis.statistical_analysis import compare_regular_vs_random
    print("\nAlpha ROI 統計比較（相對功率）:")
    compare_regular_vs_random(alpha_roi_stim_reg_rel, alpha_roi_stim_ran_rel)
    
    # --- Stimulus Lock - Theta ROI（對照分析）---
    print("\n【對照分析】Stimulus Lock - Theta ROI (200-400ms)")
    theta_roi = define_roi_channels('theta')  # Fz, FCz, Cz, C3, C4
    print(f"Theta ROI: {theta_roi}")
    
    theta_roi_stim_reg_rel, _, _ = compute_roi_power_with_freq_baseline(
        epochs_stim_regular, theta_roi,
        fmin=4, fmax=8,
        task_tmin=0.1, task_tmax=0.3,
        baseline_tmin=-0.5, baseline_tmax=-0.1
    )
    
    theta_roi_stim_ran_rel, _, _ = compute_roi_power_with_freq_baseline(
        epochs_stim_random, theta_roi,
        fmin=4, fmax=8,
        task_tmin=0.1, task_tmax=0.3,
        baseline_tmin=-0.5, baseline_tmax=-0.1
    )
    
    print(f"  Regular: M(relative)={np.mean(theta_roi_stim_reg_rel):.4f}")
    print(f"  Random:  M(relative)={np.mean(theta_roi_stim_ran_rel):.4f}")
    
    # --- Response Lock - Theta ROI ---
    print("\n【主分析 2】Response Lock - Theta ROI (-300~+50ms)")
    
    theta_roi_resp_reg_rel, theta_roi_resp_reg_task, theta_roi_resp_reg_base = \
        compute_roi_power_with_freq_baseline(
            epochs_resp_regular, theta_roi,
            fmin=4, fmax=8,
            task_tmin=-0.3, task_tmax=0.05,    # 分析窗口：-300~+50ms
            baseline_tmin=-0.75, baseline_tmax=-0.25  # Baseline：-750~-500ms
        )
    
    theta_roi_resp_ran_rel, theta_roi_resp_ran_task, theta_roi_resp_ran_base = \
        compute_roi_power_with_freq_baseline(
            epochs_resp_random, theta_roi,
            fmin=4, fmax=8,
            task_tmin=-0.3, task_tmax=0.05,
            baseline_tmin=-0.75, baseline_tmax=-0.25
        )
    
    print(f"  Regular: M(relative)={np.mean(theta_roi_resp_reg_rel):.4f}")
    print(f"  Random:  M(relative)={np.mean(theta_roi_resp_ran_rel):.4f}")
    
    print("\nTheta ROI 統計比較（相對功率）:")
    compare_regular_vs_random(theta_roi_resp_reg_rel, theta_roi_resp_ran_rel)
    
    # --- Response Lock - Alpha ROI（對照分析）---
    print("\n【對照分析】Response Lock - Alpha ROI (-300~+50ms)")
    
    alpha_roi_resp_reg_rel, _, _ = compute_roi_power_with_freq_baseline(
        epochs_resp_regular, alpha_roi,
        fmin=8, fmax=13,
        task_tmin=-0.3, task_tmax=0.05,
        baseline_tmin=-0.75, baseline_tmax=-0.25
    )
    
    alpha_roi_resp_ran_rel, _, _ = compute_roi_power_with_freq_baseline(
        epochs_resp_random, alpha_roi,
        fmin=8, fmax=13,
        task_tmin=-0.3, task_tmax=0.05,
        baseline_tmin=-0.75, baseline_tmax=-0.25
    )
    
    print(f"  Regular: M(relative)={np.mean(alpha_roi_resp_reg_rel):.4f}")
    print(f"  Random:  M(relative)={np.mean(alpha_roi_resp_ran_rel):.4f}")
    
    # 步驟 4: 統計分析（全腦，逐電極）
    print("\n步驟 4: 全腦分析（排除 HEOG/VEOG/A1/A2）")
    
    # 排除特定電極
    epochs_stim_reg_eeg = exclude_non_eeg_channels(epochs_stim_regular)
    epochs_stim_ran_eeg = exclude_non_eeg_channels(epochs_stim_random)
    epochs_resp_reg_eeg = exclude_non_eeg_channels(epochs_resp_regular)
    epochs_resp_ran_eeg = exclude_non_eeg_channels(epochs_resp_random)
    
    # --- Stimulus Lock - 全腦 Alpha ---
    print("\n全腦分析 1: Stimulus Lock - Alpha (200-400ms)")
    
    alpha_whole_stim_reg_rel, alpha_whole_stim_reg_task, _ = \
        compute_power_with_freq_baseline(
            epochs_stim_reg_eeg,
            fmin=8, fmax=13,
            task_tmin=0.1, task_tmax=0.3,
            baseline_tmin=-0.5, baseline_tmax=-0.1
        )
    
    alpha_whole_stim_ran_rel, alpha_whole_stim_ran_task, _ = \
        compute_power_with_freq_baseline(
            epochs_stim_ran_eeg,
            fmin=8, fmax=13,
            task_tmin=0.1, task_tmax=0.3,
            baseline_tmin=-0.5, baseline_tmax=-0.1
        )
    
    # 逐電極統計
    from scipy import stats
    sig_channels_alpha = []
    for ch_idx, ch_name in enumerate(epochs_stim_reg_eeg.ch_names):
        t_stat, p_val = stats.ttest_ind(
            alpha_whole_stim_reg_rel[:, ch_idx],
            alpha_whole_stim_ran_rel[:, ch_idx]
        )
        if p_val < 0.05:
            sig_channels_alpha.append(ch_name)
            print(f"  {ch_name}: t={t_stat:.3f}, p={p_val:.4f} *")
    
    if not sig_channels_alpha:
        print("  無顯著電極")
    
    # --- Response Lock - 全腦 Theta ---
    print("\n全腦分析 2: Response Lock - Theta (-300~+50ms)")
    
    theta_whole_resp_reg_rel, theta_whole_resp_reg_task, _ = \
        compute_power_with_freq_baseline(
            epochs_resp_reg_eeg,
            fmin=4, fmax=8,
            task_tmin=-0.3, task_tmax=0.05,
            baseline_tmin=-0.75, baseline_tmax=-0.25
        )
    
    theta_whole_resp_ran_rel, theta_whole_resp_ran_task, _ = \
        compute_power_with_freq_baseline(
            epochs_resp_ran_eeg,
            fmin=4, fmax=8,
            task_tmin=-0.3, task_tmax=0.05,
            baseline_tmin=-0.75, baseline_tmax=-0.25
        )
    
    # 逐電極統計
    sig_channels_theta = []
    for ch_idx, ch_name in enumerate(epochs_resp_reg_eeg.ch_names):
        t_stat, p_val = stats.ttest_ind(
            theta_whole_resp_reg_rel[:, ch_idx],
            theta_whole_resp_ran_rel[:, ch_idx]
        )
        if p_val < 0.05:
            sig_channels_theta.append(ch_name)
            print(f"  {ch_name}: t={t_stat:.3f}, p={p_val:.4f} *")
    
    if not sig_channels_theta:
        print("  無顯著電極")
    
    # 步驟 5: 繪製地形圖（相對功率）
    print("\n步驟 5: 繪製地形圖（相對功率）")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # (1,1) Stimulus-Alpha - 主要分析
    diff = np.mean(alpha_whole_stim_reg_rel, axis=0) - np.mean(alpha_whole_stim_ran_rel, axis=0)
    vmax = np.max(np.abs(diff))
    mne.viz.plot_topomap(diff, epochs_stim_reg_eeg.info, axes=axes[0,0], show=False,
                         cmap='RdBu_r', vlim=(-vmax, vmax))
    axes[0,0].set_title('【Main analysis 1】Stimulus Lock - Alpha\n(200-400ms, Regular-Random)\nFreq-baseline corrected')
    
    # (1,2) Stimulus-Theta - 對照
    theta_whole_stim_reg_rel, _, _ = compute_power_with_freq_baseline(
        epochs_stim_reg_eeg, 4, 8, 0.1, 0.3, -0.5, -0.1)
    theta_whole_stim_ran_rel, _, _ = compute_power_with_freq_baseline(
        epochs_stim_ran_eeg, 4, 8, 0.1, 0.3, -0.5, -0.1)
    
    diff = np.mean(theta_whole_stim_reg_rel, axis=0) - np.mean(theta_whole_stim_ran_rel, axis=0)
    vmax = np.max(np.abs(diff))
    mne.viz.plot_topomap(diff, epochs_stim_reg_eeg.info, axes=axes[0,1], show=False,
                         cmap='RdBu_r', vlim=(-vmax, vmax))
    axes[0,1].set_title('【Coresponding analysis 1】Stimulus Lock - Theta\n(200-400ms)')
    
    # (2,1) Response-Theta - 主要分析
    diff = np.mean(theta_whole_resp_reg_rel, axis=0) - np.mean(theta_whole_resp_ran_rel, axis=0)
    vmax = np.max(np.abs(diff))
    mne.viz.plot_topomap(diff, epochs_resp_reg_eeg.info, axes=axes[1,0], show=False,
                         cmap='RdBu_r', vlim=(-vmax, vmax))
    axes[1,0].set_title('【Main analysis 2】Response Lock - Theta\n(-300~+50ms, Regular-Random)\nFreq-baseline corrected')
    
    # (2,2) Response-Alpha - 對照
    diff = np.mean(alpha_roi_resp_reg_rel, axis=0) - np.mean(alpha_roi_resp_ran_rel, axis=0) if alpha_roi_resp_reg_rel.ndim > 1 else 0
    # Note: 如果是 ROI 平均，這裡無法畫 topomap，改用文字說明
    axes[1,1].text(0.5, 0.5, f'【Coresponding analysis 2】Response Lock - Alpha\n(-300~+50ms)\n\nROI Average:\nRegular={np.mean(alpha_roi_resp_reg_rel):.4f}\nRandom={np.mean(alpha_roi_resp_ran_rel):.4f}',
                   ha='center', va='center', fontsize=12)
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{subject_id}_ASRT_complete_freq_baseline.png', dpi=300)
    print(f"✓ 地形圖已儲存: {subject_id}_ASRT_complete_freq_baseline.png")
    plt.show(block=False)
    
    # 步驟 6: 地形圖（2×2）
    def plot_asrt_topomaps(epochs_stim_regular, epochs_stim_random,
                           epochs_resp_regular, epochs_resp_random,
                           subject_id='subject'):
        """
        繪製 ASRT 完整分析地形圖（2×2）
        
        使用 frequency-domain baseline correction 後的相對功率
        
        Parameters
        ----------
        epochs_stim_regular, epochs_stim_random : mne.Epochs
            Stimulus-locked epochs (已排除 HEOG/VEOG/A1/A2)
        epochs_resp_regular, epochs_resp_random : mne.Epochs
            Response-locked epochs (已排除 HEOG/VEOG/A1/A2)
        subject_id : str
            受試者 ID
        """
        from mne_python_analysis.spectral_analysis import compute_power_with_freq_baseline
        
        print("\n步驟 6: 繪製地形圖（2×2）")
        
        # ============================================================
        # 方法 1: 使用相對功率（推薦）
        # ============================================================
        
        # (1) Stimulus Lock - Alpha (主要分析)
        print("  計算 Stimulus-Alpha...")
        alpha_stim_reg_rel, _, _ = compute_power_with_freq_baseline(
            epochs_stim_regular, fmin=8, fmax=13,
            task_tmin=0.1, task_tmax=0.3,
            baseline_tmin=-0.5, baseline_tmax=-0.1
        )
        
        alpha_stim_ran_rel, _, _ = compute_power_with_freq_baseline(
            epochs_stim_random, fmin=8, fmax=13,
            task_tmin=0.1, task_tmax=0.3,
            baseline_tmin=-0.5, baseline_tmax=-0.1
        )
        
        # (2) Stimulus Lock - Theta (對照分析)
        print("  計算 Stimulus-Theta...")
        theta_stim_reg_rel, _, _ = compute_power_with_freq_baseline(
            epochs_stim_regular, fmin=4, fmax=8,
            task_tmin=0.1, task_tmax=0.3,
            baseline_tmin=-0.5, baseline_tmax=-0.1
        )
        
        theta_stim_ran_rel, _, _ = compute_power_with_freq_baseline(
            epochs_stim_random, fmin=4, fmax=8,
            task_tmin=0.1, task_tmax=0.3,
            baseline_tmin=-0.5, baseline_tmax=-0.1
        )
        
        # (3) Response Lock - Theta (主要分析)
        print("  計算 Response-Theta...")
        theta_resp_reg_rel, _, _ = compute_power_with_freq_baseline(
            epochs_resp_regular, fmin=4, fmax=8,
            task_tmin=-0.3, task_tmax=0.05,
            baseline_tmin=-0.75, baseline_tmax=-0.25
        )
        
        theta_resp_ran_rel, _, _ = compute_power_with_freq_baseline(
            epochs_resp_random, fmin=4, fmax=8,
            task_tmin=-0.3, task_tmax=0.05,
            baseline_tmin=-0.75, baseline_tmax=-0.25
        )
        
        # (4) Response Lock - Alpha (對照分析)
        print("  計算 Response-Alpha...")
        alpha_resp_reg_rel, _, _ = compute_power_with_freq_baseline(
            epochs_resp_regular, fmin=8, fmax=13,
            task_tmin=-0.3, task_tmax=0.05,
            baseline_tmin=-0.75, baseline_tmax=-0.25
        )
        
        alpha_resp_ran_rel, _, _ = compute_power_with_freq_baseline(
            epochs_resp_random, fmin=8, fmax=13,
            task_tmin=-0.3, task_tmax=0.05,
            baseline_tmin=-0.75, baseline_tmax=-0.25
        )
        
        # ============================================================
        # 繪製地形圖
        # ============================================================
        
        print("  繪製地形圖...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # --- (1,1) Stimulus-Alpha - 主要分析 ---
        diff = np.mean(alpha_stim_reg_rel, axis=0) - np.mean(alpha_stim_ran_rel, axis=0)
        vmax = np.max(np.abs(diff))
        
        mne.viz.plot_topomap(diff, epochs_stim_regular.info, axes=axes[0,0], 
                             show=False, cmap='RdBu_r', vlim=(-vmax, vmax),
                             contours=6, colorbar=True)
        axes[0,0].set_title('【Main analysis 1】Stimulus Lock - Alpha\n'
                           '200-400ms, Regular - Random\n'
                           'Relative Power (freq-baseline corrected)',
                           fontsize=11, fontweight='bold')
        
        # 顯示統計資訊
        mean_diff = np.mean(diff)
        axes[0,0].text(0.02, 0.98, f'Mean Δ = {mean_diff:.3f}',
                      transform=axes[0,0].transAxes,
                      fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # --- (1,2) Stimulus-Theta - 對照分析 ---
        diff = np.mean(theta_stim_reg_rel, axis=0) - np.mean(theta_stim_ran_rel, axis=0)
        vmax = np.max(np.abs(diff))
        
        mne.viz.plot_topomap(diff, epochs_stim_regular.info, axes=axes[0,1],
                             show=False, cmap='RdBu_r', vlim=(-vmax, vmax),
                             contours=6, colorbar=True)
        axes[0,1].set_title('【Coresponding analysis】Stimulus Lock - Theta\n'
                           '200-400ms, Regular - Random\n'
                           'Relative Power (freq-baseline corrected)',
                           fontsize=11)
        
        mean_diff = np.mean(diff)
        axes[0,1].text(0.02, 0.98, f'Mean Δ = {mean_diff:.3f}',
                      transform=axes[0,1].transAxes,
                      fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # --- (2,1) Response-Theta - 主要分析 ---
        diff = np.mean(theta_resp_reg_rel, axis=0) - np.mean(theta_resp_ran_rel, axis=0)
        vmax = np.max(np.abs(diff))
        
        mne.viz.plot_topomap(diff, epochs_resp_regular.info, axes=axes[1,0],
                             show=False, cmap='RdBu_r', vlim=(-vmax, vmax),
                             contours=6, colorbar=True)
        axes[1,0].set_title('【Main analysis 2】Response Lock - Theta\n'
                           '-300~+50ms, Regular - Random\n'
                           'Relative Power (freq-baseline corrected)',
                           fontsize=11, fontweight='bold')
        
        mean_diff = np.mean(diff)
        axes[1,0].text(0.02, 0.98, f'Mean Δ = {mean_diff:.3f}',
                      transform=axes[1,0].transAxes,
                      fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # --- (2,2) Response-Alpha - 對照分析 ---
        diff = np.mean(alpha_resp_reg_rel, axis=0) - np.mean(alpha_resp_ran_rel, axis=0)
        vmax = np.max(np.abs(diff))
        
        mne.viz.plot_topomap(diff, epochs_resp_regular.info, axes=axes[1,1],
                             show=False, cmap='RdBu_r', vlim=(-vmax, vmax),
                             contours=6, colorbar=True)
        axes[1,1].set_title('【Coresponding analysis】Response Lock - Alpha\n'
                           '-300~+50ms, Regular - Random\n'
                           'Relative Power (freq-baseline corrected)',
                           fontsize=11)
        
        mean_diff = np.mean(diff)
        axes[1,1].text(0.02, 0.98, f'Mean Δ = {mean_diff:.3f}',
                      transform=axes[1,1].transAxes,
                      fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 整體標題
        fig.suptitle(f'ASRT Complete Analysis - Topographic Maps ({subject_id})\n'
                     'Frequency-domain baseline corrected relative power',
                     fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 儲存
        filename = f'{subject_id}_ASRT_complete_topomaps_freqbaseline.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ 地形圖已儲存: {filename}")
        
        plt.show(block=False)
        
        return fig


    # ============================================================
    # 方法 2: 如果想要使用絕對功率（備選）
    # ============================================================

    def plot_asrt_topomaps_absolute_power(epochs_stim_regular, epochs_stim_random,
                                         epochs_resp_regular, epochs_resp_random,
                                         subject_id='subject'):
        """
        繪製 ASRT 地形圖（使用絕對功率）
        
        這是備選方案，如果您想看絕對功率的地形圖
        """
        from mne_python_analysis.spectral_analysis import compute_band_power_multiple_bands
        
        print("\n步驟 6: 繪製地形圖（絕對功率版本）")
        
        # 計算絕對功率
        freq_bands = {'theta': (4, 8), 'alpha': (8, 13)}
        
        # Stimulus Lock
        stim_reg_power = compute_band_power_multiple_bands(
            epochs_stim_regular, freq_bands, tmin=0.1, tmax=0.3)
        stim_ran_power = compute_band_power_multiple_bands(
            epochs_stim_random, freq_bands, tmin=0.1, tmax=0.3)
        
        # Response Lock
        resp_reg_power = compute_band_power_multiple_bands(
            epochs_resp_regular, freq_bands, tmin=-0.3, tmax=0.05)
        resp_ran_power = compute_band_power_multiple_bands(
            epochs_resp_random, freq_bands, tmin=-0.3, tmax=0.05)
        
        # 繪製（與原始版本相同）
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # (1,1) Stimulus-Alpha
        diff = np.mean(stim_reg_power['alpha'], axis=0) - np.mean(stim_ran_power['alpha'], axis=0)
        vmax = np.max(np.abs(diff))
        mne.viz.plot_topomap(diff, epochs_stim_regular.info, axes=axes[0,0], 
                             show=False, cmap='RdBu_r', vlim=(-vmax, vmax))
        axes[0,0].set_title('【Main analysis 1】Stimulus - Alpha\n(100-300ms, Regular-Random)\nAbsolute Power')
        
        # (1,2) Stimulus-Theta
        diff = np.mean(stim_reg_power['theta'], axis=0) - np.mean(stim_ran_power['theta'], axis=0)
        vmax = np.max(np.abs(diff))
        mne.viz.plot_topomap(diff, epochs_stim_regular.info, axes=axes[0,1],
                             show=False, cmap='RdBu_r', vlim=(-vmax, vmax))
        axes[0,1].set_title('【Coresponding analysis 1】Stimulus - Theta\n(100-300ms, Regular-Random)\nAbsolute Power')
        
        # (2,1) Response-Theta
        diff = np.mean(resp_reg_power['theta'], axis=0) - np.mean(resp_ran_power['theta'], axis=0)
        vmax = np.max(np.abs(diff))
        mne.viz.plot_topomap(diff, epochs_resp_regular.info, axes=axes[1,0],
                             show=False, cmap='RdBu_r', vlim=(-vmax, vmax))
        axes[1,0].set_title('【Main analysis 2】Response - Theta\n(-300~+50ms, Regular-Random)\nAbsolute Power')
        
        # (2,2) Response-Alpha
        diff = np.mean(resp_reg_power['alpha'], axis=0) - np.mean(resp_ran_power['alpha'], axis=0)
        vmax = np.max(np.abs(diff))
        mne.viz.plot_topomap(diff, epochs_resp_regular.info, axes=axes[1,1],
                             show=False, cmap='RdBu_r', vlim=(-vmax, vmax))
        axes[1,1].set_title('【Coresponding analysis 2】Response - Alpha\n(-300~+50ms, Regular-Random)\nAbsolute Power')
        
        plt.tight_layout()
        filename = f'{subject_id}_ASRT_topomaps_absolute.png'
        plt.savefig(filename, dpi=300)
        print(f"✓ 地形圖已儲存: {filename}")
        plt.show(block=False)
        
        return fig


    
    # 步驟 7: 儲存結果
    # 先整理要輸出的結果（全部都來自前面已經計算好的變數）
    results = {
        # Stimulus Lock - Alpha ROI（主分析 1）
        "stim_alpha_regular": alpha_roi_stim_reg_rel,
        "stim_alpha_random": alpha_roi_stim_ran_rel,
        "sig_channels_stim_alpha": sig_channels_alpha,  # 來自全腦 Alpha 統計

        # Response Lock - Theta ROI（主分析 2）
        "resp_theta_regular": theta_roi_resp_reg_rel,
        "resp_theta_random": theta_roi_resp_ran_rel,
        "sig_channels_resp_theta": sig_channels_theta,  # 來自全腦 Theta 統計

        # 對照分析（ROI）
        # Stimulus Lock - Theta ROI
        "stim_theta_regular": theta_roi_stim_reg_rel,
        "stim_theta_random": theta_roi_stim_ran_rel,

        # Response Lock - Alpha ROI
        "resp_alpha_regular": alpha_roi_resp_reg_rel,
        "resp_alpha_random": alpha_roi_resp_ran_rel,

        # （選擇性）也可以把全腦的 relative power 一起存起來
        "alpha_whole_stim_reg_rel": alpha_whole_stim_reg_rel,
        "alpha_whole_stim_ran_rel": alpha_whole_stim_ran_rel,
        "theta_whole_resp_reg_rel": theta_whole_resp_reg_rel,
        "theta_whole_resp_ran_rel": theta_whole_resp_ran_rel,
    }

    save_choice = input("\n是否儲存結果檔案? (y/n): ").strip().lower()
    if save_choice == "y":
        np.savez(
            f"{subject_id}_ASRT_complete_results.npz",
            **results
        )
        print(f"✓ 結果已儲存: {subject_id}_ASRT_complete_results.npz")

    return results


# ============================================================
# ASRT ROI 頻譜分析
# 來源：main.py 第 1060-1439 行（380 行）
# ============================================================

def asrt_roi_spectral_analysis(epochs, subject_id):
    """ASRT ROI 頻譜分析（針對已有的 epochs，支援 baseline correction）"""
    print("\n" + "="*60)
    print("ASRT ROI 頻譜分析")
    print("="*60)
    
    if not ASRT_MODULES_AVAILABLE:
        print("⚠️  ASRT 模組未安裝，無法執行此功能")
        return None
    
    if epochs is None:
        print("⚠️  請先建立 Epochs")
        return None
    
    # 檢查 epochs 類型
    print(f"\n✓ Epochs 資訊:")
    print(f"  - 總 epochs: {len(epochs)}")
    print(f"  - 時間範圍: {epochs.tmin:.2f} ~ {epochs.tmax:.2f} s")
    
    # 判斷是 Stimulus-locked 還是 Response-locked
    if epochs.tmin >= -0.6 and epochs.tmax <= 0.7:
        epoch_type = "Stimulus"
        print(f"  - 類型: Stimulus-locked")
    elif epochs.tmin <= -0.7 and epochs.tmax >= 0.2:
        epoch_type = "Response"
        print(f"  - 類型: Response-locked")
    else:
        print(f"  - 類型: 未知")
        epoch_type = input("\n請手動選擇類型 (stimulus/response): ").strip().lower()
        if epoch_type not in ['stimulus', 'response']:
            epoch_type = 'response'
    
    # 檢查 metadata
    has_phase = hasattr(epochs, 'metadata') and 'phase' in epochs.metadata.columns
    has_test_type = hasattr(epochs, 'metadata') and 'test_type' in epochs.metadata.columns
    
    # 顯示 metadata 中實際存在的值 (診斷用)
    if has_phase:
        unique_phases = epochs.metadata['phase'].unique()
        print(f"\n  - Metadata 中的 phase 值: {unique_phases}")
        print(f"  - 各 phase 的 epochs 數:")
        for phase_val in unique_phases:
            count = len(epochs[epochs.metadata['phase'] == phase_val])
            print(f"    • {phase_val}: {count} epochs")
    
    # 1. 選擇實驗階段
    print("\n" + "="*60)
    print("選擇實驗階段")
    print("="*60)
    print("1. Learning 階段")
    print("2. Testing 階段")
    print("3. 全部")
    
    phase_choice = input("\n請選擇 (1/2/3) [預設 3]: ").strip() or '3'
    
    # 根據選擇篩選 epochs
    if phase_choice == '1':
        if has_phase:
            # 支援多種可能的命名
            learning_values = ['learning', 'Learning', 'LEARNING', 'learn', 'Learn']
            epochs_filtered = None
            for val in learning_values:
                if val in epochs.metadata['phase'].values:
                    epochs_filtered = epochs[epochs.metadata['phase'] == val]
                    phase_name = "Learning"
                    break
            
            if epochs_filtered is None:
                print(f"⚠️  找不到 Learning 階段資料")
                print(f"   Metadata 中的 phase 值: {epochs.metadata['phase'].unique()}")
                print("   使用全部 epochs")
                epochs_filtered = epochs.copy()
                phase_name = "All"
        else:
            print("⚠️  Metadata 中沒有 phase 欄位,使用全部 epochs")
            epochs_filtered = epochs.copy()
            phase_name = "All"
            
    elif phase_choice == '2':
        if has_phase:
            # 支援多種可能的命名
            test_values = ['test', 'Test', 'TEST', 'testing', 'Testing', 'TESTING']
            epochs_filtered = None
            for val in test_values:
                if val in epochs.metadata['phase'].values:
                    epochs_filtered = epochs[epochs.metadata['phase'] == val]
                    phase_name = "Testing"
                    break
            
            if epochs_filtered is None:
                print(f"⚠️  找不到 Testing 階段資料")
                print(f"   Metadata 中的 phase 值: {epochs.metadata['phase'].unique()}")
                print("   使用全部 epochs")
                epochs_filtered = epochs.copy()
                phase_name = "All"
        else:
            print("⚠️  Metadata 中沒有 phase 欄位,使用全部 epochs")
            epochs_filtered = epochs.copy()
            phase_name = "All"
    else:
        epochs_filtered = epochs.copy()
        phase_name = "All"
    
    print(f"\n✓ 選擇階段: {phase_name} ({len(epochs_filtered)} epochs)")
    
    # 檢查篩選後的 epochs 數量
    if len(epochs_filtered) == 0:
        print("\n" + "="*60)
        print("⚠️  錯誤: 篩選後沒有任何 epochs")
        print("="*60)
        print("\n可能原因:")
        print("1. Metadata 中的 phase 值命名不符合預期")
        print("2. 選擇的階段在資料中不存在")
        print("\n建議:")
        print("• 檢查 metadata 中的實際值")
        print("• 使用選項 3 (顯示 Epochs 資訊) 查看 metadata")
        print("• 選擇 '3. 全部' 來使用所有 epochs")
        return None
    
    # 2. 如果選擇 Testing 階段,進一步選擇 motor/perceptual
    test_type_filter = None
    if phase_choice == '2' and has_test_type:
        print("\n選擇 Testing 類型:")
        print("1. Motor")
        print("2. Perceptual")
        print("3. 全部")
        
        test_choice = input("\n請選擇 (1/2/3) [預設 3]: ").strip() or '3'
        
        if test_choice == '1':
            # 支援多種可能的命名
            motor_values = ['motor', 'Motor', 'MOTOR']
            epochs_temp = None
            for val in motor_values:
                if val in epochs_filtered.metadata['test_type'].values:
                    epochs_temp = epochs_filtered[epochs_filtered.metadata['test_type'] == val]
                    test_type_filter = 'motor'
                    phase_name += " - Motor"
                    break
            
            if epochs_temp is not None:
                epochs_filtered = epochs_temp
            else:
                print(f"⚠️  找不到 Motor 條件")
                print(f"   Metadata 中的 test_type 值: {epochs_filtered.metadata['test_type'].unique()}")
                
        elif test_choice == '2':
            # 支援多種可能的命名
            perceptual_values = ['perceptual', 'Perceptual', 'PERCEPTUAL']
            epochs_temp = None
            for val in perceptual_values:
                if val in epochs_filtered.metadata['test_type'].values:
                    epochs_temp = epochs_filtered[epochs_filtered.metadata['test_type'] == val]
                    test_type_filter = 'perceptual'
                    phase_name += " - Perceptual"
                    break
            
            if epochs_temp is not None:
                epochs_filtered = epochs_temp
            else:
                print(f"⚠️  找不到 Perceptual 條件")
                print(f"   Metadata 中的 test_type 值: {epochs_filtered.metadata['test_type'].unique()}")
        
        print(f"✓ 篩選後: {phase_name} ({len(epochs_filtered)} epochs)")
        
        # 再次檢查篩選後的 epochs 數量
        if len(epochs_filtered) == 0:
            print("\n" + "="*60)
            print("⚠️  錯誤: 篩選後沒有任何 epochs")
            print("="*60)
            print("\n可能原因:")
            print("1. Metadata 中的 test_type 值命名不符合預期")
            print("2. 選擇的條件在資料中不存在")
            print("\n建議:")
            print("• 檢查 metadata 中的實際值")
            print("• 選擇 '3. 全部' 來使用所有 Testing epochs")
            return None
    
    # 3. 選擇 ROI
    print("\n" + "="*60)
    print("選擇 ROI")
    print("="*60)
    
    if epoch_type == "Stimulus":
        print("推薦: Alpha (Stimulus-locked 分析)")
        print("1. Theta (Fz, FCz, Cz, C3, C4) - 4-8 Hz")
        print("2. Alpha (O1, O2, Oz, P3, P4, Pz) - 8-13 Hz")
        print("3. 兩者都分析")
        default_roi = '2'
    else:
        print("推薦: Theta (Response-locked 分析)")
        print("1. Theta (Fz, FCz, Cz, C3, C4) - 4-8 Hz")
        print("2. Alpha (O1, O2, Oz, P3, P4, Pz) - 8-13 Hz")
        print("3. 兩者都分析")
        default_roi = '1'
    
    roi_choice = input(f"\n請選擇 (1/2/3) [預設 {default_roi}]: ").strip()
    
    if not roi_choice:
        roi_choice = default_roi
    
    if roi_choice == '1':
        roi_list = ['theta']
    elif roi_choice == '2':
        roi_list = ['alpha']
    else:
        roi_list = ['theta', 'alpha']
    
    # 2. 選擇分析方法
    print("\n選擇分析方法:")
    print("1. FFT 功率分析 (帶 frequency-domain baseline correction)")
    print("2. 時頻分析 (Morlet wavelet)")
    
    method_choice = input("\n請選擇 (1/2): ").strip()
    
    # 3. 設定時間窗口和 baseline
    if epoch_type == "Stimulus":
        # Stimulus-locked: 分析 200-400ms
        task_tmin, task_tmax = 0.1, 0.3
        baseline_tmin, baseline_tmax = -0.5, -0.1
    else:
        # Response-locked: 分析 -300~+50ms
        task_tmin, task_tmax = -0.3, 0.05
        baseline_tmin, baseline_tmax = -0.75, -0.25
    
    print(f"\n  - 分析窗口: {task_tmin*1000:.0f}-{task_tmax*1000:.0f} ms")
    print(f"  - Baseline: {baseline_tmin*1000:.0f}-{baseline_tmax*1000:.0f} ms")
    
    # 4. 分析各 ROI
    results = {}
    
    for roi_name in roi_list:
        print(f"\n{'='*60}")
        print(f"分析 {roi_name.upper()} ROI")
        print(f"{'='*60}")
        
        # 定義 ROI
        roi_channels = define_roi_channels(roi_name)
        fmin = 4 if roi_name == 'theta' else 8
        fmax = 8 if roi_name == 'theta' else 13
        
        print(f"  - 頻率範圍: {fmin}-{fmax} Hz")
        print(f"  - 電極: {roi_channels}")
        
        if method_choice == '1':
            # === FFT 分析（帶 baseline correction）===
            
            # 分離 Regular 和 Random
            if hasattr(epochs_filtered, 'metadata') and 'trial_type' in epochs_filtered.metadata.columns:
                epochs_regular = epochs_filtered[epochs_filtered.metadata['trial_type'] == 'Regular']
                epochs_random = epochs_filtered[epochs_filtered.metadata['trial_type'] == 'Random']
                
                print(f"\n計算 Regular trials ({len(epochs_regular)} epochs)...")
                power_rel_reg, power_task_reg, power_base_reg = compute_roi_power_with_freq_baseline(
                    epochs_regular, roi_channels,
                    fmin=fmin, fmax=fmax,
                    task_tmin=task_tmin, task_tmax=task_tmax,
                    baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
                    method='relative'
                )
                
                print(f"計算 Random trials ({len(epochs_random)} epochs)...")
                power_rel_ran, power_task_ran, power_base_ran = compute_roi_power_with_freq_baseline(
                    epochs_random, roi_channels,
                    fmin=fmin, fmax=fmax,
                    task_tmin=task_tmin, task_tmax=task_tmax,
                    baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
                    method='relative'
                )
                
                # 統計比較
                from scipy import stats
                t_stat, p_val = stats.ttest_ind(power_rel_reg, power_rel_ran)
                
                print(f"\n{roi_name.upper()} 功率摘要:")
                print(f"  Regular:")
                print(f"    - 相對功率: M = {np.mean(power_rel_reg):.4f}, SD = {np.std(power_rel_reg):.4f}")
                print(f"    - Task 功率:  M = {np.mean(power_task_reg):.4e}, SD = {np.std(power_task_reg):.4e}")
                print(f"    - Base 功率:  M = {np.mean(power_base_reg):.4e}, SD = {np.std(power_base_reg):.4e}")
                
                print(f"  Random:")
                print(f"    - 相對功率: M = {np.mean(power_rel_ran):.4f}, SD = {np.std(power_rel_ran):.4f}")
                print(f"    - Task 功率:  M = {np.mean(power_task_ran):.4e}, SD = {np.std(power_task_ran):.4e}")
                print(f"    - Base 功率:  M = {np.mean(power_base_ran):.4e}, SD = {np.std(power_base_ran):.4e}")
                
                print(f"\n  統計比較 (t-test):")
                print(f"    t({len(power_rel_reg)+len(power_rel_ran)-2}) = {t_stat:.3f}, p = {p_val:.4f}")
                
                if p_val < 0.001:
                    print(f"    *** 顯著差異 p < 0.001")
                elif p_val < 0.01:
                    print(f"    ** 顯著差異 p < 0.01")
                elif p_val < 0.05:
                    print(f"    * 顯著差異 p < 0.05")
                else:
                    print(f"    n.s. (無顯著差異)")
                
                # 視覺化
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # 左圖：相對功率 bar plot
                means = [np.mean(power_rel_reg), np.mean(power_rel_ran)]
                sems = [stats.sem(power_rel_reg), stats.sem(power_rel_ran)]
                x_pos = [0, 1]
                colors = ['blue', 'red']
                
                ax1.bar(x_pos, means, yerr=sems, color=colors, alpha=0.7, capsize=5)
                ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(['Regular', 'Random'])
                ax1.set_ylabel(f'{roi_name.capitalize()} Power (Relative to Baseline)')
                ax1.set_title(f'{phase_name} - {roi_name.capitalize()} Power: Regular vs Random')
                ax1.grid(True, alpha=0.3, axis='y')
                
                # 右圖：分布 violin plot
                data_to_plot = [power_rel_reg, power_rel_ran]
                parts = ax2.violinplot(data_to_plot, positions=[0, 1], showmeans=True, showmedians=True)
                ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax2.set_xticks([0, 1])
                ax2.set_xticklabels(['Regular', 'Random'])
                ax2.set_ylabel(f'{roi_name.capitalize()} Power (Relative to Baseline)')
                ax2.set_title('Distribution')
                ax2.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                # 將階段名稱加入檔案名稱
                phase_suffix = phase_name.replace(' ', '_').replace('-', '_')
                filename = f'{subject_id}_{phase_suffix}_{roi_name}_power_comparison.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"\n✓ 圖表已儲存: {filename}")
                plt.show(block=False)
                
                results[roi_name] = {
                    'regular': {'rel': power_rel_reg, 'task': power_task_reg, 'base': power_base_reg},
                    'random': {'rel': power_rel_ran, 'task': power_task_ran, 'base': power_base_ran},
                    't_stat': t_stat,
                    'p_val': p_val,
                    'phase': phase_name
                }
            else:
                # 沒有 metadata，計算全部 epochs
                print(f"\n計算所有 epochs ({len(epochs_filtered)} epochs)...")
                power_rel, power_task, power_base = compute_roi_power_with_freq_baseline(
                    epochs_filtered, roi_channels,
                    fmin=fmin, fmax=fmax,
                    task_tmin=task_tmin, task_tmax=task_tmax,
                    baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
                    method='relative'
                )
                
                print(f"\n{roi_name.upper()} 功率:")
                print(f"  - 相對功率: M = {np.mean(power_rel):.4f}, SD = {np.std(power_rel):.4f}")
                print(f"  - Task 功率:  M = {np.mean(power_task):.4e}, SD = {np.std(power_task):.4e}")
                print(f"  - Base 功率:  M = {np.mean(power_base):.4e}, SD = {np.std(power_base):.4e}")
                
                results[roi_name] = {
                    'all': {'rel': power_rel, 'task': power_task, 'base': power_base}
                }
        
        elif method_choice == '2':
            # === 時頻分析 ===
            # 創建虛擬電極
            virtual_epochs = create_virtual_channel_epochs(
                epochs_filtered, roi_channels, f'{roi_name.upper()}_ROI'
            )
            
            freqs = np.arange(4, 40, 1)
            tfr = compute_tfr_morlet(virtual_epochs, freqs=freqs, average=True)
            results[f'{roi_name}_tfr'] = tfr
            
            # 繪圖
            tfr.plot(picks=[0], title=f'{roi_name.upper()} ROI Time-Frequency')
            plt.show(block=False)
    
    print("\n" + "="*60)
    print("✓ 分析完成")
    print("="*60)
    
    return results


# ============================================================
# ASRT Block 比較分析
# 來源：main.py 第 1440-1756 行（317 行）
# ============================================================

def asrt_block_comparison(epochs, subject_id):
    """ASRT Block 比較分析"""
    print("\n" + "="*60)
    print("ASRT Block 比較分析")
    print("="*60)
    
    if not ASRT_MODULES_AVAILABLE:
        print("⚠️  ASRT 模組未安裝，無法執行此功能")
        return None
    
    if epochs is None:
        print("⚠️  請先建立 Epochs")
        return None
    
    # 檢查metadata
    if not hasattr(epochs, 'metadata') or epochs.metadata is None:
        print("⚠️  Epochs 沒有 metadata，無法進行分析")
        return None
    
    if 'block' not in epochs.metadata.columns:
        print("⚠️  Metadata 中缺少 'block' 欄位")
        return None
    
    # 顯示 epochs 資訊
    print(f"\n✓ Epochs 資訊:")
    print(f"  - 總 epochs: {len(epochs)}")
    print(f"  - Block 範圍: {epochs.metadata['block'].min()}-{epochs.metadata['block'].max()}")
    print(f"  - 時間範圍: {epochs.tmin:.2f} ~ {epochs.tmax:.2f} s")
    
    # 判斷是 Stimulus-locked 還是 Response-locked
    if epochs.tmin >= -0.6 and epochs.tmax <= 0.7:
        epoch_type = "Stimulus"
        print(f"  - 類型: Stimulus-locked")
    elif epochs.tmin <= -0.7 and epochs.tmax >= 0.2:
        epoch_type = "Response"
        print(f"  - 類型: Response-locked")
    else:
        print(f"  - 類型: 未知")
        epoch_type = input("\n請手動選擇類型 (stimulus/response): ").strip().lower()
        if epoch_type not in ['stimulus', 'response']:
            epoch_type = 'response'
    
    # === 1. 選擇分析階段 ===
    print("\n" + "="*60)
    print("選擇分析階段")
    print("="*60)
    print("1. Learning 階段 (Block 7-26)")
    print("2. Testing 階段 (Block 27-34)")
    print("3. 全部 (Block 7-34)")
    
    phase_choice = input("\n請選擇 (1/2/3) [預設 1]: ").strip()
    
    if phase_choice == '2':
        phase_filter = 'test'
        blocks_to_analyze = [27, 28, 29, 30, 31, 32, 33, 34]
    elif phase_choice == '3':
        phase_filter = 'all'
        blocks_to_analyze = list(range(7, 35))
    else:
        phase_filter = 'learning'
        blocks_to_analyze = list(range(7, 27))
    
    # === 2. 如果是 Testing 階段，選擇 motor/perceptual ===
    test_type_filter = None
    if phase_filter in ['test', 'all'] and 'test_type' in epochs.metadata.columns:
        print("\n" + "="*60)
        print("選擇 Testing 類型")
        print("="*60)
        print("1. Motor learning blocks")
        print("2. Perceptual learning blocks")
        print("3. 全部")
        
        test_choice = input("\n請選擇 (1/2/3) [預設 3]: ").strip()
        
        if test_choice == '1':
            test_type_filter = 'motor'
        elif test_choice == '2':
            test_type_filter = 'perceptual'
        else:
            test_type_filter = None
    
    # === 3. 定義 ROI ===
    def define_roi_channels(roi_name):
        """定義 ROI 電極"""
        roi_definitions = {
            'theta': ['Fz', 'FCz', 'Cz', 'C3', 'C4'],
            'alpha': ['O1', 'Oz', 'O2', 'P3', 'Pz', 'P4']
        }
        return roi_definitions.get(roi_name.lower(), [])
    
    # === 4. 設定時間窗口 ===
    if epoch_type == "Stimulus":
        task_tmin, task_tmax = 0.1, 0.3
        baseline_tmin, baseline_tmax = -0.5, -0.1
        print(f"\n  - 分析窗口: {task_tmin*1000:.0f}-{task_tmax*1000:.0f} ms")
        print(f"  - Baseline: {baseline_tmin*1000:.0f}-{baseline_tmax*1000:.0f} ms")
    else:
        task_tmin, task_tmax = -0.3, 0.05
        baseline_tmin, baseline_tmax = -0.75, -0.25
        print(f"\n  - 分析窗口: {task_tmin*1000:.0f}-{task_tmax*1000:.0f} ms")
        print(f"  - Baseline: {baseline_tmin*1000:.0f}-{baseline_tmax*1000:.0f} ms")
    
    # === 5. 篩選 epochs ===
    print("\n" + "="*60)
    print("篩選資料...")
    print("="*60)
    
    mask = epochs.metadata['block'].isin(blocks_to_analyze)
    
    if test_type_filter:
        mask &= (epochs.metadata.get('test_type', pd.Series([None]*len(epochs.metadata))) == test_type_filter)
    
    epochs_filtered = epochs[mask]
    print(f"✓ 篩選後 epochs: {len(epochs_filtered)}")
    
    # === 6. 分離 Regular 和 Random ===
    epochs_regular = epochs_filtered[epochs_filtered.metadata['trial_type'] == 'Regular']
    epochs_random = epochs_filtered[epochs_filtered.metadata['trial_type'] == 'Random']
    
    print(f"  - Regular: {len(epochs_regular)}")
    print(f"  - Random: {len(epochs_random)}")
    
    # === 7. 計算 Theta 和 Alpha 功率 ===
    print(f"\n計算各 block 的 Theta 和 Alpha 功率...")
    
    # Theta ROI
    theta_roi = define_roi_channels('theta')
    theta_fmin, theta_fmax = 4, 8
    
    # Alpha ROI
    alpha_roi = define_roi_channels('alpha')
    alpha_fmin, alpha_fmax = 8, 13
    
    # 儲存結果
    results = {
        'theta': {'regular': {}, 'random': {}},
        'alpha': {'regular': {}, 'random': {}}
    }
    
    for block_num in blocks_to_analyze:
        # === Theta ===
        # Regular
        block_ep_reg = epochs_regular[epochs_regular.metadata['block'] == block_num]
        if len(block_ep_reg) > 0:
            power_rel, _, _ = compute_roi_power_with_freq_baseline(
                block_ep_reg, theta_roi,
                fmin=theta_fmin, fmax=theta_fmax,
                task_tmin=task_tmin, task_tmax=task_tmax,
                baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
                method='relative'
            )
            results['theta']['regular'][block_num] = power_rel
        
        # Random
        block_ep_ran = epochs_random[epochs_random.metadata['block'] == block_num]
        if len(block_ep_ran) > 0:
            power_rel, _, _ = compute_roi_power_with_freq_baseline(
                block_ep_ran, theta_roi,
                fmin=theta_fmin, fmax=theta_fmax,
                task_tmin=task_tmin, task_tmax=task_tmax,
                baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
                method='relative'
            )
            results['theta']['random'][block_num] = power_rel
        
        # === Alpha ===
        # Regular
        if len(block_ep_reg) > 0:
            power_rel, _, _ = compute_roi_power_with_freq_baseline(
                block_ep_reg, alpha_roi,
                fmin=alpha_fmin, fmax=alpha_fmax,
                task_tmin=task_tmin, task_tmax=task_tmax,
                baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
                method='relative'
            )
            results['alpha']['regular'][block_num] = power_rel
        
        # Random
        if len(block_ep_ran) > 0:
            power_rel, _, _ = compute_roi_power_with_freq_baseline(
                block_ep_ran, alpha_roi,
                fmin=alpha_fmin, fmax=alpha_fmax,
                task_tmin=task_tmin, task_tmax=task_tmax,
                baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
                method='relative'
            )
            results['alpha']['random'][block_num] = power_rel
    
    # === 8. 計算平均值 ===
    block_numbers = sorted(set(results['theta']['regular'].keys()) & 
                           set(results['theta']['random'].keys()))
    
    theta_means_regular = []
    theta_means_random = []
    alpha_means_regular = []
    alpha_means_random = []
    
    for block_num in block_numbers:
        theta_means_regular.append(np.mean(results['theta']['regular'][block_num]))
        theta_means_random.append(np.mean(results['theta']['random'][block_num]))
        alpha_means_regular.append(np.mean(results['alpha']['regular'][block_num]))
        alpha_means_random.append(np.mean(results['alpha']['random'][block_num]))
    
    # === 9. 計算統一的 Y 軸範圍 ===
    all_values = (theta_means_regular + theta_means_random + 
                  alpha_means_regular + alpha_means_random)
    
    y_min = min(all_values)
    y_max = max(all_values)
    y_range = y_max - y_min
    y_padding = y_range * 0.1  # 10% padding
    
    ylim_min = y_min - y_padding
    ylim_max = y_max + y_padding
    
    print(f"\n統一 Y 軸範圍: {ylim_min:.3f} ~ {ylim_max:.3f}")
    
    # === 10. 視覺化 ===
    print("\n生成圖表...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # === 上圖：Theta ===
    ax1.plot(block_numbers, theta_means_regular, 'o-', linewidth=2, markersize=8,
             label='Regular', color='blue')
    ax1.plot(block_numbers, theta_means_random, 's-', linewidth=2, markersize=8,
             label='Random', color='red')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Block Number', fontsize=12)
    ax1.set_ylabel('Theta Power (Relative to Baseline)', fontsize=12)
    ax1.set_title(f'{subject_id} - Theta Power Across Blocks (Fz, FCz, Cz, C3, C4)', 
                  fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(ylim_min, ylim_max)  # 統一 Y 軸
    ax1.set_xticks(block_numbers)
    ax1.set_xticklabels([str(int(b)) for b in block_numbers])
    
    # === 下圖：Alpha ===
    ax2.plot(block_numbers, alpha_means_regular, 'o-', linewidth=2, markersize=8,
             label='Regular', color='blue')
    ax2.plot(block_numbers, alpha_means_random, 's-', linewidth=2, markersize=8,
             label='Random', color='red')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Block Number', fontsize=12)
    ax2.set_ylabel('Alpha Power (Relative to Baseline)', fontsize=12)
    ax2.set_title(f'{subject_id} - Alpha Power Across Blocks (O1, Oz, O2, P3, Pz, P4)', 
                  fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(ylim_min, ylim_max)  # 統一 Y 軸
    ax2.set_xticks(block_numbers)
    ax2.set_xticklabels([str(int(b)) for b in block_numbers])
    
    plt.tight_layout()
    
    # === 11. 儲存圖表 ===
    filename = f'{subject_id}_block_comparison_theta_alpha_{phase_filter}'
    if test_type_filter:
        filename += f'_{test_type_filter}'
    filename += '.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ 圖表已儲存: {filename}")
    plt.show(block=False)
    
    # === 12. 統計比較 ===
    print("\n" + "="*60)
    print("統計摘要")
    print("="*60)
    
    # Theta
    all_theta_regular = np.concatenate([results['theta']['regular'][b] for b in block_numbers])
    all_theta_random = np.concatenate([results['theta']['random'][b] for b in block_numbers])
    t_theta, p_theta = stats.ttest_ind(all_theta_regular, all_theta_random)
    
    print(f"\nTheta (4-8 Hz):")
    print(f"  Regular: M={np.mean(all_theta_regular):.4f}, SD={np.std(all_theta_regular):.4f}")
    print(f"  Random:  M={np.mean(all_theta_random):.4f}, SD={np.std(all_theta_random):.4f}")
    print(f"  t({len(all_theta_regular)+len(all_theta_random)-2})={t_theta:.3f}, p={p_theta:.4f}")
    
    # Alpha
    all_alpha_regular = np.concatenate([results['alpha']['regular'][b] for b in block_numbers])
    all_alpha_random = np.concatenate([results['alpha']['random'][b] for b in block_numbers])
    t_alpha, p_alpha = stats.ttest_ind(all_alpha_regular, all_alpha_random)
    
    print(f"\nAlpha (8-13 Hz):")
    print(f"  Regular: M={np.mean(all_alpha_regular):.4f}, SD={np.std(all_alpha_regular):.4f}")
    print(f"  Random:  M={np.mean(all_alpha_random):.4f}, SD={np.std(all_alpha_random):.4f}")
    print(f"  t({len(all_alpha_regular)+len(all_alpha_random)-2})={t_alpha:.3f}, p={p_alpha:.4f}")
    
    # === 13. 返回結果 ===
    return_results = {
        'block_numbers': block_numbers,
        'theta': {
            'regular_means': theta_means_regular,
            'random_means': theta_means_random,
            'all_regular': all_theta_regular,
            'all_random': all_theta_random,
            't_statistic': t_theta,
            'p_value': p_theta
        },
        'alpha': {
            'regular_means': alpha_means_regular,
            'random_means': alpha_means_random,
            'all_regular': all_alpha_regular,
            'all_random': all_alpha_random,
            't_statistic': t_alpha,
            'p_value': p_alpha
        },
        'ylim': (ylim_min, ylim_max)
    }
    
    return return_results


# ============================================================
# ASRT 極端值排除 (整合 Visual Inspection)
# 來源：main.py 第 1757-1968 行（212 行）
# 修改：新增 Visual Inspection 功能
# ============================================================

def asrt_artifact_rejection(epochs, method='flexible', threshold=None, autoreject_params=None):
    """
    ASRT 極端值排除
    
    三種方法：
    1. 'flexible': 彈性閾值（基於標準差）+ Visual Inspection
    2. 'autoreject': 使用 autoreject 套件自動排除
    3. 'fixed': 固定閾值 + Visual Inspection
    
    參數:
        epochs: mne.Epochs 物件
        method: 排除方法 ('flexible', 'autoreject', 'fixed')
        threshold: 固定閾值 (μV)，僅在 method='fixed' 時使用
        autoreject_params: autoreject 參數字典
    
    回傳:
        epochs_clean: 清理後的 Epochs
        rejection_log: 排除記錄字典
    """
    import matplotlib.pyplot as plt
    
    print("\n" + "="*60)
    print("ASRT 極端值排除")
    print("="*60)
    
    n_epochs = len(epochs)
    
    if method == 'flexible':
        # 彈性閾值：基於標準差
        print("\n使用彈性閾值（基於標準差）+ Visual Inspection")
        
        # 計算每個 trial 的峰峰值
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        peak_to_peak = np.max(data, axis=2) - np.min(data, axis=2)  # (n_epochs, n_channels)
        max_pp = np.max(peak_to_peak, axis=1)  # 每個 trial 的最大峰峰值
        
        # 計算統計量
        mean_pp = np.mean(max_pp)
        std_pp = np.std(max_pp)
        
        print(f"\n峰峰值統計:")
        print(f"  平均: {mean_pp*1e6:.2f} μV")
        print(f"  標準差: {std_pp*1e6:.2f} μV")
        
        # 視覺化分佈
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.hist(max_pp * 1e6, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(mean_pp * 1e6, color='blue', linestyle='--', linewidth=2, label=f'Mean')
        plt.axvline((mean_pp + 3*std_pp) * 1e6, color='orange', linestyle='--', linewidth=2, label=f'Mean+3σ')
        plt.axvline((mean_pp + 4*std_pp) * 1e6, color='red', linestyle='--', linewidth=2, label=f'Mean+4σ')
        plt.axvline((mean_pp + 5*std_pp) * 1e6, color='purple', linestyle='--', linewidth=2, label=f'Mean+5σ')
        plt.xlabel('Peak-to-Peak (μV)')
        plt.ylabel('Count')
        plt.title('Distribution of Peak-to-Peak Amplitudes')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.scatter(range(n_epochs), max_pp * 1e6, alpha=0.6, s=30)
        plt.axhline(mean_pp * 1e6, color='blue', linestyle='--', linewidth=2, label='Mean')
        plt.axhline((mean_pp + 3*std_pp) * 1e6, color='orange', linestyle='--', linewidth=2, label='Mean+3σ')
        plt.axhline((mean_pp + 4*std_pp) * 1e6, color='red', linestyle='--', linewidth=2, label='Mean+4σ')
        plt.axhline((mean_pp + 5*std_pp) * 1e6, color='purple', linestyle='--', linewidth=2, label='Mean+5σ')
        plt.xlabel('Trial Number')
        plt.ylabel('Peak-to-Peak (μV)')
        plt.title('Peak-to-Peak Across Trials')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show(block=False)
        
        # 詢問閾值
        print("\n建議使用 3-5 倍標準差")
        n_std_str = input("請輸入標準差倍數 (預設 3): ").strip()
        n_std = float(n_std_str) if n_std_str else 3.0
        
        threshold_value = mean_pp + n_std * std_pp
        print(f"\n閾值設定為: {threshold_value*1e6:.2f} μV")
        print(f"  (平均 + {n_std} × 標準差)")
        
        # 找出超過閾值的 trials
        bad_trials = max_pp > threshold_value
        candidate_idx = np.where(bad_trials)[0]
        n_bad = len(candidate_idx)
        
        print(f"\n找到 {n_bad} 個候選 bad trials")
        
        # ===== Visual Inspection =====
        if n_bad > 0:
            do_vi = input(f"是否進行Visual Inspection? (y/n) [推薦: y]: ").strip().lower()
            
            if do_vi == 'y':
                print("\n開始Visual Inspection...")
                print("指示: y=reject / n=keep / q=quit VI (剩餘自動reject)")
                
                confirmed_bad = []
                
                for i, trial_idx in enumerate(candidate_idx):
                    print(f"\n檢查 Trial {trial_idx} ({i+1}/{n_bad})")
                    print(f"Peak-to-peak: {max_pp[trial_idx]*1e6:.1f} μV")
                    
                    # 顯示波形
                    epochs[trial_idx].plot(scalings='auto', 
                                          title=f'Trial {trial_idx}')
                    
                    # 詢問是否reject
                    while True:
                        decision = input("確認reject此trial? (y/n/q): ").strip().lower()
                        if decision == 'y':
                            confirmed_bad.append(trial_idx)
                            print(f"  → Reject Trial {trial_idx}")
                            break
                        elif decision == 'n':
                            print(f"  → 保留 Trial {trial_idx}")
                            break
                        elif decision == 'q':
                            print("  → 提前結束VI，剩餘trials自動reject")
                            remaining = candidate_idx[i+1:]
                            confirmed_bad.extend(remaining.tolist())
                            break
                        else:
                            print("  請輸入 y/n/q")
                    
                    plt.close('all')
                    
                    if decision == 'q':
                        break
                
                bad_indices = confirmed_bad
                print(f"\n✓ VI完成: 候選{n_bad}個, 確認reject {len(confirmed_bad)}個")
            else:
                bad_indices = candidate_idx.tolist()
                print("跳過VI，使用自動判定結果")
        else:
            bad_indices = []
            print("✓ 所有trials都在閾值範圍內")
        
        # 執行排除
        if len(bad_indices) > 0:
            epochs_clean = epochs.copy().drop(bad_indices)
            print(f"\n排除結果:")
            print(f"  原始: {n_epochs} trials")
            print(f"  排除: {len(bad_indices)} trials ({100*len(bad_indices)/n_epochs:.1f}%)")
            print(f"  保留: {len(epochs_clean)} trials ({100*len(epochs_clean)/n_epochs:.1f}%)")
            
            rejection_log = {
                'method': 'flexible',
                'n_std': n_std,
                'threshold_uv': threshold_value * 1e6,
                'original_count': n_epochs,
                'rejected_count': len(bad_indices),
                'retained_count': len(epochs_clean),
                'retention_rate': 100 * len(epochs_clean) / n_epochs,
                'rejected_indices': bad_indices,
                'visual_inspection_used': do_vi == 'y' if n_bad > 0 else False
            }
            
            return epochs_clean, rejection_log
        else:
            print("✓ 無trials被排除")
            rejection_log = {
                'method': 'flexible',
                'n_std': n_std,
                'threshold_uv': threshold_value * 1e6,
                'original_count': n_epochs,
                'rejected_count': 0,
                'retained_count': n_epochs,
                'retention_rate': 100.0,
                'rejected_indices': [],
                'visual_inspection_used': False
            }
            return epochs.copy(), rejection_log
    
    elif method == 'fixed':
        # 固定閾值 + Visual Inspection
        print("\n使用固定閾值 + Visual Inspection")
        
        if threshold is None:
            threshold_value = 100e-6  # 預設100μV
            print(f"未提供閾值，使用預設: {threshold_value*1e6:.0f} μV")
        else:
            threshold_value = threshold
            print(f"閾值: {threshold_value*1e6:.0f} μV")
        
        # 計算峰峰值
        data = epochs.get_data()
        peak_to_peak = np.max(data, axis=2) - np.min(data, axis=2)
        max_pp = np.max(peak_to_peak, axis=1)
        
        # 找出超過閾值的trials
        bad_trials = max_pp > threshold_value
        candidate_idx = np.where(bad_trials)[0]
        n_bad = len(candidate_idx)
        
        print(f"\n找到 {n_bad} 個候選 bad trials")
        
        # Visual Inspection
        if n_bad > 0:
            do_vi = input(f"是否進行Visual Inspection? (y/n) [推薦: y]: ").strip().lower()
            
            if do_vi == 'y':
                confirmed_bad = []
                
                for i, trial_idx in enumerate(candidate_idx):
                    print(f"\n檢查 Trial {trial_idx} ({i+1}/{n_bad})")
                    print(f"Peak-to-peak: {max_pp[trial_idx]*1e6:.1f} μV")
                    
                    epochs[trial_idx].plot(scalings='auto', title=f'Trial {trial_idx}')
                    
                    while True:
                        decision = input("確認reject? (y/n/q): ").strip().lower()
                        if decision == 'y':
                            confirmed_bad.append(trial_idx)
                            break
                        elif decision == 'n':
                            break
                        elif decision == 'q':
                            remaining = candidate_idx[i+1:]
                            confirmed_bad.extend(remaining.tolist())
                            break
                        else:
                            print("請輸入 y/n/q")
                    
                    plt.close('all')
                    if decision == 'q':
                        break
                
                bad_indices = confirmed_bad
            else:
                bad_indices = candidate_idx.tolist()
        else:
            bad_indices = []
        
        # 執行排除
        if len(bad_indices) > 0:
            epochs_clean = epochs.copy().drop(bad_indices)
            rejection_log = {
                'method': 'fixed',
                'threshold_uv': threshold_value * 1e6,
                'original_count': n_epochs,
                'rejected_count': len(bad_indices),
                'retained_count': len(epochs_clean),
                'retention_rate': 100 * len(epochs_clean) / n_epochs,
                'rejected_indices': bad_indices,
                'visual_inspection_used': do_vi == 'y' if n_bad > 0 else False
            }
            return epochs_clean, rejection_log
        else:
            rejection_log = {
                'method': 'fixed',
                'threshold_uv': threshold_value * 1e6,
                'original_count': n_epochs,
                'rejected_count': 0,
                'retained_count': n_epochs,
                'retention_rate': 100.0,
                'rejected_indices': [],
                'visual_inspection_used': False
            }
            return epochs.copy(), rejection_log
    
    elif method == 'autoreject':
        # autoreject 方法保持原樣
        print("\n使用 autoreject 自動排除")
        try:
            from autoreject import AutoReject
        except ImportError:
            print("錯誤: autoreject 套件未安裝")
            print("請使用: pip install autoreject")
            return None, None
        
        if autoreject_params is None:
            autoreject_params = {}
        
        ar = AutoReject(**autoreject_params)
        epochs_clean = ar.fit_transform(epochs)
        
        n_rejected = len(epochs) - len(epochs_clean)
        print(f"\nautoreject 結果:")
        print(f"  原始: {len(epochs)} trials")
        print(f"  排除: {n_rejected} trials")
        print(f"  保留: {len(epochs_clean)} trials")
        
        rejection_log = {
            'method': 'autoreject',
            'original_count': len(epochs),
            'rejected_count': n_rejected,
            'retained_count': len(epochs_clean),
            'retention_rate': 100 * len(epochs_clean) / len(epochs),
            'visual_inspection_used': False
        }
        
        return epochs_clean, rejection_log
    
    else:
        raise ValueError(f"Unknown method: {method}")

# ============================================================
# Stimulus → Response ERSP 分析
# ============================================================
def asrt_stimulus_to_response_full_baseline(preprocessed_file, subject_id, output_dir,
                                           rt_file=None, blocks_range=None, 
                                           raw=None, events=None, response_times=None):
    """
    ERSP 分析：Stimulus → Response 對齊 + 整段平均 baseline
    
    這是一個整合的 workflow，結合：
    1. Time domain: 使用 Stimulus 時刻做固定參考點，根據 RT 重新對齊到 Response
    2. Frequency domain: 使用整段 Stimulus epoch 平均做 baseline（Lum 2023 方法）
    
    Parameters
    ----------
    preprocessed_file : str
        前處理後的 .fif 檔案路徑
        例如: 'output/sub0001_preprocessed-raw.fif'
    subject_id : str
        受試者 ID
        例如: 'sub0001'
    output_dir : str
        輸出資料夾路徑
        例如: 'output/ersp_analysis'
    rt_file : str, optional
        反應時間檔案路徑（.txt 或 .csv）
        如果為 None，會提示使用者輸入
    blocks_range : tuple, optional
        分析的 block 範圍
        例如: (7, 34) 表示只分析 block 7-34
        如果為 None，則分析所有 blocks
    raw : mne.io.Raw, optional
        如果已經載入 raw，可以直接傳入（避免重複載入）
    events : ndarray, optional
        如果已經提取 events，可以直接傳入
    response_times : ndarray, optional
        如果已經載入 RT（秒），可以直接傳入
        
    Returns
    -------
    power_resp : mne.time_frequency.AverageTFR
        Response-locked ERSP 結果
        
    Saves
    -----
    {subject_id}_response_ersp_full_baseline.png : 圖片檔
        ERSP topomap 圖
    {subject_id}_response_ersp_full_baseline-tfr.h5 : 資料檔
        AverageTFR 物件（可用 mne.time_frequency.read_tfrs 讀取）
        
    Examples
    --------
    >>> # 基本使用（會提示輸入 RT 檔案）
    >>> power_resp = asrt_stimulus_to_response_full_baseline(
    ...     'output/sub0001_preprocessed-raw.fif',
    ...     'sub0001',
    ...     'output/ersp_analysis'
    ... )
    
    >>> # 直接指定 RT 檔案
    >>> power_resp = asrt_stimulus_to_response_full_baseline(
    ...     'output/sub0001_preprocessed-raw.fif',
    ...     'sub0001',
    ...     'output/ersp_analysis',
    ...     rt_file='data/sub0001_rt.txt'
    ... )
    
    >>> # 只分析特定 blocks
    >>> power_resp = asrt_stimulus_to_response_full_baseline(
    ...     'output/sub0001_preprocessed-raw.fif',
    ...     'sub0001',
    ...     'output/ersp_analysis',
    ...     rt_file='data/sub0001_rt.txt',
    ...     blocks_range=(7, 26)  # 只分析 Learning phase
    ... )
    
    Notes
    -----
    此方法適合用於：
    - RT 變異較大的實驗
    - 需要固定 baseline 參考點的分析
    - 比較不同 RT 分布的 conditions
    
    優點：
    - 時間對齊一致（所有 trials 用相同 Stimulus 參考）
    - Baseline 標準化（不依賴特定時間窗口）
    - 避免 Response baseline 受 RT 影響
    
    Trigger Codes:
    - 自動處理 ASRT trigger codes (41-49 for stimulus, 21-29 for response)
    - 自動轉換為 event_id = {"Random": 1, "Regular": 2}
    """
    
    # 注意：這裡使用的 imports 已經在 workflows.py 開頭定義好了
    # create_stimulus_locked_epochs, create_response_locked_epochs 來自 mne_python_analysis.epochs
    # stimulus_to_response_with_full_epoch_baseline 來自 .ersp
    from .ersp import stimulus_to_response_with_full_epoch_baseline
    
    # ========== 開始分析 ==========
    print("\n" + "="*70)
    print("ASRT ERSP Analysis")
    print("Method: Stimulus → Response alignment + Full-epoch baseline")
    print("="*70)
    print(f"受試者: {subject_id}")
    print(f"輸入檔案: {preprocessed_file}")
    print(f"輸出資料夾: {output_dir}")
    if blocks_range:
        print(f"分析範圍: Block {blocks_range[0]} - {blocks_range[1]}")
    print("="*70 + "\n")
    
    # ========== 載入資料 ==========
    if raw is None:
        print("載入前處理後的資料...")
        raw = mne.io.read_raw_fif(preprocessed_file, preload=True)
        print(f"  ✓ 資料載入完成")
        print(f"    採樣率: {raw.info['sfreq']} Hz")
        print(f"    通道數: {len(raw.ch_names)}")
        print(f"    資料長度: {raw.times[-1]:.1f} 秒\n")
    else:
        print("使用已載入的 raw 物件\n")
    
    # ========== 提取 Events ==========
    if events is None:
        print("從 raw 提取 events...")
        try:
            events, event_id_map = mne.events_from_annotations(raw)
            print(f"  ✓ 從 annotations 找到 {len(events)} 個事件")
        except:
            events = mne.find_events(raw, stim_channel='Trigger', shortest_event=1)
            event_id_map = None
            print(f"  ✓ 從 Trigger 通道找到 {len(events)} 個事件")
        
        print(f"  事件碼種類: {np.unique(events[:, 2])}\n")
    else:
        print("使用已提供的 events\n")
    
    # ========== Event ID ==========
    # ASRT 使用固定的 event_id（在 create_*_epochs 中會自動處理原始 trigger codes）
    event_id = {"Random": 1, "Regular": 2}
    print("使用 ASRT event_id:")
    print(f"  Random: 1 (來自 stimulus 41-44 或 response 21-24)")
    print(f"  Regular: 2 (來自 stimulus 46-49 或 response 26-29)")
    print(f"  (原始 trigger codes 會在 epochs 函數中自動轉換)\n")
    
    # ========== 建立 Stimulus-locked Epochs ==========
    print("="*70)
    print("建立 Stimulus-locked epochs...")
    print("="*70)
    print("  參數:")
    print("    時間範圍: -0.8 to 1.0 s")
    print("    Baseline: None (稍後用整段平均)")
    
    epochs_stim = create_stimulus_locked_epochs(
        raw, 
        events,
        event_id, 
        tmin=-0.8, 
        tmax=1.0,
        baseline=None  # 不用 MNE 內建的 baseline
    )
    
    # 如果有 blocks_range，篩選 epochs
    if blocks_range is not None and hasattr(epochs_stim, 'metadata'):
        if 'block' in epochs_stim.metadata.columns:
            mask = (epochs_stim.metadata['block'] >= blocks_range[0]) & \
                   (epochs_stim.metadata['block'] <= blocks_range[1])
            epochs_stim = epochs_stim[mask]
            print(f"  ✓ 篩選 blocks {blocks_range[0]}-{blocks_range[1]}")
    
    print(f"  ✓ Stimulus epochs 建立完成")
    print(f"    Epochs 數量: {len(epochs_stim)}")
    print(f"    通道數: {len(epochs_stim.ch_names)}")
    print(f"    採樣點: {len(epochs_stim.times)}\n")
    
    # ========== 載入反應時間 ==========
    if response_times is None:
        print("="*70)
        print("載入反應時間...")
        print("="*70)
        
        if rt_file is None:
            rt_file = input("請輸入反應時間檔案路徑 (.txt/.csv): ").strip()
        
        if not os.path.exists(rt_file):
            print(f"  ✗ 檔案不存在: {rt_file}")
            return None
        
        # 載入 RT（假設單位是毫秒）
        response_times_ms = np.loadtxt(rt_file)
        response_times = response_times_ms / 1000.0  # 轉換為秒
        
        print(f"  ✓ 載入 {len(response_times)} 個反應時間")
        print(f"    RT 範圍: {response_times.min()*1000:.1f} - {response_times.max()*1000:.1f} ms")
        print(f"    RT 平均: {response_times.mean()*1000:.1f} ± {response_times.std()*1000:.1f} ms\n")
    else:
        print("使用已提供的 response_times\n")
    
    # 檢查 RT 數量是否和 events 一致
    if len(response_times) != len(events):
        print(f"  ⚠ 警告: RT 數量 ({len(response_times)}) 與 events 數量 ({len(events)}) 不一致")
        # 取較小值
        min_len = min(len(response_times), len(events))
        response_times = response_times[:min_len]
        events = events[:min_len]
        print(f"  已截斷為 {min_len} 個 trials\n")
    
    # ========== 建立 Response-locked Epochs ==========
    print("="*70)
    print("建立 Response-locked epochs...")
    print("="*70)
    print("  參數:")
    print("    時間範圍: -1.1 to 0.5 s")
    print("    Baseline: None (使用 Stimulus baseline)")
    
    epochs_resp = create_response_locked_epochs(
        raw,
        events,
        response_times,  # 傳入 RT
        event_id,
        tmin=-1.1,
        tmax=0.5,
        baseline=None
    )
    
    # 如果有 blocks_range，篩選 epochs
    if blocks_range is not None and hasattr(epochs_resp, 'metadata'):
        if 'block' in epochs_resp.metadata.columns:
            mask = (epochs_resp.metadata['block'] >= blocks_range[0]) & \
                   (epochs_resp.metadata['block'] <= blocks_range[1])
            epochs_resp = epochs_resp[mask]
    
    print(f"  ✓ Response epochs 建立完成")
    print(f"    Epochs 數量: {len(epochs_resp)}")
    
    # 檢查是否有 RT 資訊
    if hasattr(epochs_resp, 'metadata') and epochs_resp.metadata is not None:
        if 'rt' in epochs_resp.metadata.columns:
            RTs = epochs_resp.metadata['rt'].values
            print(f"    RT 範圍: {RTs.min()*1000:.1f} - {RTs.max()*1000:.1f} ms")
            print(f"    RT 平均: {RTs.mean()*1000:.1f} ± {RTs.std()*1000:.1f} ms")
        else:
            print("  ⚠ 警告: metadata 中找不到 'rt' 欄位")
            print("  請確保 create_response_locked_epochs 函數有加入 RT 到 metadata")
    else:
        print("  ⚠ 警告: epochs 沒有 metadata")
    print()
    
    # ========== 定義頻率參數 ==========
    print("="*70)
    print("設定頻率參數...")
    print("="*70)
    
    freqs = np.arange(4, 30, 1)
    n_cycles = freqs / 2.0
    
    print(f"  頻率範圍: {freqs[0]} - {freqs[-1]} Hz")
    print(f"  頻率點數: {len(freqs)}")
    print(f"  Wavelet cycles: freqs / 2.0")
    print(f"    最小: {n_cycles[0]:.1f} cycles @ {freqs[0]} Hz")
    print(f"    最大: {n_cycles[-1]:.1f} cycles @ {freqs[-1]} Hz")
    print(f"  時間解析度: ~{1000/(2*freqs[0]):.0f} - {1000/(2*freqs[-1]):.0f} ms\n")
    
    # ========== 計算 Response-locked ERSP ==========
    print("="*70)
    print("開始計算 Response-locked ERSP...")
    print("="*70)
    print("  方法:")
    print("    1. 計算 Stimulus-locked ERSP (保留每個 trial)")
    print("    2. 用整段 Stimulus epoch 平均做 baseline")
    print("    3. 根據 RT 重新對齊到 Response 時間軸")
    print("    4. 平均所有 trials\n")
    
    power_resp = stimulus_to_response_with_full_epoch_baseline(
        epochs_stim,
        epochs_resp,
        freqs,
        n_cycles,
        target_times=None,  # 使用預設 -1.1 to 0.5
        n_jobs=1
    )
    
    # ========== 建立輸出資料夾 ==========
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== 繪圖並儲存 ==========
    print("="*70)
    print("繪製並儲存結果...")
    print("="*70)
    
    # 1. Topomap
    output_file = os.path.join(output_dir, 
                               f'{subject_id}_response_ersp_full_baseline.png')
    
    print(f"  繪製 topomap...")
    fig = power_resp.plot_topo(
        baseline=None,  # 已經做過 baseline
        mode='logratio',
        title=f'{subject_id} - Response-locked ERSP\n'
              f'(Stimulus time alignment + Full-epoch baseline)',
        show=False
    )
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ 圖片已儲存: {output_file}")
    plt.close(fig)
    
    # 2. 儲存 TFR 資料
    default_tfr_fname = f'{subject_id}_response_ersp_full_baseline-tfr.h5'
    tfr_fname = input(f"\n請輸入 TFR 資料檔名 [預設: {default_tfr_fname}]: ").strip() or default_tfr_fname
    if not tfr_fname.endswith('.h5'):
        tfr_fname += '.h5'
    tfr_file = os.path.join(output_dir, tfr_fname)
    power_resp.save(tfr_file, overwrite=True)
    print(f"  ✓ 資料已儲存: {tfr_file}")
    
    # 3. 儲存分析參數
    params_file = os.path.join(output_dir,
                               f'{subject_id}_analysis_params.txt')
    with open(params_file, 'w', encoding='utf-8') as f:
        f.write("ASRT ERSP Analysis Parameters\n")
        f.write("="*70 + "\n\n")
        f.write(f"Subject: {subject_id}\n")
        f.write(f"Input file: {preprocessed_file}\n\n")
        
        f.write("Epochs:\n")
        f.write(f"  Stimulus-locked: -0.8 to 1.0 s\n")
        f.write(f"  Response-locked: -1.1 to 0.5 s\n")
        f.write(f"  Trials: {len(epochs_stim)}\n")
        if blocks_range:
            f.write(f"  Blocks: {blocks_range[0]} - {blocks_range[1]}\n")
        f.write("\n")
        
        f.write("Frequency:\n")
        f.write(f"  Range: {freqs[0]} - {freqs[-1]} Hz\n")
        f.write(f"  Points: {len(freqs)}\n")
        f.write(f"  Cycles: freqs / 2.0\n\n")
        
        f.write("Baseline:\n")
        f.write(f"  Method: Full Stimulus epoch average\n")
        f.write(f"  Range: -0.8 to 1.0 s (1.8 s total)\n")
        f.write(f"  Unit: dB (10*log10(power/baseline))\n\n")
        
        if hasattr(epochs_resp, 'metadata') and 'rt' in epochs_resp.metadata.columns:
            RTs = epochs_resp.metadata['rt'].values
            f.write("RT Statistics:\n")
            f.write(f"  Min: {RTs.min()*1000:.1f} ms\n")
            f.write(f"  Max: {RTs.max()*1000:.1f} ms\n")
            f.write(f"  Mean: {RTs.mean()*1000:.1f} ms\n")
            f.write(f"  SD: {RTs.std()*1000:.1f} ms\n")
    
    print(f"  ✓ 參數已儲存: {params_file}\n")
    
    # ========== 完成 ==========
    print("="*70)
    print("✓ 分析完成！")
    print("="*70)
    print(f"輸出檔案:")
    print(f"  1. {output_file}")
    print(f"  2. {tfr_file}")
    print(f"  3. {params_file}")
    print("="*70 + "\n")
    
    return power_resp