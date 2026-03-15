import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import stats

# 內部模組
from mne_python_analysis.spectral_analysis import (
    compute_roi_power_with_freq_baseline,
    compute_fft_power,
    compute_power_with_freq_baseline
)
from mne_python_analysis.roi_analysis import (
    define_roi_channels,
    create_virtual_channel_epochs
)

def asrt_testing_phase_topomap(epochs, subject_id):
    """
    Testing Phase 的 Motor/Perceptual 細分 Topomap 分析
    顯示：Motor-Regular, Motor-Random, Perceptual-Regular, Perceptual-Random
    
    Parameters
    ----------
    epochs : mne.Epochs
        包含 Testing Phase 的 epochs 資料
        需要 metadata 包含: 'test_type', 'trial_type', 'block'
    subject_id : str
        受試者 ID
    
    Returns
    -------
    results : dict
        包含四組條件的功率數據
    """
    print("\n" + "="*60)
    print("ASRT Testing Phase Topomap 分析")
    print("="*60)
    
    if epochs is None:
        print("⚠️  請先建立 Epochs")
        return None
    
    # 檢查 metadata
    if not hasattr(epochs, 'metadata') or epochs.metadata is None:
        print("⚠️  Epochs 沒有 metadata")
        return None
    
    required_cols = ['test_type', 'trial_type', 'block']
    for col in required_cols:
        if col not in epochs.metadata.columns:
            print(f"⚠️  Metadata 缺少 '{col}' 欄位")
            return None
    
    # 顯示資訊
    print(f"\n✓ Epochs 資訊:")
    print(f"  - 總 epochs: {len(epochs)}")
    print(f"  - Block 範圍: {epochs.metadata['block'].min()}-{epochs.metadata['block'].max()}")
    print(f"  - 時間範圍: {epochs.tmin:.2f} ~ {epochs.tmax:.2f} s")
    
    # 判斷 epoch 類型
    if epochs.tmin >= -0.9 and epochs.tmax >= 0.8:
        epoch_type = "Stimulus"
        print(f"  - 類型: Stimulus-locked")
    elif epochs.tmin <= -1.2 and epochs.tmax >= 0.4:
        epoch_type = "Response"
        print(f"  - 類型: Response-locked")
    else:
        print(f"  - 類型: 未知")
        epoch_type = input("\n請手動選擇類型 (stimulus/response): ").strip().lower()
        if epoch_type not in ['stimulus', 'response']:
            epoch_type = 'response'
    
    # === 1. 選擇頻段 ===
    print("\n" + "="*60)
    print("選擇頻段")
    print("="*60)
    
    if epoch_type == "Stimulus":
        print("推薦: Alpha (Stimulus-locked 分析)")
        print("1. Theta (4-8 Hz)")
        print("2. Alpha (8-13 Hz)")
        default_roi = '2'
    else:
        print("推薦: Theta (Response-locked 分析)")
        print("1. Theta (4-8 Hz)")
        print("2. Alpha (8-13 Hz)")
        default_roi = '1'
    
    roi_choice = input(f"\n請選擇 (1/2) [預設 {default_roi}]: ").strip()
    
    if roi_choice == '2' or (not roi_choice and default_roi == '2'):
        roi_name = 'alpha'
        fmin, fmax = 8, 13
    else:
        roi_name = 'theta'
        fmin, fmax = 4, 8
    
    print(f"\n✓ 選擇: {roi_name.upper()}")
    print(f"  - 頻率範圍: {fmin}-{fmax} Hz")
    
    # === 2. 設定時間窗口 ===
    if epoch_type == "Stimulus":
        task_tmin, task_tmax = 0.1, 0.3  # 修正後的窗口
        baseline_tmin, baseline_tmax = -0.5, -0.1  # 修正後的 baseline
        print(f"\n  - 分析窗口: {task_tmin*1000:.0f}-{task_tmax*1000:.0f} ms")
        print(f"  - Baseline: {baseline_tmin*1000:.0f}-{baseline_tmax*1000:.0f} ms")
    else:
        task_tmin, task_tmax = -0.3, 0.05
        baseline_tmin, baseline_tmax = -1.1, -0.6
        print(f"\n  - 分析窗口: {task_tmin*1000:.0f}-{task_tmax*1000:.0f} ms")
        print(f"  - Baseline: {baseline_tmin*1000:.0f}-{baseline_tmax*1000:.0f} ms")
    
    # === 3. 篩選 Testing Phase epochs (Block 27-34) ===
    print("\n" + "="*60)
    print("篩選 Testing Phase 資料 (Block 27-34)...")
    print("="*60)
    
    testing_mask = (epochs.metadata['block'] >= 27) & (epochs.metadata['block'] <= 34)
    epochs_testing = epochs[testing_mask]
    
    print(f"✓ Testing Phase epochs: {len(epochs_testing)}")
    
    # 排除 non-EEG channels
    eeg_channels = [ch for ch in epochs_testing.ch_names 
                    if epochs_testing.get_channel_types([ch])[0] == 'eeg']
    epochs_testing_eeg = epochs_testing.copy().pick_channels(eeg_channels)
    
    # === 4. 分離四組：Motor-Regular, Motor-Random, Perceptual-Regular, Perceptual-Random ===
    motor_regular = epochs_testing_eeg[
        (epochs_testing_eeg.metadata['test_type'] == 'motor') & 
        (epochs_testing_eeg.metadata['trial_type'] == 'Regular')
    ]
    motor_random = epochs_testing_eeg[
        (epochs_testing_eeg.metadata['test_type'] == 'motor') & 
        (epochs_testing_eeg.metadata['trial_type'] == 'Random')
    ]
    perceptual_regular = epochs_testing_eeg[
        (epochs_testing_eeg.metadata['test_type'] == 'perceptual') & 
        (epochs_testing_eeg.metadata['trial_type'] == 'Regular')
    ]
    perceptual_random = epochs_testing_eeg[
        (epochs_testing_eeg.metadata['test_type'] == 'perceptual') & 
        (epochs_testing_eeg.metadata['trial_type'] == 'Random')
    ]
    
    print(f"  - Motor-Regular: {len(motor_regular)}")
    print(f"  - Motor-Random: {len(motor_random)}")
    print(f"  - Perceptual-Regular: {len(perceptual_regular)}")
    print(f"  - Perceptual-Random: {len(perceptual_random)}")
    
    # === 5. 計算各組的功率 ===
    print(f"\n計算 {roi_name.upper()} 功率...")
    
    # Motor-Regular
    print("  - Motor-Regular...")
    mr_corrected, mr_baseline, mr_task = compute_power_with_freq_baseline(
        motor_regular, fmin=fmin, fmax=fmax,
        task_tmin=task_tmin, task_tmax=task_tmax,
        baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
        method='relative'
    )
    
    # Motor-Random
    print("  - Motor-Random...")
    mrn_corrected, mrn_baseline, mrn_task = compute_power_with_freq_baseline(
        motor_random, fmin=fmin, fmax=fmax,
        task_tmin=task_tmin, task_tmax=task_tmax,
        baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
        method='relative'
    )
    
    # Perceptual-Regular
    print("  - Perceptual-Regular...")
    pr_corrected, pr_baseline, pr_task = compute_power_with_freq_baseline(
        perceptual_regular, fmin=fmin, fmax=fmax,
        task_tmin=task_tmin, task_tmax=task_tmax,
        baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
        method='relative'
    )
    
    # Perceptual-Random
    print("  - Perceptual-Random...")
    prn_corrected, prn_baseline, prn_task = compute_power_with_freq_baseline(
        perceptual_random, fmin=fmin, fmax=fmax,
        task_tmin=task_tmin, task_tmax=task_tmax,
        baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
        method='relative'
    )
    
    # === 6. 計算各組的平均功率（跨 epochs） ===
    mr_mean = np.mean(mr_corrected, axis=0) * 100  # 轉換成百分比
    mrn_mean = np.mean(mrn_corrected, axis=0) * 100
    pr_mean = np.mean(pr_corrected, axis=0) * 100
    prn_mean = np.mean(prn_corrected, axis=0) * 100
    
    # === 7. 統計摘要 ===
    print("\n" + "="*60)
    print("統計摘要")
    print("="*60)
    
    print(f"\nMotor-Regular:")
    print(f"  M={np.mean(mr_mean):.2f}%, SD={np.std(mr_mean):.2f}%")
    
    print(f"\nMotor-Random:")
    print(f"  M={np.mean(mrn_mean):.2f}%, SD={np.std(mrn_mean):.2f}%")
    
    print(f"\nPerceptual-Regular:")
    print(f"  M={np.mean(pr_mean):.2f}%, SD={np.std(pr_mean):.2f}%")
    
    print(f"\nPerceptual-Random:")
    print(f"  M={np.mean(prn_mean):.2f}%, SD={np.std(prn_mean):.2f}%")
    
    # === 8. 繪製 2x2 Topomap ===
    print("\n生成 Topomap...")
    
    # 創建 2x2 子圖
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 找出全局 vmin/vmax（統一色階）
    all_data = np.concatenate([mr_mean, mrn_mean, pr_mean, prn_mean])
    vmin, vmax = np.percentile(all_data, [5, 95])
    
    # Motor-Regular (左上)
    mne.viz.plot_topomap(mr_mean, epochs_testing_eeg.info, axes=axes[0, 0],
                         show=False, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                         contours=6)
    axes[0, 0].set_title(f'Motor-Regular\nM={np.mean(mr_mean):.1f}%', fontsize=12, fontweight='bold')
    
    # Motor-Random (右上)
    mne.viz.plot_topomap(mrn_mean, epochs_testing_eeg.info, axes=axes[0, 1],
                         show=False, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                         contours=6)
    axes[0, 1].set_title(f'Motor-Random\nM={np.mean(mrn_mean):.1f}%', fontsize=12, fontweight='bold')
    
    # Perceptual-Regular (左下)
    mne.viz.plot_topomap(pr_mean, epochs_testing_eeg.info, axes=axes[1, 0],
                         show=False, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                         contours=6)
    axes[1, 0].set_title(f'Perceptual-Regular\nM={np.mean(pr_mean):.1f}%', fontsize=12, fontweight='bold')
    
    # Perceptual-Random (右下)
    im, _ = mne.viz.plot_topomap(prn_mean, epochs_testing_eeg.info, axes=axes[1, 1],
                                  show=False, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                                  contours=6)
    axes[1, 1].set_title(f'Perceptual-Random\nM={np.mean(prn_mean):.1f}%', fontsize=12, fontweight='bold')
    
    # 加上總標題
    fig.suptitle(f'{subject_id} - Testing Phase {roi_name.upper()} Power ({fmin}-{fmax} Hz)\n' + 
                 f'Time: {task_tmin*1000:.0f}-{task_tmax*1000:.0f} ms (% change from baseline)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 加上色條
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', 
                       fraction=0.04, pad=0.06, aspect=40)
    cbar.set_label('Power (% change from baseline)', fontsize=11)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    # 儲存
    filename = f'{subject_id}_testing_{roi_name}_motor_perceptual_topomap.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Topomap 已儲存: {filename}")
    plt.show(block=False)
    
    # === 9. 返回結果 ===
    results = {
        'motor_regular': {'corrected': mr_corrected, 'mean': mr_mean},
        'motor_random': {'corrected': mrn_corrected, 'mean': mrn_mean},
        'perceptual_regular': {'corrected': pr_corrected, 'mean': pr_mean},
        'perceptual_random': {'corrected': prn_corrected, 'mean': prn_mean}
    }
    
    print("\n" + "="*60)
    print("✓ 分析完成")
    print("="*60)
    
    return results

def asrt_testing_phase_detailed_topomap(epochs_test, epochs_eeg, subject_id,
                                        fmin, fmax, band_name,
                                        task_tmin, task_tmax,
                                        baseline_tmin, baseline_tmax,
                                        eeg_channels):
    """
    Testing Phase 的 Motor/Perceptual × Regular/Random 細分 Topomap
    
    顯示 2×2 layout:
    - Motor-Regular | Motor-Random
    - Perceptual-Regular | Perceptual-Random
    """
    print("\n" + "="*60)
    print("生成 Regular/Random 細分 Topomap...")
    print("="*60)
    
    # 分離四組
    motor_regular = epochs_test[
        (epochs_test.metadata['test_type'].str.lower() == 'motor') & 
        (epochs_test.metadata['trial_type'] == 'Regular')
    ]
    motor_random = epochs_test[
        (epochs_test.metadata['test_type'].str.lower() == 'motor') & 
        (epochs_test.metadata['trial_type'] == 'Random')
    ]
    perceptual_regular = epochs_test[
        (epochs_test.metadata['test_type'].str.lower() == 'perceptual') & 
        (epochs_test.metadata['trial_type'] == 'Regular')
    ]
    perceptual_random = epochs_test[
        (epochs_test.metadata['test_type'].str.lower() == 'perceptual') & 
        (epochs_test.metadata['trial_type'] == 'Random')
    ]
    
    print(f"  Motor-Regular: {len(motor_regular)} epochs")
    print(f"  Motor-Random: {len(motor_random)} epochs")
    print(f"  Perceptual-Regular: {len(perceptual_regular)} epochs")
    print(f"  Perceptual-Random: {len(perceptual_random)} epochs")
    
    # 計算各組的功率
    print(f"\n計算各組 {band_name} 功率...")
    
    def compute_group_power(epochs_group, ch_list):
        """計算一組 epochs 在所有電極的功率"""
        powers = []
        for ch in ch_list:
            power_rel, _, _ = compute_roi_power_with_freq_baseline(
                epochs_group, [ch], fmin=fmin, fmax=fmax,
                task_tmin=task_tmin, task_tmax=task_tmax,
                baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
                method='percent'
            )
            powers.append(np.mean(power_rel))
        return np.array(powers)
    
    print("  - Motor-Regular...")
    mr_power = compute_group_power(motor_regular, eeg_channels)
    
    print("  - Motor-Random...")
    mrn_power = compute_group_power(motor_random, eeg_channels)
    
    print("  - Perceptual-Regular...")
    pr_power = compute_group_power(perceptual_regular, eeg_channels)
    
    print("  - Perceptual-Random...")
    prn_power = compute_group_power(perceptual_random, eeg_channels)
    
    # 統計摘要
    print("\n統計摘要:")
    print(f"  Motor-Regular:      M={np.mean(mr_power):.1f}%, SD={np.std(mr_power):.1f}%")
    print(f"  Motor-Random:       M={np.mean(mrn_power):.1f}%, SD={np.std(mrn_power):.1f}%")
    print(f"  Perceptual-Regular: M={np.mean(pr_power):.1f}%, SD={np.std(pr_power):.1f}%")
    print(f"  Perceptual-Random:  M={np.mean(prn_power):.1f}%, SD={np.std(prn_power):.1f}%")
    
    # 繪製 2×2 Topomap
    fig, axes = plt.subplots(2, 2, figsize=(10, 11))
    
    # 統一色階使用 vlim (tuple)
    all_data = np.concatenate([mr_power, mrn_power, pr_power, prn_power])
    vlim_min, vlim_max = np.percentile(all_data, [5, 95])
    vlim = (vlim_min, vlim_max)  # tuple 格式
    
    # 調整 subplot 間距，為 colorbar 留空間
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.12, 
                       hspace=0.25, wspace=0.15)
    
    # Motor-Regular (左上)
    mne.viz.plot_topomap(mr_power, epochs_eeg.info, axes=axes[0, 0],
                         show=False, cmap='RdBu_r', vlim=vlim, contours=6)
    axes[0, 0].set_title(f'Motor-Regular\nM={np.mean(mr_power):.1f}%', 
                         fontsize=11, fontweight='bold')
    
    # Motor-Random (右上)
    mne.viz.plot_topomap(mrn_power, epochs_eeg.info, axes=axes[0, 1],
                         show=False, cmap='RdBu_r', vlim=vlim, contours=6)
    axes[0, 1].set_title(f'Motor-Random\nM={np.mean(mrn_power):.1f}%', 
                         fontsize=11, fontweight='bold')
    
    # Perceptual-Regular (左下)
    mne.viz.plot_topomap(pr_power, epochs_eeg.info, axes=axes[1, 0],
                         show=False, cmap='RdBu_r', vlim=vlim, contours=6)
    axes[1, 0].set_title(f'Perceptual-Regular\nM={np.mean(pr_power):.1f}%', 
                         fontsize=11, fontweight='bold')
    
    # Perceptual-Random (右下)
    im, _ = mne.viz.plot_topomap(prn_power, epochs_eeg.info, axes=axes[1, 1],
                                  show=False, cmap='RdBu_r', vlim=vlim, contours=6)
    axes[1, 1].set_title(f'Perceptual-Random\nM={np.mean(prn_power):.1f}%', 
                         fontsize=11, fontweight='bold')
    
    # 總標題
    fig.suptitle(f'{subject_id} - Testing Phase {band_name} Power ({fmin}-{fmax} Hz)\n' + 
                 f'Time: {task_tmin*1000:.0f}-{task_tmax*1000:.0f} ms',
                 fontsize=13, fontweight='bold')
    
    # 調整 colorbar 位置和大小
    # 在底部留出更多空間給 colorbar
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Power (% change from baseline)', fontsize=10)
    
    # 儲存
    filename = f'{subject_id}_testing_{band_name.lower()}_detailed_topomap.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ 細分圖已儲存: {filename}")
    plt.show(block=False)