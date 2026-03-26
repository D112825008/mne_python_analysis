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

def asrt_visualization(epochs, subject_id):
    """
    ASRT EEG 視覺化函數
    
    Parameters
    ----------
    epochs : mne.Epochs
        EEG epochs 資料
    subject_id : str
        受試者 ID
    
    Returns
    -------
    None
    """
    print("\n" + "="*60)
    print("ASRT EEG 視覺化")
    print("="*60)
            
    # 判斷 epoch 類型
    if epochs.tmin >= -0.6 and epochs.tmax <= 0.7:
        epoch_type = "Stimulus"
        task_tmin, task_tmax = 0.1, 0.3
        baseline_tmin, baseline_tmax = -0.5, -0.1
        recommended_band = "Alpha"
    else:
        epoch_type = "Response"
        task_tmin, task_tmax = -0.3, 0.05
        baseline_tmin, baseline_tmax = -0.75, -0.25
        recommended_band = "Theta"

    print(f"\nEpochs 資訊:")
    print(f"  - 類型: {epoch_type}-locked")
    print(f"  - Epochs 數量: {len(epochs)}")
    print(f"  - 時間範圍: {epochs.tmin:.2f} ~ {epochs.tmax:.2f} s")
    print(f"  - Analysis time window: {task_tmin*1000:.0f}-{task_tmax*1000:.0f} ms")
    print(f"  - Baseline: {baseline_tmin*1000:.0f}-{baseline_tmax*1000:.0f} ms")
    print(f"  - Recommended band: {recommended_band}")

    # 檢查metadata
    has_trial_type = hasattr(epochs, 'metadata') and 'trial_type' in epochs.metadata.columns
    has_phase = hasattr(epochs, 'metadata') and 'phase' in epochs.metadata.columns
    has_test_type = hasattr(epochs, 'metadata') and 'test_type' in epochs.metadata.columns
    
    # 顯示 metadata 診斷資訊
    if has_phase:
        unique_phases = epochs.metadata['phase'].unique()
        print(f"\n  - Metadata 中的 phase 值: {unique_phases}")
    
    # 選擇實驗階段
    print("\n" + "="*60)
    print("選擇實驗階段")
    print("="*60)
    print("1. Learning 階段")
    print("2. Testing 階段 (可進一步選擇 motor/perceptual)")
    print("3. 全部")
    
    phase_choice = input("\n請選擇 (1/2/3) [預設 3]: ").strip() or '3'
    
    # 根據選擇篩選 epochs
    if phase_choice == '1':
        if has_phase:
            learning_values = ['learning', 'Learning', 'LEARNING', 'learn', 'Learn']
            epochs_for_viz = None
            for val in learning_values:
                if val in epochs.metadata['phase'].values:
                    epochs_for_viz = epochs[epochs.metadata['phase'] == val]
                    phase_name = "Learning"
                    break
            if epochs_for_viz is None:
                print(f"⚠️  找不到 Learning 階段,使用全部 epochs")
                epochs_for_viz = epochs.copy()
                phase_name = "All"
        else:
            print("⚠️  Metadata 中沒有 phase 欄位,使用全部 epochs")
            epochs_for_viz = epochs.copy()
            phase_name = "All"
    elif phase_choice == '2':
        if has_phase:
            test_values = ['test', 'Test', 'TEST', 'testing', 'Testing', 'TESTING']
            epochs_for_viz = None
            for val in test_values:
                if val in epochs.metadata['phase'].values:
                    epochs_for_viz = epochs[epochs.metadata['phase'] == val]
                    phase_name = "Testing"
                    break
            if epochs_for_viz is None:
                print(f"⚠️  找不到 Testing 階段,使用全部 epochs")
                epochs_for_viz = epochs.copy()
                phase_name = "All"
        else:
            print("⚠️  Metadata 中沒有 phase 欄位,使用全部 epochs")
            epochs_for_viz = epochs.copy()
            phase_name = "All"
    else:
        epochs_for_viz = epochs.copy()
        phase_name = "All"
    
    print(f"\n✓ 選擇階段: {phase_name} ({len(epochs_for_viz)} epochs)")
    
    # 檢查篩選後的 epochs 數量
    if len(epochs_for_viz) == 0:
        print("⚠️  篩選後沒有任何 epochs,使用全部 epochs")
        epochs_for_viz = epochs.copy()
        phase_name = "All"
    
    # 如果選擇 Testing 階段,進一步選擇 motor/perceptual
    test_type_filter = None
    if phase_choice == '2' and has_test_type and len(epochs_for_viz) > 0:
        print("\n選擇 Testing 類型:")
        print("1. Motor")
        print("2. Perceptual")
        print("3. 全部")
        
        test_choice = input("\n請選擇 (1/2/3) [預設 3]: ").strip() or '3'
        
        if test_choice == '1':
            motor_values = ['motor', 'Motor', 'MOTOR']
            epochs_temp = None
            for val in motor_values:
                if val in epochs_for_viz.metadata['test_type'].values:
                    epochs_temp = epochs_for_viz[epochs_for_viz.metadata['test_type'] == val]
                    phase_name += " - Motor"
                    break
            if epochs_temp is not None and len(epochs_temp) > 0:
                epochs_for_viz = epochs_temp
            else:
                print(f"⚠️  找不到 Motor 條件或篩選後為空,使用所有 Testing epochs")
                
        elif test_choice == '2':
            perceptual_values = ['perceptual', 'Perceptual', 'PERCEPTUAL']
            epochs_temp = None
            for val in perceptual_values:
                if val in epochs_for_viz.metadata['test_type'].values:
                    epochs_temp = epochs_for_viz[epochs_for_viz.metadata['test_type'] == val]
                    phase_name += " - Perceptual"
                    break
            if epochs_temp is not None and len(epochs_temp) > 0:
                epochs_for_viz = epochs_temp
            else:
                print(f"⚠️  找不到 Perceptual 條件或篩選後為空,使用所有 Testing epochs")
        
        print(f"✓ 篩選後: {phase_name} ({len(epochs_for_viz)} epochs)")

    # 檢查篩選後的 metadata (使用篩選後的 epochs)
    has_trial_type = hasattr(epochs_for_viz, 'metadata') and 'trial_type' in epochs_for_viz.metadata.columns
    has_phase_filtered = hasattr(epochs_for_viz, 'metadata') and 'phase' in epochs_for_viz.metadata.columns
    has_test_type_filtered = hasattr(epochs_for_viz, 'metadata') and 'test_type' in epochs_for_viz.metadata.columns

    # 選擇要視覺化的內容
    while True:
        print("\n請選擇要視覺化的內容:")
        print("1. PSD Topomap (Regular vs Random)")
        print("2. 頻帶功率 Topomap (Theta/Alpha, with baseline correction)")
        print("3. TFR 比較 (Regular vs Random)")
        print("4. 單一電極頻段功率 (Regular vs Random)")
        print("5. Motor vs Perceptual 比較 (Testing phase)")
        print("0. 返回上層選單")

        choice = input("\n請選擇 (0-5): ").strip()
    
        if choice == '0':
            print("\n返回上層選單...")
            return

        # 排除非 EEG 電極 (使用篩選後的 epochs)
        eeg_channels = [ch for ch in epochs_for_viz.ch_names 
                        if epochs_for_viz.get_channel_types([ch])[0] == 'eeg']
        epochs_eeg = epochs_for_viz.copy().pick_channels(eeg_channels)

        # ================== 1. PSD Topomap (Regular vs Random) ==================
        if choice == '1':
            print("\n" + "="*60)
            print("1. PSD Topomap (Regular vs Random, with baseline correction)")
            print("="*60)
            
            if not has_trial_type:
                print("⚠️ Epochs 沒有 trial_type metadata")
                return
            
            # 分離條件
            epochs_reg = epochs_eeg[epochs_eeg.metadata['trial_type'] == 'Regular']
            epochs_ran = epochs_eeg[epochs_eeg.metadata['trial_type'] == 'Random']
            
            # 選擇頻帶
            print("\n選擇要顯示的頻帶:")
            print("1. Theta (4-8 Hz)")
            print("2. Alpha (8-13 Hz)")
            print("3. Beta (13-30 Hz)")
            print("4. 全部")
            
            band_choice = input("請選擇 (1-4) [預設 4]: ").strip() or '4'
            
            if band_choice == '4':
                bands = {'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30)}
            elif band_choice == '1':
                bands = {'Theta': (4, 8)}
            elif band_choice == '2':
                bands = {'Alpha': (8, 13)}
            else:
                bands = {'Beta': (13, 30)}
            
            print(f"\n計算 PSD (baseline corrected)...")
            
            # ========== 修正：使用 compute_roi_power_with_freq_baseline ==========
            band_results = {}
            
            for band_name, (fmin, fmax) in bands.items():
                print(f"  處理 {band_name}...")
                
                power_reg_all = []
                power_ran_all = []
                
                for ch in eeg_channels:
                    # Regular
                    power_rel_reg, _, _ = compute_roi_power_with_freq_baseline(
                        epochs_reg, [ch], fmin=fmin, fmax=fmax,
                        task_tmin=task_tmin, task_tmax=task_tmax,
                        baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
                        method='percent'
                    )
                    power_reg_all.append(np.mean(power_rel_reg))
                    
                    # Random
                    power_rel_ran, _, _ = compute_roi_power_with_freq_baseline(
                        epochs_ran, [ch], fmin=fmin, fmax=fmax,
                        task_tmin=task_tmin, task_tmax=task_tmax,
                        baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
                        method='percent'
                    )
                    power_ran_all.append(np.mean(power_rel_ran))
                
                band_results[band_name] = {
                    'regular': np.array(power_reg_all),
                    'random': np.array(power_ran_all),
                    'difference': np.array(power_reg_all) - np.array(power_ran_all)
                }
            
            # 繪圖
            n_bands = len(bands)
            fig, axes = plt.subplots(n_bands, 3, figsize=(15, 5*n_bands))
            
            if n_bands == 1:
                axes = axes.reshape(1, -1)
            
            for i, (band_name, results) in enumerate(band_results.items()):
                # Regular
                mne.viz.plot_topomap(results['regular'], epochs_eeg.info, 
                                    axes=axes[i, 0], show=False, cmap='RdBu_r', contours=6)
                axes[i, 0].set_title(f'{band_name} - Regular ({np.mean(results["regular"]):.1f}%)')
                
                # Random
                mne.viz.plot_topomap(results['random'], epochs_eeg.info, 
                                    axes=axes[i, 1], show=False, cmap='RdBu_r', contours=6)
                axes[i, 1].set_title(f'{band_name} - Random ({np.mean(results["random"]):.1f}%)')
                
                # Difference
                mne.viz.plot_topomap(results['difference'], epochs_eeg.info, 
                                    axes=axes[i, 2], show=False, cmap='RdBu_r', contours=6)
                axes[i, 2].set_title(f'{band_name} - Difference ({np.mean(results["difference"]):.1f}%)')
            
            fig.suptitle(f'{subject_id} - {phase_name} - PSD Topomap (% change from baseline)', fontsize=14)
            try:
                plt.tight_layout()
            except RuntimeError:
                pass
            
            phase_suffix = phase_name.replace(' ', '_').replace('-', '_')
            filename = f'{subject_id}_{phase_suffix}_psd_topomap_baseline_corrected.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ 已儲存: {filename}")
            plt.show(block=False)

        # ================== 2. PSD Topomap ==================
        elif choice == '2':
            print("\n" + "="*60)
            print("2. 頻帶功率 Topomap (Theta/Alpha, with baseline correction)")
            print("="*60)
            
            if not has_trial_type:
                print("⚠️ Epochs 沒有 trial_type metadata")
                return
            
            # 選擇頻帶
            print("\n選擇要顯示的頻帶:")
            print("1. Theta (4-8 Hz)")
            print("2. Alpha (8-13 Hz)")
            band_choice = input(f"請選擇 [預設 {'2'if epoch_type=='Stimulus' else '1'}: ").strip()
            
            if not band_choice:
                band_choice = '2' if epoch_type == 'Stimulus' else '1'
            
            if band_choice == '1':
                fmin, fmax = 4, 8
                band_name = 'Theta'
            else:
                fmin, fmax = 8, 13
                band_name = 'Alpha'
                
            print(f"\n計算 {band_name} 功率 (baseline corrected)...")
            
            # 分離條件
            epochs_reg = epochs_eeg[epochs_eeg.metadata['trial_type'] == 'Regular']
            epochs_ran = epochs_eeg[epochs_eeg.metadata['trial_type'] == 'Random']
            
            # 計算每個電極的功率
            power_reg_all = []
            power_ran_all = []
            
            for ch in eeg_channels:
                # Regular
                power_rel_reg, _, _ = compute_roi_power_with_freq_baseline(
                    epochs_reg, [ch], fmin=fmin, fmax=fmax,
                    task_tmin=task_tmin, task_tmax=task_tmax,
                    baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
                    method='percent'
                )
                power_reg_all.append(np.mean(power_rel_reg))
                
                # Random
                power_rel_ran, _, _ = compute_roi_power_with_freq_baseline(
                    epochs_ran, [ch], fmin=fmin, fmax=fmax,
                    task_tmin=task_tmin, task_tmax=task_tmax,
                    baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
                    method='percent'
                )
                power_ran_all.append(np.mean(power_rel_ran))
            
            power_reg_all = np.array(power_reg_all)
            power_ran_all = np.array(power_ran_all)
            power_diff = power_reg_all - power_ran_all
            
            
            # 繪製 topomap
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Regular
            mne.viz.plot_topomap(power_reg_all, epochs_eeg.info, axes=axes[0],
                                show=False, cmap='RdBu_r', contours=6)
            axes[0].set_title(f'Regular ({np.mean(power_reg_all):.1f}%)')
            
            # Random
            mne.viz.plot_topomap(power_ran_all, epochs_eeg.info, axes=axes[1],
                                show=False, cmap='RdBu_r', contours=6)
            axes[1].set_title(f'Random ({np.mean(power_ran_all):.1f}%)')
            
            # Difference
            mne.viz.plot_topomap(power_diff, epochs_eeg.info, axes=axes[2],
                                show=False, cmap='RdBu_r', contours=6)
            axes[2].set_title(f'Difference ({np.mean(power_diff):.1f}%)')
            
            fig.suptitle(f'{subject_id} - {band_name} Power (% change from baseline)', fontsize=14)
            try:
                plt.tight_layout()
            except RuntimeError:
                pass
            filename = f'{subject_id}_{band_name.lower()}_power_topomap_baseline_corrected.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ 已儲存: {filename}")
            plt.show(block=False)

        # ================== 3. TFR 比較 ==================
        elif choice == '3':
            print("\n" + "="*60)
            print("3. TFR Comparison (Regular vs Random)")
            print("="*60)
            
            if not has_trial_type:
                print("⚠️ Epochs does not have trial_type metadata")
                return
            
            # 選擇 ROI
            print("\nSelect analysis target:")
            print("1. Theta ROI (Fz, FCz, Cz, C3, C4)")
            print("2. Alpha ROI (O1, Oz, O2, P3, Pz, P4)")
            roi_choice = input(f"Please select [default {'2' if epoch_type=='Stimulus' else '1'}]: ").strip()
            
            if not roi_choice:
                roi_choice = '2' if epoch_type == 'Stimulus' else '1'
            
            if roi_choice == '1':
                roi_name = 'theta'
                fmin, fmax = 4, 8
            else:
                roi_name = 'alpha'
                fmin, fmax = 8, 13
            
            roi_channels = define_roi_channels(roi_name)
            roi_channels = [ch for ch in roi_channels if ch in eeg_channels]
            
            print(f"\nComputing {roi_name.upper()} ROI TFR...")
            
            # 分離條件
            epochs_reg = epochs_eeg[epochs_eeg.metadata['trial_type'] == 'Regular']
            epochs_ran = epochs_eeg[epochs_eeg.metadata['trial_type'] == 'Random']
            
            # 創建虛擬電極
            from mne_python_analysis.roi_analysis import create_virtual_channel_epochs
            epochs_reg_roi = create_virtual_channel_epochs(epochs_reg, roi_channels, f'{roi_name.upper()}_ROI')
            epochs_ran_roi = create_virtual_channel_epochs(epochs_ran, roi_channels, f'{roi_name.upper()}_ROI')
            
            # 計算 TFR（不做 baseline correction）
            freqs = np.arange(4, 40, 1)
            n_cycles = freqs / 2.0
            
            print("Computing time-frequency analysis...")
            print("  Computing Regular...")
            power_reg_roi = mne.time_frequency.tfr_morlet(
                epochs_reg_roi, freqs=freqs, n_cycles=n_cycles,
                use_fft=True, return_itc=False, average=True, n_jobs=1
            )
            
            print("  Computing Random...")
            power_ran_roi = mne.time_frequency.tfr_morlet(
                epochs_ran_roi, freqs=freqs, n_cycles=n_cycles,
                use_fft=True, return_itc=False, average=True, n_jobs=1
            )
            
            print("  Computing whole-brain TFR...")
            power_reg_wholehead = mne.time_frequency.tfr_morlet(
                epochs_reg, freqs=freqs, n_cycles=n_cycles,
                use_fft=True, return_itc=False, average=True, n_jobs=1
            )
            
            power_ran_wholehead = mne.time_frequency.tfr_morlet(
                epochs_ran, freqs=freqs, n_cycles=n_cycles,
                use_fft=True, return_itc=False, average=True, n_jobs=1
            )
            
            # 手動 apply baseline correction
            print("  Applying baseline correction...")
            power_reg_roi_bc = power_reg_roi.copy().apply_baseline(
                baseline=(baseline_tmin, baseline_tmax), mode='percent'
            )
            power_ran_roi_bc = power_ran_roi.copy().apply_baseline(
                baseline=(baseline_tmin, baseline_tmax), mode='percent'
            )
            power_reg_wholehead_bc = power_reg_wholehead.copy().apply_baseline(
                baseline=(baseline_tmin, baseline_tmax), mode='percent'
            )
            power_ran_wholehead_bc = power_ran_wholehead.copy().apply_baseline(
                baseline=(baseline_tmin, baseline_tmax), mode='percent'
            )
            
            # 繪圖：上半部 TFR，下半部 Topomap
            fig = plt.figure(figsize=(15, 12))
            gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1])
            
            # 上半部：TFR 圖
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            
            # Regular TFR
            power_reg_roi_bc.plot(picks=[0], baseline=None, mode='mean',
                                axes=ax1, show=False, colorbar=False)
            ax1.set_title(f'Regular - {roi_name.upper()} ROI')
            
            # Random TFR
            power_ran_roi_bc.plot(picks=[0], baseline=None, mode='mean',
                                axes=ax2, show=False, colorbar=False)
            ax2.set_title(f'Random - {roi_name.upper()} ROI')
            
            # Difference TFR
            power_diff_roi = power_reg_roi_bc.copy()
            power_diff_roi.data = power_reg_roi_bc.data - power_ran_roi_bc.data
            power_diff_roi.plot(picks=[0], baseline=None, mode='mean',
                            axes=ax3, show=False, colorbar=False)
            ax3.set_title(f'Difference (Regular - Random)')
            
            # 下半部：Topomap
            ax4 = fig.add_subplot(gs[1, 0])
            ax5 = fig.add_subplot(gs[1, 1])
            ax6 = fig.add_subplot(gs[1, 2])
            
            # Regular Topomap
            power_reg_wholehead_bc.plot_topomap(
                tmin=task_tmin, tmax=task_tmax, fmin=fmin, fmax=fmax,
                baseline=None, mode='mean', cmap='RdBu_r', axes=ax4, show=False
            )
            ax4.set_title('Regular (Topomap)')
            
            # Random Topomap
            power_ran_wholehead_bc.plot_topomap(
                tmin=task_tmin, tmax=task_tmax, fmin=fmin, fmax=fmax,
                baseline=None, mode='mean', cmap='RdBu_r', axes=ax5, show=False
            )
            ax5.set_title('Random (Topomap)')
            
            # Difference Topomap
            power_diff_wholehead = power_reg_wholehead_bc.copy()
            power_diff_wholehead.data = power_reg_wholehead_bc.data - power_ran_wholehead_bc.data
            power_diff_wholehead.plot_topomap(
                tmin=task_tmin, tmax=task_tmax, fmin=fmin, fmax=fmax,
                baseline=None, mode='mean', cmap='RdBu_r', axes=ax6, show=False
            )
            ax6.set_title('Difference (Topomap)')
            
            fig.suptitle(f'{subject_id} - TFR & Topomap Comparison ({roi_name.upper()})', fontsize=16)
            try:
                plt.tight_layout()
            except RuntimeError:
                pass
            
            filename = f'{subject_id}_tfr_topomap_comparison_{roi_name}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
            plt.show(block=False)

        # ================== 4. 單一電極頻帶功率 ==================
        elif choice == '4':
            print("\n" + "="*60)
            print("4. 單一電極頻帶功率")
            print("="*60)
            
            if not has_trial_type:
                print("⚠️ Epochs 沒有 trial_type metadata")
                return
            
            # 選擇電極
            print(f"\n可用電極: {eeg_channels[:10]}... (共 {len(eeg_channels)} 个)")
            electrode = input("請輸入電極名稱 [預設 Fz]: ").strip() or 'Fz'
            
            if electrode not in eeg_channels:
                print(f"⚠️ 電極 {electrode} 不存在，使用 Fz")
                electrode = 'Fz'
            
            # 選擇頻帶
            print("\n選擇頻帶:")
            print("1. Theta (4-8 Hz)")
            print("2. Alpha (8-13 Hz)")
            band_choice = input(f"請選擇 [預設 {'2' if epoch_type=='Stimulus' else '1'}]: ").strip()
            
            if not band_choice:
                band_choice = '2' if epoch_type == 'Stimulus' else '1'
            
            if band_choice == '1':
                fmin, fmax = 4, 8
                band_name = 'Theta'
            else:
                fmin, fmax = 8, 13
                band_name = 'Alpha'
            
            print(f"\n計算 {electrode} 的 {band_name} 功率...")
            
            # 分離條件
            epochs_reg = epochs_eeg[epochs_eeg.metadata['trial_type'] == 'Regular']
            epochs_ran = epochs_eeg[epochs_eeg.metadata['trial_type'] == 'Random']
            
            # 計算功率
            power_rel_reg, _, _ = compute_roi_power_with_freq_baseline(
                epochs_reg, [electrode], fmin=fmin, fmax=fmax,
                task_tmin=task_tmin, task_tmax=task_tmax,
                baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
                method='percent'
            )
            
            power_rel_ran, _, _ = compute_roi_power_with_freq_baseline(
                epochs_ran, [electrode], fmin=fmin, fmax=fmax,
                task_tmin=task_tmin, task_tmax=task_tmax,
                baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
                method='percent'
            )
            
            # 計算全腦功率（用於 topomap）
            print(f"  計算全腦 {band_name} 功率...")
            power_reg_all = []
            power_ran_all = []
            
            for ch in eeg_channels:
                # Regular
                power_rel_reg_ch, _, _ = compute_roi_power_with_freq_baseline(
                    epochs_reg, [ch], fmin=fmin, fmax=fmax,
                    task_tmin=task_tmin, task_tmax=task_tmax,
                    baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
                    method='percent'
                )
                power_reg_all.append(np.mean(power_rel_reg_ch))
                
                # Random
                power_rel_ran_ch, _, _ = compute_roi_power_with_freq_baseline(
                    epochs_ran, [ch], fmin=fmin, fmax=fmax,
                    task_tmin=task_tmin, task_tmax=task_tmax,
                    baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
                    method='percent'
                )
                power_ran_all.append(np.mean(power_rel_ran_ch))
            
            power_reg_all = np.array(power_reg_all)
            power_ran_all = np.array(power_ran_all)
            power_diff_all = power_reg_all - power_ran_all
            
            # 統計
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(power_rel_reg, power_rel_ran)
            
            print(f"\n{band_name} 功率 at {electrode}:")
            print(f"  Regular: M={np.mean(power_rel_reg):.2f}%, SD={np.std(power_rel_reg):.2f}%")
            print(f"  Random:  M={np.mean(power_rel_ran):.2f}%, SD={np.std(power_rel_ran):.2f}%")
            print(f"  t({len(power_rel_reg)+len(power_rel_ran)-2})={t_stat:.3f}, p={p_val:.4f}")
            
            # 繪製 topomap
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
            
            # 上半部：Bar plot 和 Violin plot
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            
            # Bar plot
            means = [np.mean(power_rel_reg), np.mean(power_rel_ran)]
            sems = [stats.sem(power_rel_reg), stats.sem(power_rel_ran)]
            x_pos = [0, 1]
            
            ax1.bar(x_pos, means, yerr=sems, color=['blue', 'red'], 
                alpha=0.7, capsize=5)
            ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(['Regular', 'Random'])
            ax1.set_ylabel(f'{band_name} Power (% change)')
            ax1.set_title(f'{electrode} - {band_name} Power')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Violin plot
            data_to_plot = [power_rel_reg, power_rel_ran]
            parts = ax2.violinplot(data_to_plot, positions=[0, 1], 
                                showmeans=True, showmedians=True)
            ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_xticks([0, 1])
            ax2.set_xticklabels(['Regular', 'Random'])
            ax2.set_ylabel(f'{band_name} Power (% change)')
            ax2.set_title('Distribution')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # 統計文字
            ax3.axis('off')
            stats_text = f'{band_name} Power at {electrode}\n\n'
            stats_text += f'Regular:\n'
            stats_text += f'  M = {np.mean(power_rel_reg):.2f}%\n'
            stats_text += f'  SD = {np.std(power_rel_reg):.2f}%\n\n'
            stats_text += f'Random:\n'
            stats_text += f'  M = {np.mean(power_rel_ran):.2f}%\n'
            stats_text += f'  SD = {np.std(power_rel_ran):.2f}%\n\n'
            stats_text += f'Statistical Test:\n'
            stats_text += f't({len(power_rel_reg)+len(power_rel_ran)-2}) = {t_stat:.3f}\n'
            stats_text += f'p = {p_val:.4f}\n'
            if p_val < 0.001:
                stats_text += '*** p < 0.001'
            elif p_val < 0.01:
                stats_text += '** p < 0.01'
            elif p_val < 0.05:
                stats_text += '* p < 0.05'
            else:
                stats_text += 'n.s.'
            
            ax3.text(0.5, 0.5, stats_text, ha='center', va='center',
                    fontsize=11, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            # 下半部：Topomap
            ax4 = fig.add_subplot(gs[1, 0])
            ax5 = fig.add_subplot(gs[1, 1])
            ax6 = fig.add_subplot(gs[1, 2])
            
            # Regular Topomap
            mne.viz.plot_topomap(power_reg_all, epochs_eeg.info, axes=ax4,
                                show=False, cmap='RdBu_r', contours=6)
            ax4.set_title(f'Regular ({np.mean(power_reg_all):.1f}%)')
            
            # 標記選定電極
            electrode_idx = eeg_channels.index(electrode)
            pos = mne.channels.layout._find_topomap_coords(epochs_eeg.info, picks=[electrode_idx])
            ax4.plot(pos[0, 0], pos[0, 1], 'k*', markersize=15, markeredgewidth=2)
            
            # Random Topomap
            mne.viz.plot_topomap(power_ran_all, epochs_eeg.info, axes=ax5,
                                show=False, cmap='RdBu_r', contours=6)
            ax5.set_title(f'Random ({np.mean(power_ran_all):.1f}%)')
            ax5.plot(pos[0, 0], pos[0, 1], 'k*', markersize=15, markeredgewidth=2)
            
            # Difference Topomap
            mne.viz.plot_topomap(power_diff_all, epochs_eeg.info, axes=ax6,
                                show=False, cmap='RdBu_r', contours=6)
            ax6.set_title(f'Difference ({np.mean(power_diff_all):.1f}%)')
            ax6.plot(pos[0, 0], pos[0, 1], 'k*', markersize=15, markeredgewidth=2)
            
            fig.suptitle(f'{subject_id} - {electrode} {band_name} Power & Topomap', fontsize=16)
            try:
                plt.tight_layout()
            except RuntimeError:
                pass
            
            filename = f'{subject_id}_{electrode}_{band_name.lower()}_power_topomap.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\n✓ 已儲存: {filename}")
            plt.show(block=False)


        # ================== 5. Motor vs Perceptual 比較 ==================
        if choice == '5':
            print("\n" + "="*60)
            print("5. Motor vs Perceptual Comparison (Testing phase)")
            print("="*60)
            
            if not (has_phase and has_test_type):
                print("⚠️ Epochs does not have phase or test_type metadata")
                continue
            
            # 檢查 metadata
            print("\nChecking metadata...")
            unique_phases = epochs_eeg.metadata['phase'].unique()
            print(f"  Available phase values: {unique_phases}")
            
            # 篩選 testing phase（支援多種命名方式）
            test_phase_values = ['test', 'testing', 'Test', 'Testing', 'TEST', 'TESTING']
            epochs_test = None
            matched_phase = None
            
            for phase_val in test_phase_values:
                if phase_val in unique_phases:
                    epochs_test = epochs_eeg[epochs_eeg.metadata['phase'] == phase_val]
                    matched_phase = phase_val
                    print(f"  Using phase = '{phase_val}'")
                    break
            
            if epochs_test is None or len(epochs_test) == 0:
                print(f"⚠️ No testing phase epochs found")
                print(f"   Available phase values: {unique_phases}")
                print(f"   Please check your epochs metadata")
                continue
            
            print(f"  Testing phase epochs: {len(epochs_test)}")
            
            # 檢查 test_type
            unique_test_types = epochs_test.metadata['test_type'].unique()
            print(f"  Available test_type values: {unique_test_types}")
            
            # 分離 motor 和 perceptual（支援大小寫）
            motor_values = ['motor', 'Motor', 'MOTOR']
            perceptual_values = ['perceptual', 'Perceptual', 'PERCEPTUAL']
            
            epochs_motor = None
            epochs_perceptual = None
            
            for motor_val in motor_values:
                if motor_val in unique_test_types:
                    epochs_motor = epochs_test[epochs_test.metadata['test_type'] == motor_val]
                    print(f"  Motor type: '{motor_val}'")
                    break
            
            for percep_val in perceptual_values:
                if percep_val in unique_test_types:
                    epochs_perceptual = epochs_test[epochs_test.metadata['test_type'] == percep_val]
                    print(f"  Perceptual type: '{percep_val}'")
                    break
            
            if epochs_motor is None or epochs_perceptual is None:
                print(f"⚠️ Cannot find motor or perceptual epochs")
                print(f"   Available test_type: {unique_test_types}")
                continue
            
            print(f"  Motor epochs: {len(epochs_motor)}")
            print(f"  Perceptual epochs: {len(epochs_perceptual)}")
            
            if len(epochs_motor) == 0 or len(epochs_perceptual) == 0:
                print(f"⚠️ Motor or Perceptual epochs count is 0")
                continue
            
            # 選擇頻段
            print("\nSelect frequency band:")
            print("1. Theta (4-8 Hz)")
            print("2. Alpha (8-13 Hz)")
            band_choice = input(f"Please select [default {'2' if epoch_type=='Stimulus' else '1'}]: ").strip()
            
            if not band_choice:
                band_choice = '2' if epoch_type == 'Stimulus' else '1'
            
            if band_choice == '1':
                fmin, fmax = 4, 8
                band_name = 'Theta'
            else:
                fmin, fmax = 8, 13
                band_name = 'Alpha'
            
            # ========== DEBUG 輸出 ==========
#            print(f"\n[DEBUG] band_choice = '{band_choice}'")
#            print(f"[DEBUG] band_name = '{band_name}'")
#            print(f"[DEBUG] fmin = {fmin}, fmax = {fmax}")
#            print(f"[DEBUG] task_tmin = {task_tmin}, task_tmax = {task_tmax}")
#            print(f"[DEBUG] baseline_tmin = {baseline_tmin}, baseline_tmax = {baseline_tmax}")
#            print(f"[DEBUG] epoch_type = '{epoch_type}'")
            # ================================
            
            print(f"\nComputing {band_name} power...")
            
            # 計算每個電極的功率
            power_motor_all = []
            power_perceptual_all = []
            
            for i, ch in enumerate(eeg_channels):
                # ========== 在第一個電極加入 DEBUG ==========
#                if i == 0:
#                    print(f"\n[DEBUG] First channel: {ch}")
#                    print(f"[DEBUG] Before compute_roi_power_with_freq_baseline:")
#                    print(f"  fmin = {fmin}, fmax = {fmax}")
#                    print(f"  task window = ({task_tmin}, {task_tmax})")
#                    print(f"  baseline window = ({baseline_tmin}, {baseline_tmax})")
                # ============================================
                
                # Motor
                power_rel_motor, power_task_motor, power_base_motor = compute_roi_power_with_freq_baseline(
                    epochs_motor, [ch], fmin=fmin, fmax=fmax,
                    task_tmin=task_tmin, task_tmax=task_tmax,
                    baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
                    method='percent'
                )
                
                # ========== 在第一個電極加入 DEBUG ==========
#                if i == 0:
#                    print(f"\n[DEBUG] After compute (Motor):")
#                    print(f"  power_rel shape = {power_rel_motor.shape}")
#                    print(f"  Mean power_rel = {np.mean(power_rel_motor):.2f}%")
#                    print(f"  Mean power_task = {np.mean(power_task_motor):.6f}")
#                    print(f"  Mean power_base = {np.mean(power_base_motor):.6f}")
#                    print(f"  Std power_rel = {np.std(power_rel_motor):.2f}%")
                # ============================================
                
                power_motor_all.append(np.mean(power_rel_motor))
                
                # Perceptual
                power_rel_perceptual, power_task_perceptual, power_base_perceptual = compute_roi_power_with_freq_baseline(
                    epochs_perceptual, [ch], fmin=fmin, fmax=fmax,
                    task_tmin=task_tmin, task_tmax=task_tmax,
                    baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
                    method='percent'
                )
                
                # ========== 在第一個電極加入 DEBUG ==========
#                if i == 0:
#                    print(f"\n[DEBUG] After compute (Perceptual):")
#                    print(f"  power_rel shape = {power_rel_perceptual.shape}")
#                    print(f"  Mean power_rel = {np.mean(power_rel_perceptual):.2f}%")
#                    print(f"  Mean power_task = {np.mean(power_task_perceptual):.6f}")
#                    print(f"  Mean power_base = {np.mean(power_base_perceptual):.6f}")
#                    print(f"  Std power_rel = {np.std(power_rel_perceptual):.2f}%")
                # ============================================
                
                power_perceptual_all.append(np.mean(power_rel_perceptual))
            
            power_motor_all = np.array(power_motor_all)
            power_perceptual_all = np.array(power_perceptual_all)
            power_diff = power_motor_all - power_perceptual_all
            
            # ========== 最終 DEBUG ==========
#            print(f"\n[DEBUG] Final results across all channels:")
#            print(f"  Motor: mean = {np.mean(power_motor_all):.2f}%, std = {np.std(power_motor_all):.2f}%")
#            print(f"  Perceptual: mean = {np.mean(power_perceptual_all):.2f}%, std = {np.std(power_perceptual_all):.2f}%")
#            print(f"  Difference: mean = {np.mean(power_diff):.2f}%, std = {np.std(power_diff):.2f}%")
#            print(f"  Number of channels = {len(power_motor_all)}")
            # ================================
            
            # 繪圖
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Motor
            mne.viz.plot_topomap(power_motor_all, epochs_eeg.info, axes=axes[0],
                                show=False, cmap='RdBu_r', contours=6)
            axes[0].set_title(f'Motor ({np.mean(power_motor_all):.1f}%)')
            
            # Perceptual
            mne.viz.plot_topomap(power_perceptual_all, epochs_eeg.info, axes=axes[1],
                                show=False, cmap='RdBu_r', contours=6)
            axes[1].set_title(f'Perceptual ({np.mean(power_perceptual_all):.1f}%)')
            
            # Difference
            mne.viz.plot_topomap(power_diff, epochs_eeg.info, axes=axes[2],
                                show=False, cmap='RdBu_r', contours=6)
            axes[2].set_title(f'Motor - Perceptual ({np.mean(power_diff):.1f}%)')
            
            fig.suptitle(f'{subject_id} - Motor vs Perceptual ({band_name})', fontsize=14)
            try:
                plt.tight_layout()
            except RuntimeError:
                pass
            
            filename = f'{subject_id}_motor_vs_perceptual_{band_name.lower()}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved: {filename}")
            plt.show(block=False)
            
            # 詢問是否要看細分圖
            print("\n" + "="*60)
            show_detailed = input("是否要顯示 Regular/Random 細分圖？(y/n) [n]: ").strip().lower()
            
            if show_detailed == 'y':
                # 呼叫細分函數
                asrt_testing_phase_detailed_topomap(
                    epochs_test, epochs_eeg, subject_id, 
                    fmin, fmax, band_name,
                    task_tmin, task_tmax,
                    baseline_tmin, baseline_tmax,
                    eeg_channels
                )

        print("\n" + "="*60)
        print("✓ 視覺化完成")
        print("="*60)
    
def asrt_wholebrain_fft_analysis(epochs, subject_id):
    """
    全腦 FFT 功率分析 + Topomap
    """
    print("\n" + "="*60)
    print("全腦 FFT 功率分析")
    print("="*60)
    
    # 判斷 epoch 類型
    if epochs.tmin >= -0.9 and epochs.tmax >= 0.8:
        epoch_type = "Stimulus"
        task_tmin, task_tmax = 0.1, 0.3
        baseline_tmin, baseline_tmax = -0.5, -0.1
    else:
        epoch_type = "Response"
        task_tmin, task_tmax = -0.3, 0.05
        baseline_tmin, baseline_tmax = -1.0, -0.6
    
    print(f"  - Epoch 類型: {epoch_type}-locked")
    print(f"  - 分析窗口: {task_tmin*1000:.0f}-{task_tmax*1000:.0f} ms")
    print(f"  - Baseline: {baseline_tmin*1000:.0f}-{baseline_tmax*1000:.0f} ms")
    
    # 檢查 metadata
    has_phase = hasattr(epochs, 'metadata') and 'phase' in epochs.metadata.columns
    has_test_type = hasattr(epochs, 'metadata') and 'test_type' in epochs.metadata.columns
    
    # 顯示 metadata 中實際存在的值
    if has_phase:
        unique_phases = epochs.metadata['phase'].unique()
        print(f"\n  - Metadata 中的 phase 值: {unique_phases}")
    
    # 選擇實驗階段
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
            learning_values = ['learning', 'Learning', 'LEARNING', 'learn', 'Learn']
            epochs_filtered = None
            for val in learning_values:
                if val in epochs.metadata['phase'].values:
                    epochs_filtered = epochs[epochs.metadata['phase'] == val]
                    phase_name = "Learning"
                    break
            if epochs_filtered is None:
                print(f"⚠️  找不到 Learning 階段,使用全部 epochs")
                epochs_filtered = epochs.copy()
                phase_name = "All"
        else:
            print("⚠️  Metadata 中沒有 phase 欄位,使用全部 epochs")
            epochs_filtered = epochs.copy()
            phase_name = "All"
    elif phase_choice == '2':
        if has_phase:
            test_values = ['test', 'Test', 'TEST', 'testing', 'Testing', 'TESTING']
            epochs_filtered = None
            for val in test_values:
                if val in epochs.metadata['phase'].values:
                    epochs_filtered = epochs[epochs.metadata['phase'] == val]
                    phase_name = "Testing"
                    break
            if epochs_filtered is None:
                print(f"⚠️  找不到 Testing 階段,使用全部 epochs")
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
        print("⚠️  篩選後沒有任何 epochs,使用全部 epochs")
        epochs_filtered = epochs.copy()
        phase_name = "All"
    
    # 如果選擇 Testing 階段,進一步選擇 motor/perceptual
    test_type_filter = None
    if phase_choice == '2' and has_test_type and len(epochs_filtered) > 0:
        print("\n選擇 Testing 類型:")
        print("1. Motor")
        print("2. Perceptual")
        print("3. 全部")
        
        test_choice = input("\n請選擇 (1/2/3) [預設 3]: ").strip() or '3'
        
        if test_choice == '1':
            motor_values = ['motor', 'Motor', 'MOTOR']
            epochs_temp = None
            for val in motor_values:
                if val in epochs_filtered.metadata['test_type'].values:
                    epochs_temp = epochs_filtered[epochs_filtered.metadata['test_type'] == val]
                    phase_name += " - Motor"
                    break
            if epochs_temp is not None and len(epochs_temp) > 0:
                epochs_filtered = epochs_temp
            else:
                print(f"⚠️  找不到 Motor 條件或篩選後為空,使用所有 Testing epochs")
                
        elif test_choice == '2':
            perceptual_values = ['perceptual', 'Perceptual', 'PERCEPTUAL']
            epochs_temp = None
            for val in perceptual_values:
                if val in epochs_filtered.metadata['test_type'].values:
                    epochs_temp = epochs_filtered[epochs_filtered.metadata['test_type'] == val]
                    phase_name += " - Perceptual"
                    break
            if epochs_temp is not None and len(epochs_temp) > 0:
                epochs_filtered = epochs_temp
            else:
                print(f"⚠️  找不到 Perceptual 條件或篩選後為空,使用所有 Testing epochs")
        
        print(f"✓ 篩選後: {phase_name} ({len(epochs_filtered)} epochs)")
    
    # 選擇頻帶
    print("\n選擇要分析的頻帶:")
    print("1. Theta (4-8 Hz)")
    print("2. Alpha (8-13 Hz)")
    print("3. Beta (13-30 Hz)")
    print("4. 全部")
    
    band_choice = input("請選擇 (1-4) [預設 4]: ").strip() or '4'
    
    if band_choice == '1':
        bands = {'Theta': (4, 8)}
    elif band_choice == '2':
        bands = {'Alpha': (8, 13)}
    elif band_choice == '3':
        bands = {'Beta': (13, 30)}
    else:
        bands = {'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30)}
    
    # 排除非 EEG 電極
    eeg_channels = [ch for ch in epochs_filtered.ch_names 
                    if epochs_filtered.get_channel_types([ch])[0] == 'eeg']
    epochs_eeg = epochs_filtered.copy().pick_channels(eeg_channels)
    
    # 分離 Regular 和 Random
    if hasattr(epochs_filtered, 'metadata') and 'trial_type' in epochs_filtered.metadata.columns:
        epochs_regular = epochs_eeg[epochs_eeg.metadata['trial_type'] == 'Regular']
        epochs_random = epochs_eeg[epochs_eeg.metadata['trial_type'] == 'Random']
        has_conditions = True
    else:
        has_conditions = False
    
    # 計算各頻帶的功率
    results = {}
    
    for band_name, (fmin, fmax) in bands.items():
        print(f"\n計算 {band_name} ({fmin}-{fmax} Hz)...")
        
        if has_conditions:
            # Regular
            power_rel_reg, power_task_reg, power_base_reg = compute_power_with_freq_baseline(
                epochs_regular, fmin=fmin, fmax=fmax,
                task_tmin=task_tmin, task_tmax=task_tmax,
                baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
                method='percent'
            )
            
            # Random
            power_rel_ran, power_task_ran, power_base_ran = compute_power_with_freq_baseline(
                epochs_random, fmin=fmin, fmax=fmax,
                task_tmin=task_tmin, task_tmax=task_tmax,
                baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
                method='percent'
            )
            
            # 計算平均（跨 epochs）
            avg_power_reg = np.mean(power_rel_reg, axis=0)  # (n_channels,)
            avg_power_ran = np.mean(power_rel_ran, axis=0)
            avg_power_diff = avg_power_reg - avg_power_ran
            
            results[band_name] = {
                'regular': avg_power_reg,
                'random': avg_power_ran,
                'difference': avg_power_diff
            }
        else:
            # 全部一起計算
            power_rel, _, _ = compute_power_with_freq_baseline(
                epochs_eeg, fmin=fmin, fmax=fmax,
                task_tmin=task_tmin, task_tmax=task_tmax,
                baseline_tmin=baseline_tmin, baseline_tmax=baseline_tmax,
                method='percent'
            )
            
            avg_power = np.mean(power_rel, axis=0)
            results[band_name] = {'all': avg_power}
    
    # === 視覺化：Topomap ===
    print("\n生成 Topography 圖...")
    
    if has_conditions:
        # 3 列：Regular, Random, Difference
        n_bands = len(bands)
        fig, axes = plt.subplots(n_bands, 3, figsize=(12, 4*n_bands))
        
        if n_bands == 1:
            axes = axes.reshape(1, -1)
        
        for i, (band_name, band_data) in enumerate(results.items()):
            # Regular
            im1, cn1 = mne.viz.plot_topomap(
                band_data['regular'], epochs_eeg.info,
                axes=axes[i, 0], show=False, cmap='RdBu_r',
                contours=6, vlim=(None, None)
            )
            axes[i, 0].set_title(f'{band_name} - Regular ({np.mean(band_data["regular"]):.1f}%)')
            
            # Random
            im2, cn2 = mne.viz.plot_topomap(
                band_data['random'], epochs_eeg.info,
                axes=axes[i, 1], show=False, cmap='RdBu_r',
                contours=6, vlim=(None, None)
            )
            axes[i, 1].set_title(f'{band_name} - Random ({np.mean(band_data["random"]):.1f}%)')
            
            # Difference
            im3, cn3 = mne.viz.plot_topomap(
                band_data['difference'], epochs_eeg.info,
                axes=axes[i, 2], show=False, cmap='RdBu_r',
                contours=6, vlim=(None, None)
            )
            axes[i, 2].set_title(f'{band_name} - Difference ({np.mean(band_data["difference"]):.1f}%)')
        
        plt.suptitle(f'{subject_id} - {phase_name} - Whole-Brain FFT Power (% change from baseline)', 
                    fontsize=14, y=0.98)
        
        # 為 colorbar 預留空間
        fig.subplots_adjust(bottom=0.12, top=0.92)
        
        # 在底部加入共用的 colorbar
        cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02])
        cbar = fig.colorbar(im3, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Power (% change from baseline)', fontsize=11)
            
    else:
        # 只有 1 列
        n_bands = len(bands)
        fig, axes = plt.subplots(1, n_bands, figsize=(5*n_bands, 5))
        
        if n_bands == 1:
            axes = [axes]
        
        for i, (band_name, band_data) in enumerate(results.items()):
            im, cn = mne.viz.plot_topomap(
                band_data['all'], epochs_eeg.info,
                axes=axes[i], show=False, cmap='RdBu_r',
                contours=6, vlim=(None, None)
            )
            axes[i].set_title(f'{band_name}')
        
        plt.suptitle(f'{subject_id} - {phase_name} - Whole-Brain FFT Power (% change from baseline)', 
                    fontsize=14, y=0.98)
        
        fig.subplots_adjust(bottom=0.12, top=0.92)
        
        # 加入 colorbar
        cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Power (% change from baseline)', fontsize=11)
    
    phase_suffix = phase_name.replace(' ', '_').replace('-', '_')
    filename = f'{subject_id}_{phase_suffix}_wholebrain_fft_topomap.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ 已儲存: {filename}")
    plt.show(block=False)
    
    # === 輸出統計摘要 ===
    print("\n" + "="*60)
    print("統計摘要")
    print("="*60)
    
    if has_conditions:
        for band_name, band_data in results.items():
            print(f"\n{band_name}:")
            print(f"  Regular: M={np.mean(band_data['regular']):.2f}%, "
                  f"Max={np.max(band_data['regular']):.2f}% at {eeg_channels[np.argmax(band_data['regular'])]}")
            print(f"  Random:  M={np.mean(band_data['random']):.2f}%, "
                  f"Max={np.max(band_data['random']):.2f}% at {eeg_channels[np.argmax(band_data['random'])]}")
            print(f"  Difference: M={np.mean(band_data['difference']):.2f}%, "
                  f"Max={np.max(np.abs(band_data['difference'])):.2f}% at "
                  f"{eeg_channels[np.argmax(np.abs(band_data['difference']))]}")
    else:
        for band_name, band_data in results.items():
            print(f"\n{band_name}:")
            print(f"  Mean: {np.mean(band_data['all']):.2f}%")
            print(f"  Max: {np.max(band_data['all']):.2f}% at {eeg_channels[np.argmax(band_data['all'])]}")
    
    print("\n" + "="*60)
    print("✓ 分析完成")
    print("="*60)
    
    return results