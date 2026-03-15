import numpy as np
import mne
from pathlib import Path

# 新增：用於 Response 重新對齊
from scipy.interpolate import interp1d

# 內部模組
from mne_python_analysis.roi_analysis import define_roi_channels

# 同一資料夾內的模組
from .ersp_plots import (
    plot_ersp_lum2023_style,
    plot_ersp_comparison,
    plot_learning_comparison,
    plot_testing_comparison
)

# 群體分析（可選）
try:
    from mne_python_analysis.group_ersp_analysis import save_subject_ersp
    HAS_GROUP_ANALYSIS = True
except ImportError:
    HAS_GROUP_ANALYSIS = False

def _save_ersp_h5(power_dict, filepath):
    """
    將 power_dict 儲存為 HDF5 (.h5) 格式。

    Parameters
    ----------
    power_dict : dict
        {roi_name: {'power': ndarray, 'times': ndarray,
                    'freqs': ndarray, 'channels': list}}
    filepath : str or Path
        輸出的 .h5 檔案路徑
    """
    try:
        import h5py
        with h5py.File(filepath, 'w') as f:
            for roi_name, roi_data in power_dict.items():
                grp = f.create_group(roi_name)
                grp.create_dataset('power',   data=roi_data['power'])
                grp.create_dataset('times',   data=roi_data['times'])
                grp.create_dataset('freqs',   data=roi_data['freqs'])
                # channels 為字串列表，轉成 bytes 儲存
                ch_bytes = [ch.encode('utf-8') for ch in roi_data['channels']]
                grp.create_dataset('channels', data=ch_bytes)
        print(f"  ✓ 已儲存: {filepath}")
    except ImportError:
        # h5py 未安裝，改用 numpy npz 格式
        npz_path = str(filepath).replace('.h5', '.npz')
        flat = {}
        for roi_name, roi_data in power_dict.items():
            flat[f"{roi_name}_power"] = roi_data['power']
            flat[f"{roi_name}_times"] = roi_data['times']
            flat[f"{roi_name}_freqs"] = roi_data['freqs']
        np.savez(npz_path, **flat)
        print(f"  ⚠ h5py 未安裝，已改存為 npz: {npz_path}")


def _compute_single_ersp(epochs_subset, available_groups, freqs, n_cycles, baseline_window):
    """
    對一組 epochs 計算各 ROI 的 ERSP，回傳 power_dict。

    Parameters
    ----------
    epochs_subset : mne.Epochs
    available_groups : dict  {roi_name: [channel_names]}
    freqs : ndarray
    n_cycles : ndarray
    baseline_window : tuple (tmin, tmax)

    Returns
    -------
    power_dict : dict or None
    """
    power_dict = {}

    for roi_name, channels in available_groups.items():
        # 只挑這次 subset 裡真的存在的電極（避免因 subset 缺少電極而出錯）
        valid_ch = [ch for ch in channels if ch in epochs_subset.ch_names]
        if not valid_ch:
            print(f"    ⚠ {roi_name} ROI: subset 中無可用電極，跳過")
            continue

        print(f"    計算 {roi_name} ROI ({len(epochs_subset)} trials)...")
        epochs_roi = epochs_subset.copy().pick_channels(valid_ch)

        power = mne.time_frequency.tfr_morlet(
            epochs_roi,
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=False,
            average=True,
            n_jobs=-1
        )
        power.apply_baseline(mode='logratio', baseline=baseline_window)

        power_dict[roi_name] = {
            'power':    power.data.mean(axis=0),   # (freqs, times)
            'times':    power.times,
            'freqs':    power.freqs,
            'channels': valid_ch
        }
        print(f"    ✓ {roi_name} 完成")

    return power_dict if power_dict else None


def asrt_ersp_analysis(epochs, subject_id, freqs=None, n_cycles=None, output_dir='./'):
    """
    ASRT ERSP 分析（使用 Dillian 的參數 + Lum et al. 2023 視覺化風格）

    若 metadata 包含 'test_type' 欄位，自動分為 3 個條件：
      - Learning phase
      - Test phase / motor
      - Test phase / perceptual
    各條件分別計算 ERSP，儲存獨立 .h5 檔案並產生 3 張圖。

    否則維持原始行為（單一 power_dict）。

    Parameters
    ----------
    epochs : mne.Epochs
    subject_id : str
    freqs : ndarray or None
    n_cycles : ndarray or None
    output_dir : str

    Returns
    -------
    power_dict or dict of power_dicts
    """

    # === 1. 設定頻率參數 ===
    if freqs is None:
        freqs = np.logspace(np.log10(4), np.log10(30), num=50)
    if n_cycles is None:
        n_cycles = freqs / 2.0

    # === 2. 判斷 baseline 窗口（根據 epoch 類型）===
    if epochs.tmin >= -0.9 and epochs.tmax >= 0.8:
        baseline_window = (-0.5, -0.1)
        epoch_type = "Stimulus-locked"
    else:
        baseline_window = (-1.1, -0.6)
        epoch_type = "Response-locked"

    print(f"\n{'='*60}")
    print(f"ERSP 分析 (使用 Dillian 的參數)")
    print(f"{'='*60}")
    print(f"受試者: {subject_id}")
    print(f"Epochs 總數: {len(epochs)} trials")
    print(f"Epoch 類型: {epoch_type}")
    print(f"頻率範圍: {freqs[0]:.2f} - {freqs[-1]:.2f} Hz ({len(freqs)} steps)")
    print(f"Wavelet cycles: {n_cycles[0]:.2f} - {n_cycles[-1]:.2f}")
    print(f"時間範圍: {epochs.tmin:.3f} - {epochs.tmax:.3f} s")
    print(f"Baseline 窗口: {baseline_window[0]:.2f} - {baseline_window[1]:.2f} s")

    # === 3. 定義 ROI ===
    roi_groups = {
        'Theta': ['Fz', 'FCz', 'Cz', 'C3', 'C4'],
        'Alpha': ['O1', 'Oz', 'O2', 'P3', 'Pz', 'P4']
    }
    available_groups = {}
    for roi_name, channels in roi_groups.items():
        available = [ch for ch in channels if ch in epochs.ch_names]
        if available:
            available_groups[roi_name] = available
            print(f"  {roi_name} ROI: {available}")
        else:
            print(f"  {roi_name} ROI: ⚠️  無可用電極")

    if not available_groups:
        print("⚠️  沒有可用的 ROI！")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === 4. 依 metadata 決定分析路徑 ===
    # 只有當 epochs 同時包含 Learning 和 Test 兩種 phase 時才分條件
    # 避免 _ersp_testing_phase() 傳入的子集（單一 phase）被再次拆分
    has_test_type = (
        epochs.metadata is not None and
        'test_type' in epochs.metadata.columns and
        'phase' in epochs.metadata.columns and
        epochs.metadata['phase'].nunique() >= 2
    )

    if has_test_type:
        # ── 多條件路徑：Learning / MotorTest / PerceptualTest ──
        print(f"\n  偵測到 'test_type' 欄位 → 分 3 個條件計算")

        conditions = {
            'Learning':       epochs[epochs.metadata['phase'] == 'Learning'],
            'MotorTest':      epochs[
                (epochs.metadata['phase'] == 'Test') &
                (epochs.metadata['test_type'] == 'motor')
            ],
            'PerceptualTest': epochs[
                (epochs.metadata['phase'] == 'Test') &
                (epochs.metadata['test_type'] == 'perceptual')
            ],
        }

        # 欄位名稱對應到檔名標籤
        file_labels = {
            'Learning':       'Learning',
            'MotorTest':      'MotorTest',
            'PerceptualTest': 'PerceptualTest',
        }

        results = {}

        for cond_name, epochs_subset in conditions.items():
            print(f"\n{'─'*60}")
            print(f"  條件: {cond_name}  ({len(epochs_subset)} trials)")
            print(f"{'─'*60}")

            if len(epochs_subset) == 0:
                print(f"  ⚠  {cond_name}: 沒有資料，跳過")
                continue

            power_dict = _compute_single_ersp(
                epochs_subset, available_groups, freqs, n_cycles, baseline_window
            )

            if power_dict is None:
                print(f"  ⚠  {cond_name}: 計算失敗，跳過")
                continue

            # 儲存 .h5
            label = file_labels[cond_name]
            h5_path = output_dir / f"{subject_id}_Stimulus_{label}_ERSP.h5"
            _save_ersp_h5(power_dict, h5_path)

            # 繪圖（reuse 現有函數，帶條件名稱）
            plot_ersp_lum2023_style(
                power_dict,
                f"{subject_id}_{cond_name}",
                str(output_dir)
            )

            results[cond_name] = power_dict

        print(f"\n{'='*60}")
        print(f"✓ 多條件 ERSP 完成，共 {len(results)} 個條件")
        print(f"{'='*60}")
        return results

    else:
        # ── 原始單一條件路徑 ──
        power_dict = {}

        for roi_name, channels in available_groups.items():
            print(f"\n計算 {roi_name} ROI 時頻功率...")
            epochs_roi = epochs.copy().pick_channels(channels)

            power = mne.time_frequency.tfr_morlet(
                epochs_roi,
                freqs=freqs,
                n_cycles=n_cycles,
                use_fft=True,
                return_itc=False,
                average=True,
                n_jobs=-1
            )
            power.apply_baseline(mode='logratio', baseline=baseline_window)

            power_dict[roi_name] = {
                'power':    power.data.mean(axis=0),
                'times':    power.times,
                'freqs':    power.freqs,
                'channels': channels
            }
            print(f"  ✓ {roi_name} 完成")

        plot_ersp_lum2023_style(power_dict, subject_id, str(output_dir))
        return power_dict

def asrt_ersp_comparison(epochs_dict, subject_id, condition_labels, 
                         freqs=None, n_cycles=None, output_dir='./'):
    """
    比較不同條件的 ERSP（例如 Regular vs Random）
    
    Parameters
    ----------
    epochs_dict : dict
        {condition_name: epochs} 的字典
        例如 {'Regular': epochs_block6, 'Random': epochs_block7}
    subject_id : str
        受試者 ID
    condition_labels : list
        條件標籤，例如 ['Regular (Block 6)', 'Random (Block 7)']
    freqs : ndarray or None
        頻率陣列
    n_cycles : ndarray or None
        Wavelet cycles
    output_dir : str
        輸出目錄
        
    Returns
    -------
    power_by_condition : dict
        各條件的功率資料
    """
    
    if freqs is None:
        freqs = np.logspace(np.log10(4), np.log10(30), num=50)  # 4-30 Hz
    if n_cycles is None:
        n_cycles = freqs / 2.0  # 頻率相依
    
    # === 計算各條件的 ERSP ===
    power_by_condition = {}
    
    for cond_name, epochs in epochs_dict.items():
        print(f"\n{'='*60}")
        print(f"計算 {cond_name} 條件...")
        print(f"{'='*60}")
        power_dict = asrt_ersp_analysis(
            epochs, subject_id=f"{subject_id}_{cond_name}",
            freqs=freqs, n_cycles=n_cycles, output_dir=output_dir
        )
        power_by_condition[cond_name] = power_dict
    
    # === 繪製比較圖 ===
    plot_ersp_comparison(power_by_condition, subject_id, condition_labels, output_dir)
    
    return power_by_condition

def asrt_ersp_full_analysis(epochs, subject_id, phase='learning', lock_type='stimulus', 
                            freqs=None, n_cycles=None, output_dir='./ersp_results',
                            save_for_group_analysis=False,  # ← 新增參數
                            group_data_dir=r'C:\Experiment\Result\h5'):  # h5 儲存路徑
    """
    完整的 ASRT ERSP 分析（階層式）
    
    根據 phase 不同，metadata 結構不同：
    
    Learning 階段：
      - trial_type: 'Regular' / 'Random'
      - 無 task_type 區分
    
    Testing 階段：
      - test_type: 'motor' / 'perceptual'
      - trial_type: 'Regular' / 'Random'
      - 分析：test_type × trial_type (2×2)
    
    Parameters
    ----------
    epochs : mne.Epochs
        完整的 epochs 資料（必須包含 metadata）
    subject_id : str
        受試者 ID
    phase : str
        'learning' 或 'testing'
    lock_type : str
        'stimulus' 或 'response'
    freqs : ndarray or None
        頻率陣列
    n_cycles : ndarray or None
        Wavelet cycles
    output_dir : str
        輸出目錄
    save_for_group_analysis : bool
        是否儲存供群體分析用
    group_data_dir : str
        群體分析資料儲存目錄
        
    Returns
    -------
    results : dict
        完整的分析結果
    """
    
    print(f"\n{'='*60}")
    print(f"完整 ASRT ERSP 分析")
    print(f"{'='*60}")
    print(f"受試者: {subject_id}")
    print(f"Phase: {phase.upper()}")
    print(f"Lock Type: {lock_type.upper()}")
    print(f"總 Epochs: {len(epochs)}")
    
    # === 1. 檢查 metadata ===
    if epochs.metadata is None:
        raise ValueError("Epochs 必須包含 metadata")
    
    # === 2. 設定頻率參數 ===
    if freqs is None:
        freqs = np.logspace(np.log10(4), np.log10(30), num=50)  # 4-30 Hz (避免低頻 wavelet 太長)
    if n_cycles is None:
        n_cycles = freqs / 2.0  # 頻率相依 (4Hz用2 cycles, 30Hz用15 cycles)
    
    # === 3. 根據 phase 分析 ===
    if phase.lower() == 'learning':
        results = _ersp_learning_phase(
            epochs, subject_id, lock_type, freqs, n_cycles, output_dir
        )
    elif phase.lower() == 'testing':
        results = _ersp_testing_phase(
            epochs, subject_id, lock_type, freqs, n_cycles, output_dir
        )
    else:
        raise ValueError(f"Unknown phase: {phase}. Use 'learning' or 'testing'")
    
    # ============================================================
    # 儲存個別受試者資料供群體分析用
    # ============================================================
    if save_for_group_analysis:
        from mne_python_analysis.group_ersp_analysis import save_subject_ersp

        print(f"\n{'─'*60}")
        print(f"儲存個別受試者 ERSP 資料供群體分析用...")
        print(f"{'─'*60}")

        saved_count = 0

        if phase.lower() == 'learning':
            # 新結構：results[group_label][trial_type] = power_dict
            for group_label, group_results in results.items():
                for trial_type in ['Regular', 'Random']:
                    if trial_type not in group_results:
                        continue
                    power_dict = group_results[trial_type]
                    if power_dict is None:
                        continue
                    for roi_name, roi_data in power_dict.items():
                        try:
                            condition_name = f"{trial_type}_{group_label}"
                            save_subject_ersp(
                                ersp_data=roi_data['power'],
                                subject_id=subject_id,
                                condition=condition_name,
                                phase=phase,
                                lock_type=lock_type,
                                freqs=roi_data['freqs'],
                                times=roi_data['times'],
                                roi_name=roi_name.lower(),
                                output_dir=group_data_dir
                            )
                            saved_count += 1
                        except Exception as e:
                            print(f"⚠️  儲存 {group_label} {trial_type} {roi_name} 時發生錯誤: {str(e)}")

        elif phase.lower() == 'testing':
            # 新結構：results[group_label][test_type][trial_type] = power_dict
            for group_label, group_results in results.items():
                for test_type in ['motor', 'perceptual']:
                    if test_type not in group_results:
                        continue
                    for trial_type in ['Regular', 'Random']:
                        if trial_type not in group_results[test_type]:
                            continue
                        power_dict = group_results[test_type][trial_type]
                        if power_dict is None:
                            continue
                        for roi_name, roi_data in power_dict.items():
                            try:
                                condition_name = f"{test_type}_{trial_type}_{group_label}"
                                save_subject_ersp(
                                    ersp_data=roi_data['power'],
                                    subject_id=subject_id,
                                    condition=condition_name,
                                    phase=phase,
                                    lock_type=lock_type,
                                    freqs=roi_data['freqs'],
                                    times=roi_data['times'],
                                    roi_name=roi_name.lower(),
                                    output_dir=group_data_dir
                                )
                                saved_count += 1
                            except Exception as e:
                                print(f"⚠️  儲存 {group_label} {test_type} {trial_type} {roi_name} 時發生錯誤: {str(e)}")

        print(f"{'─'*60}")
        print(f"✓ 已儲存 {saved_count} 個 ERSP 檔案至: {group_data_dir}")
        print(f"{'─'*60}")

    
    print(f"\n{'='*60}")
    print(f"✓ 完整分析完成")
    print(f"{'='*60}")
    
    return results

def _ersp_learning_phase(epochs, subject_id, lock_type, freqs, n_cycles, output_dir):
    """
    Learning 階段 ERSP 分析

    每 5 個 block 一組（Block 7-11, 12-16, 17-21, 22-26），
    各組分別計算 Regular vs Random 比較圖。
    """
    print(f"\n{'─'*60}")
    print(f"Learning 階段分析（每 5 個 block 一組）")
    print(f"{'─'*60}")

    if 'trial_type' not in epochs.metadata.columns:
        raise ValueError("Learning 階段需要 'trial_type' metadata")

    # 若有 phase 欄位，只取 Learning phase 的 trials
    if 'phase' in epochs.metadata.columns:
        phase_mask = epochs.metadata['phase'] == 'Learning'
        epochs = epochs[phase_mask]
        print(f"  → 限定 Learning phase：{len(epochs)} trials")

    LEARNING_GROUPS = [(7, 11), (12, 16), (17, 21), (22, 26)]
    has_block = 'block' in epochs.metadata.columns

    results = {}

    if has_block:
        for (blk_start, blk_end) in LEARNING_GROUPS:
            group_label = f"Block{blk_start}-{blk_end}"
            blk_mask = (
                (epochs.metadata['block'] >= blk_start) &
                (epochs.metadata['block'] <= blk_end)
            )
            epochs_group = epochs[blk_mask]

            if len(epochs_group) == 0:
                print(f"\n  ⚠️  {group_label}: 沒有資料，跳過")
                continue

            print(f"\n  {'─'*50}")
            print(f"  {group_label}（{len(epochs_group)} trials）")
            print(f"  {'─'*50}")

            group_results = {}
            for trial_type in ['Regular', 'Random']:
                mask = epochs_group.metadata['trial_type'] == trial_type
                epochs_subset = epochs_group[mask]

                if len(epochs_subset) == 0:
                    print(f"    ⚠️  {trial_type}: 沒有資料")
                    continue

                print(f"    {trial_type}: {len(epochs_subset)} trials")

                power_dict = asrt_ersp_analysis(
                    epochs_subset,
                    subject_id=f"{subject_id}_{lock_type}_learning_{group_label}_{trial_type}",
                    freqs=freqs,
                    n_cycles=n_cycles,
                    output_dir=output_dir
                )
                group_results[trial_type] = power_dict

            results[group_label] = group_results

            if 'Regular' in group_results and 'Random' in group_results:
                plot_learning_comparison(
                    group_results, subject_id, lock_type, output_dir,
                    block_label=group_label
                )
    else:
        # 沒有 block 欄位時，fallback 為整體分析
        print(f"  ⚠️  metadata 無 'block' 欄位，改為整體分析")
        for trial_type in ['Regular', 'Random']:
            mask = epochs.metadata['trial_type'] == trial_type
            epochs_subset = epochs[mask]
            if len(epochs_subset) == 0:
                continue
            print(f"\n  {trial_type}: {len(epochs_subset)} trials")
            power_dict = asrt_ersp_analysis(
                epochs_subset,
                subject_id=f"{subject_id}_{lock_type}_learning_{trial_type}",
                freqs=freqs, n_cycles=n_cycles, output_dir=output_dir
            )
            results[trial_type] = power_dict

        if 'Regular' in results and 'Random' in results:
            plot_learning_comparison(results, subject_id, lock_type, output_dir)

    return results


def _ersp_testing_phase(epochs, subject_id, lock_type, freqs, n_cycles, output_dir):
    """
    Testing 階段 ERSP 分析

    每 2 個 block 一組（Block 27-28, 29-30, 31-32, 33-34），
    各組分別計算 motor/perceptual × Regular vs Random 比較圖。
    """
    print(f"\n{'─'*60}")
    print(f"Testing 階段分析（每 2 個 block 一組）")
    print(f"{'─'*60}")

    required_cols = ['test_type', 'trial_type']
    for col in required_cols:
        if col not in epochs.metadata.columns:
            raise ValueError(f"Testing 階段需要 '{col}' metadata")

    # 若有 phase 欄位，只取 Test phase 的 trials
    if 'phase' in epochs.metadata.columns:
        phase_mask = epochs.metadata['phase'] == 'Test'
        epochs = epochs[phase_mask]
        print(f"  → 限定 Test phase：{len(epochs)} trials")

    TESTING_GROUPS = [(27, 28), (29, 30), (31, 32), (33, 34)]
    has_block = 'block' in epochs.metadata.columns

    results = {}

    if has_block:
        for (blk_start, blk_end) in TESTING_GROUPS:
            group_label = f"Block{blk_start}-{blk_end}"
            blk_mask = (
                (epochs.metadata['block'] >= blk_start) &
                (epochs.metadata['block'] <= blk_end)
            )
            epochs_group = epochs[blk_mask]

            if len(epochs_group) == 0:
                print(f"\n  ⚠️  {group_label}: 沒有資料，跳過")
                continue

            print(f"\n  {'─'*50}")
            print(f"  {group_label}（{len(epochs_group)} trials）")
            print(f"  {'─'*50}")

            results[group_label] = {}

            for test_type in ['motor', 'perceptual']:
                results[group_label][test_type] = {}

                print(f"\n    {test_type.upper()}")

                test_results = {}
                for trial_type in ['Regular', 'Random']:
                    mask = (
                        (epochs_group.metadata['test_type'].str.lower() == test_type.lower()) &
                        (epochs_group.metadata['trial_type'] == trial_type)
                    )
                    epochs_subset = epochs_group[mask]

                    if len(epochs_subset) == 0:
                        print(f"      ⚠️  {trial_type}: 沒有資料")
                        continue

                    print(f"      {trial_type}: {len(epochs_subset)} trials")

                    power_dict = asrt_ersp_analysis(
                        epochs_subset,
                        subject_id=f"{subject_id}_{lock_type}_testing_{test_type}_{group_label}_{trial_type}",
                        freqs=freqs,
                        n_cycles=n_cycles,
                        output_dir=output_dir
                    )
                    test_results[trial_type] = power_dict
                    results[group_label][test_type][trial_type] = power_dict

                if 'Regular' in test_results and 'Random' in test_results:
                    plot_testing_comparison(
                        test_results,
                        subject_id,
                        lock_type,
                        test_type,
                        output_dir,
                        block_label=group_label
                    )
    else:
        # 沒有 block 欄位時，fallback 為整體分析
        print(f"  ⚠️  metadata 無 'block' 欄位，改為整體分析")
        for test_type in ['motor', 'perceptual']:
            results[test_type] = {}
            print(f"\n  {test_type.upper()}")
            for trial_type in ['Regular', 'Random']:
                mask = (
                    (epochs.metadata['test_type'].str.lower() == test_type.lower()) &
                    (epochs.metadata['trial_type'] == trial_type)
                )
                epochs_subset = epochs[mask]
                if len(epochs_subset) == 0:
                    continue
                print(f"    {trial_type}: {len(epochs_subset)} trials")
                power_dict = asrt_ersp_analysis(
                    epochs_subset,
                    subject_id=f"{subject_id}_{lock_type}_testing_{test_type}_{trial_type}",
                    freqs=freqs, n_cycles=n_cycles, output_dir=output_dir
                )
                results[test_type][trial_type] = power_dict

            if 'Regular' in results[test_type] and 'Random' in results[test_type]:
                plot_testing_comparison(
                    results[test_type], subject_id, lock_type, test_type, output_dir
                )

    return results



    
    
# ============================================================
# 新增函數：Stimulus → Response 對齊 + 整段平均 baseline
# ============================================================

def stimulus_to_response_with_full_epoch_baseline(
    epochs_stim,
    epochs_resp,
    freqs,
    n_cycles,
    target_times=None,
    n_jobs=1
):
    """
    Time domain: Stimulus → Response 對齊
    Frequency domain: 整段 epoch 平均做 baseline
    
    這個函數結合兩種方法：
    1. 時間對齊：使用固定的 Stimulus 時刻做參考，然後根據 RT 重新對齊到 Response
    2. Baseline：使用整段 Stimulus epoch 的平均功率做標準化（Lum 2023 方法）
    
    Parameters
    ----------
    epochs_stim : mne.Epochs
        Stimulus-locked epochs (建議 -0.8 to 1.0 s)
        用於計算 ERSP 和提供固定的時間參考點
    epochs_resp : mne.Epochs
        Response-locked epochs (建議 -1.1 to 0.5 s)
        只用來取得每個 trial 的 RT 資訊和建立最終的 info
    freqs : array
        頻率範圍 (例如 np.arange(4, 30, 1))
    n_cycles : array or float
        Wavelet cycles (例如 freqs / 2.0)
    target_times : array, optional
        Response-locked 目標時間軸
        如果為 None，則自動生成 -1.1 to 0.5 s
    n_jobs : int
        平行運算核心數 (建議使用 1 避免 GIL 問題)
        
    Returns
    -------
    power_resp : mne.time_frequency.AverageTFR
        Response-locked ERSP
        - 時間軸: 相對於 Response (0 = response onset)
        - Baseline: 整段 Stimulus epoch 平均
        - 單位: dB (10*log10(power/baseline))
        
    Notes
    -----
    此方法的優點：
    1. 時間對齊一致：所有 trials 使用相同的 Stimulus 參考點
    2. Baseline 標準化：使用整段 epoch 平均，不依賴特定時間窗口
    3. 避免 RT 變異影響：固定的 Stimulus baseline 不受 RT 變化影響
    
    Example
    -------
    >>> freqs = np.arange(4, 30, 1)
    >>> n_cycles = freqs / 2.0
    >>> power_resp = stimulus_to_response_with_full_epoch_baseline(
    ...     epochs_stim, epochs_resp, freqs, n_cycles, n_jobs=1
    ... )
    >>> power_resp.plot_topo(baseline=None, mode='logratio')
    """
    
    # ========== 1. 計算 Stimulus-locked ERSP（保留每個 trial） ==========
    print("\n" + "="*70)
    print("Step 1: 計算 Stimulus-locked ERSP (保留每個 trial)...")
    print("="*70)
    
    power_stim = mne.time_frequency.tfr_morlet(
        epochs_stim,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        average=False,  # 重要：保留每個 trial
        n_jobs=n_jobs,
        decim=1,
        verbose=True
    )
    
    print(f"  ✓ ERSP shape: {power_stim.data.shape}")
    print(f"    (trials, channels, freqs, times) = "
        f"({power_stim.data.shape[0]}, {power_stim.data.shape[1]}, "
        f"{power_stim.data.shape[2]}, {power_stim.data.shape[3]})")
    
    # ========== 2. Baseline correction: 整段 epoch 平均 ==========
    print("\n" + "="*70)
    print("Step 2: Baseline correction (整段 Stimulus epoch 平均)...")
    print("="*70)
    
    # power_stim.data shape: (n_trials, n_channels, n_freqs, n_times)
    baseline_power = power_stim.data.mean(axis=-1, keepdims=True)
    # baseline_power shape: (n_trials, n_channels, n_freqs, 1)
    
    # dB 轉換: 10*log10(power/baseline)
    power_stim.data = 10 * np.log10(power_stim.data / baseline_power)
    
    print(f"  ✓ Baseline 時間範圍: {power_stim.times[0]:.3f} to {power_stim.times[-1]:.3f} s")
    print(f"  ✓ Baseline 方法: 整段 epoch 平均 (Lum 2023)")
    print(f"  ✓ 單位: dB (10*log10(power/baseline))")
    
    # ========== 3. 取得每個 trial 的 RT ==========
    print("\n" + "="*70)
    print("Step 3: 取得每個 trial 的 RT...")
    print("="*70)
    
    if hasattr(epochs_resp, 'metadata') and epochs_resp.metadata is not None:
        if 'rt' in epochs_resp.metadata.columns:
            RTs = epochs_resp.metadata['rt'].values
            print(f"  ✓ 從 metadata 取得 RT")
            print(f"  ✓ RT 範圍: {RTs.min()*1000:.1f} - {RTs.max()*1000:.1f} ms")
            print(f"  ✓ RT 平均: {RTs.mean()*1000:.1f} ms")
        else:
            raise ValueError("epochs_resp.metadata 中找不到 'rt' 欄位")
    else:
        raise NotImplementedError(
            "請在 epochs_resp.metadata 中提供 RT 資訊\n"
            "建議在建立 response-locked epochs 時加入 metadata:\n"
            "  metadata = pd.DataFrame({'rt': RTs})\n"
            "  epochs_resp = mne.Epochs(..., metadata=metadata)"
        )
    
    # 檢查 trials 數量是否一致
    if len(RTs) != power_stim.data.shape[0]:
        raise ValueError(
            f"RT 數量 ({len(RTs)}) 與 Stimulus epochs 數量 "
            f"({power_stim.data.shape[0]}) 不一致"
        )
    
    # ========== 4. 定義 Response-locked 目標時間軸 ==========
    print("\n" + "="*70)
    print("Step 4: 定義 Response-locked 目標時間軸...")
    print("="*70)
    
    if target_times is None:
        # 預設: -1.1 to 0.5 s, 使用 Stimulus epochs 的採樣率
        target_times = np.arange(-1.1, 0.5, 1/epochs_stim.info['sfreq'])
    
    print(f"  ✓ Response-locked 時間範圍: {target_times[0]:.3f} to {target_times[-1]:.3f} s")
    print(f"  ✓ 時間點數: {len(target_times)}")
    print(f"  ✓ 採樣率: {epochs_stim.info['sfreq']} Hz")
    
    # ========== 5. 重新對齊到 Response-locked 時間軸 ==========
    print("\n" + "="*70)
    print("Step 5: 重新對齊到 Response-locked 時間軸...")
    print("="*70)
    
    times_stim = power_stim.times
    n_trials, n_channels, n_freqs, n_times_stim = power_stim.data.shape
    n_times_resp = len(target_times)
    
    print(f"  處理 {n_trials} trials × {n_channels} channels × {n_freqs} freqs...")
    
    # 初始化 Response-locked power array
    power_resp_data = np.full(
        (n_trials, n_channels, n_freqs, n_times_resp),
        np.nan
    )
    
    # 統計有效 trials
    valid_trials = 0
    partial_trials = 0
    invalid_trials = 0
    
    # 逐 trial 插值
    for trial_idx in range(n_trials):
        RT = RTs[trial_idx]
        
        # Response-locked 時間對應到 Stimulus-locked 時間
        # Response 0s = Stimulus RT
        # Response -0.5s = Stimulus (RT - 0.5)
        # Response +0.3s = Stimulus (RT + 0.3)
        times_resp_in_stim = target_times + RT
        
        # 檢查有效範圍
        valid_mask = (times_resp_in_stim >= times_stim[0]) & \
                    (times_resp_in_stim <= times_stim[-1])
        
        n_valid = np.sum(valid_mask)
        
        if n_valid == 0:
            invalid_trials += 1
            if invalid_trials <= 3:  # 只顯示前 3 個警告
                print(f"  ⚠ Trial {trial_idx}: RT={RT*1000:.1f}ms, 無可用時間點")
            continue
        elif n_valid < len(target_times):
            partial_trials += 1
        else:
            valid_trials += 1
        
        # 對每個 channel 和 frequency 插值
        for ch_idx in range(n_channels):
            for freq_idx in range(n_freqs):
                # 取得這個 trial/channel/freq 的 power
                power_stim_1d = power_stim.data[trial_idx, ch_idx, freq_idx, :]
                
                # 建立插值函數（線性插值）
                interp_func = interp1d(
                    times_stim,
                    power_stim_1d,
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
                
                # 插值到 Response-locked 時間點
                power_resp_data[trial_idx, ch_idx, freq_idx, :] = \
                    interp_func(times_resp_in_stim)
    
    print(f"\n  ✓ 完整有效 trials: {valid_trials}/{n_trials}")
    print(f"  ⚠ 部分有效 trials: {partial_trials}/{n_trials}")
    if invalid_trials > 0:
        print(f"  ✗ 無效 trials: {invalid_trials}/{n_trials}")
    
    # ========== 6. 平均所有 trials（忽略 NaN） ==========
    print("\n" + "="*70)
    print("Step 6: 平均所有 trials...")
    print("="*70)
    
    power_resp_avg = np.nanmean(power_resp_data, axis=0)
    
    # 檢查是否有時間點完全沒有資料
    nan_counts = np.isnan(power_resp_avg).sum(axis=(0, 1))  # 每個時間點的 NaN 數量
    if np.any(nan_counts > 0):
        time_with_nans = target_times[nan_counts > 0]
        print(f"  ⚠ 警告: 某些時間點有缺失資料")
        print(f"    時間範圍: {time_with_nans[0]:.3f} to {time_with_nans[-1]:.3f} s")
    else:
        print(f"  ✓ 所有時間點都有完整資料")
    
    # ========== 7. 建立 AverageTFR 物件 ==========
    print("\n" + "="*70)
    print("Step 7: 建立 AverageTFR 物件...")
    print("="*70)
    
    power_resp = mne.time_frequency.AverageTFR(
        info=epochs_resp.info,
        data=power_resp_avg,
        times=target_times,
        freqs=freqs,
        nave=n_trials,
        comment='Response-locked (Stimulus time + full-epoch baseline)',
        method='morlet-wavelet'
    )
    
    # ========== 完成 ==========
    print("\n" + "="*70)
    print("✓ 完成！Response-locked ERSP")
    print("="*70)
    print(f"  時間範圍: {target_times[0]:.3f} to {target_times[-1]:.3f} s (相對於 Response)")
    print(f"  頻率範圍: {freqs[0]:.1f} to {freqs[-1]:.1f} Hz")
    print(f"  Baseline 方法: 整段 Stimulus epoch (-0.8 to 1.0 s) 平均")
    print(f"  有效 trials: {valid_trials + partial_trials}/{n_trials}")
    print("="*70 + "\n")
    
    return power_resp