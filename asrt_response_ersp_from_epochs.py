"""
ASRT Response ERSP 分析（從當前 epochs）

支援使用已經處理過的 epochs（包含極端值排除和 RT metadata）
"""

import os
import numpy as np
import mne
from scipy import interpolate

# ROI 定義（與 asrt/ersp.py 一致）
_ROI_GROUPS = {
    'Theta': ['Fz', 'FCz', 'Cz', 'C3', 'C4'],
    'Alpha': ['O1', 'Oz', 'O2', 'P3', 'Pz', 'P4'],
}


def _tfr_to_power_dict(tfr):
    """
    將 MNE AverageTFR 物件轉成 ersp_plots 函數所需的 power_dict 格式。

    Returns
    -------
    power_dict : dict
        {'Theta': {'power': ndarray(n_freqs, n_times), 'times': ..., 'freqs': ..., 'channels': [...]},
         'Alpha': {...}}
    """
    power_dict = {}
    for roi_name, channels in _ROI_GROUPS.items():
        available = [ch for ch in channels if ch in tfr.ch_names]
        if not available:
            continue
        ch_idx = [tfr.ch_names.index(ch) for ch in available]
        # AverageTFR.data shape: (n_ch, n_freqs, n_times)
        roi_power = tfr.data[ch_idx].mean(axis=0)
        power_dict[roi_name] = {
            'power':    roi_power,
            'times':    tfr.times,
            'freqs':    tfr.freqs,
            'channels': available,
        }
    return power_dict


def _run_alignment_and_ersp(
    epochs_stim_subset,
    response_epochs_subset,
    rt_values_subset,
    response_times,
    event_id,
    freqs,
    n_cycles,
    decim,
    n_jobs,
    label='',
):
    """
    步驟 3-4：對一組 matched epochs 執行 RT 對齊 + ERSP 計算。

    Parameters
    ----------
    epochs_stim_subset : mne.Epochs
        對應此條件的 Stimulus epochs（已排序配對）
    response_epochs_subset : mne.Epochs
        對應此條件的 Response epochs
    rt_values_subset : ndarray
        對應此條件的 RT 值（秒）
    response_times : ndarray
        Response-locked 目標時間軸
    event_id : dict
    freqs, n_cycles : ndarray
    decim, n_jobs : int
    label : str
        條件名稱（僅用於 print）

    Returns
    -------
    power : mne.time_frequency.AverageTFR
    """
    prefix = f"  [{label}] " if label else "  "
    n_epochs   = len(epochs_stim_subset)
    stim_data  = epochs_stim_subset.get_data()        # (n, ch, t)
    n_channels = stim_data.shape[1]
    n_times    = len(response_times)
    stim_times = epochs_stim_subset.times

    print(f"{prefix}對齊 {n_epochs} 個 epochs...")
    aligned_data = np.zeros((n_epochs, n_channels, n_times))

    for i in range(n_epochs):
        rt = rt_values_subset[i]
        for ch in range(n_channels):
            stim_t_needed = response_times + rt
            valid_mask = (
                (stim_t_needed >= stim_times[0]) &
                (stim_t_needed <= stim_times[-1])
            )
            if np.any(valid_mask):
                f = interpolate.interp1d(
                    stim_times, stim_data[i, ch, :],
                    kind='linear', bounds_error=False, fill_value=0.0
                )
                aligned_data[i, ch, valid_mask] = f(stim_t_needed[valid_mask])

    print(f"{prefix}✓ 對齊完成")

    info = response_epochs_subset.info.copy()
    # 只保留子集裡實際出現的 event codes，避免 EpochsArray 找不到對應 id 報錯
    present_codes = set(response_epochs_subset.events[:, 2])
    event_id_sub  = {k: v for k, v in event_id.items() if v in present_codes}
    aligned_epochs = mne.EpochsArray(
        aligned_data,
        info,
        tmin=response_times[0],
        events=response_epochs_subset.events,
        event_id=event_id_sub,
        metadata=response_epochs_subset.metadata,
    )

    print(f"{prefix}計算 ERSP（整段 epoch 平均 baseline）...")
    averaged_epochs = aligned_epochs.average()
    power = mne.time_frequency.tfr_morlet(
        averaged_epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        decim=decim,
        n_jobs=n_jobs,
        average=False,
    )
    power.apply_baseline(mode='logratio', baseline=(-1.0, -0.6))
    print(f"{prefix}✓ ERSP 計算完成")

    return power


def response_ersp_from_current_epochs(
    current_raw,
    response_epochs,
    freqs=np.arange(4, 40, 1),
    n_cycles_func=None,
    decim=1,
    n_jobs=1,
    average=True,
    output_dir=r'C:\Experiment\Result\h5',
    subject_id=None
):
    """
    從當前的 Response epochs 執行 Stimulus-to-Response ERSP 分析

    流程：
    1. 從 Response epochs metadata 得知保留的 trials
    2. 重新從 Raw 建立對應的 Stimulus epochs（Stimulus baseline correction）
    3. 根據 RT 對齊到 Response 時間軸
    4. 計算 ERSP（整段平均 baseline）

    若 metadata 包含 'test_type' + 'phase' 欄位，自動分為 3 個條件：
      - Learning
      - MotorTest  (phase=='Test' & test_type=='motor')
      - PerceptualTest (phase=='Test' & test_type=='perceptual')
    各條件分別儲存：
      {subject_id}_Response_Learning_ERSP.h5
      {subject_id}_Response_MotorTest_ERSP.h5
      {subject_id}_Response_PerceptualTest_ERSP.h5

    Parameters
    ----------
    current_raw : mne.io.Raw
    response_epochs : mne.Epochs
    freqs : array
    n_cycles_func : callable or None
    decim : int
    n_jobs : int
    average : bool
    output_dir : str
    subject_id : str

    Returns
    -------
    dict of AverageTFR  (多條件時)  或  AverageTFR  (單一條件時)
    """

    print("\n" + "="*70)
    print("Response ERSP 分析（從當前 epochs）")
    print("="*70)

    # ===== 步驟 1：檢查 RT metadata =====
    print("\n步驟 1：檢查 metadata")

    if not hasattr(response_epochs, 'metadata') or response_epochs.metadata is None:
        raise ValueError("Response epochs 沒有 metadata！")

    if 'rt' not in response_epochs.metadata.columns:
        raise ValueError("Metadata 中沒有 'rt' 欄位！請先執行選項 20")

    rt_values = response_epochs.metadata['rt'].values
    valid_rt  = ~np.isnan(rt_values)

    if not np.all(valid_rt):
        print(f"  ⚠️  有 {np.sum(~valid_rt)} 個 epochs 缺少 RT，排除後繼續")
        response_epochs = response_epochs[valid_rt]
        rt_values = response_epochs.metadata['rt'].values

    print(f"  Epochs 數量: {len(response_epochs)}")
    print(f"  RT 範圍: {rt_values.min()*1000:.1f} - {rt_values.max()*1000:.1f} ms")
    print(f"  RT 平均: {rt_values.mean()*1000:.1f} ± {rt_values.std()*1000:.1f} ms")

    # ===== 步驟 2：從 Raw 重新建立 Stimulus epochs =====
    print("\n步驟 2：重新建立 Stimulus epochs（Stimulus baseline correction）")

    if current_raw.annotations:
        print(f"  使用 annotations 建立 events")
        all_events, event_dict = mne.events_from_annotations(current_raw)
        print(f"  找到 {len(all_events)} 個 events")
    else:
        stim_channels = mne.pick_types(current_raw.info, stim=True, exclude=[])
        if len(stim_channels) > 0:
            stim_channel = current_raw.ch_names[stim_channels[0]]
        elif 'Trigger' in current_raw.ch_names:
            stim_channel = 'Trigger'
        elif 'STI 014' in current_raw.ch_names:
            stim_channel = 'STI 014'
        else:
            raise ValueError("找不到 stim channel")

        print(f"  使用 stim channel: {stim_channel}")
        all_events = mne.find_events(
            current_raw, stim_channel=stim_channel, min_duration=0.002
        )
        print(f"  找到 {len(all_events)} 個 events")

    STIMULUS_CODES = {
        'Random':  [41, 42, 43, 44],
        'Regular': [46, 47, 48, 49],
    }

    if current_raw.annotations:
        stim_codes_new = []
        for code in STIMULUS_CODES['Random'] + STIMULUS_CODES['Regular']:
            str_code = str(code)
            if str_code in event_dict:
                stim_codes_new.append(event_dict[str_code])
        stimulus_events = all_events[np.isin(all_events[:, 2], stim_codes_new)]
    else:
        stim_codes    = STIMULUS_CODES['Random'] + STIMULUS_CODES['Regular']
        stimulus_events = all_events[np.isin(all_events[:, 2], stim_codes)]

    print(f"  找到 {len(stimulus_events)} 個 Stimulus events")

    event_id = {'Random': 1, 'Regular': 2}
    new_events = stimulus_events.copy()

    for i, event in enumerate(new_events):
        if current_raw.annotations:
            orig_code = None
            for desc, code in event_dict.items():
                if code == event[2]:
                    orig_code = int(desc)
                    break
            if orig_code in STIMULUS_CODES['Random']:
                new_events[i, 2] = 1
            elif orig_code in STIMULUS_CODES['Regular']:
                new_events[i, 2] = 2
        else:
            if event[2] in STIMULUS_CODES['Random']:
                new_events[i, 2] = 1
            elif event[2] in STIMULUS_CODES['Regular']:
                new_events[i, 2] = 2

    print(f"\n  建立 Stimulus epochs...")
    print(f"    時間窗口: -0.8 to 1.0 s")
    print(f"    Baseline: -0.5 to -0.1 s")

    epochs_stim = mne.Epochs(
        current_raw,
        new_events,
        event_id=event_id,
        tmin=-0.8,
        tmax=1.0,
        baseline=(-0.5, -0.1),
        preload=True,
        reject=None,
        proj=False,
    )
    print(f"  ✓ 建立 {len(epochs_stim)} 個 Stimulus epochs")

    # 對應 Stimulus ↔ Response epochs
    print(f"\n  對應 Stimulus 和 Response epochs...")
    response_event_samples = response_epochs.events[:, 0]
    matched_indices = []

    for resp_sample in response_event_samples:
        stim_samples = new_events[:, 0]
        before_resp  = stim_samples < resp_sample
        if np.any(before_resp):
            matched_indices.append(np.where(before_resp)[0][-1])
        else:
            print(f"    ⚠️  找不到 Response sample {resp_sample} 對應的 Stimulus")
            matched_indices.append(-1)

    valid_matches = np.array(matched_indices) >= 0
    if not np.all(valid_matches):
        print(f"    ⚠️  {np.sum(~valid_matches)} 個 Response 找不到對應的 Stimulus，排除")
        response_epochs  = response_epochs[valid_matches]
        rt_values        = rt_values[valid_matches]
        matched_indices  = [idx for idx, ok in zip(matched_indices, valid_matches) if ok]

    epochs_stim = epochs_stim[matched_indices]
    print(f"  ✓ 成功對應 {len(epochs_stim)} 個 epochs")

    # 共用的 Response 時間軸
    response_times = response_epochs.times

    # n_cycles
    if n_cycles_func is None:
        n_cycles = freqs / 2.0
    else:
        n_cycles = n_cycles_func(freqs)

    print(f"\n  頻率範圍: {freqs[0]}-{freqs[-1]} Hz")
    print(f"  n_cycles 範圍: {n_cycles[0]:.1f}-{n_cycles[-1]:.1f}")

    # ===== 步驟 3-4：依 metadata 決定條件 =====
    meta = response_epochs.metadata
    has_test_type = (
        meta is not None and
        'test_type' in meta.columns and
        'phase' in meta.columns
    )

    # Block grouping definitions
    LEARNING_GROUPS = [(7, 11), (12, 16), (17, 21), (22, 26)]
    TESTING_GROUPS  = [(27, 28), (29, 30), (31, 32), (33, 34)]
    has_block = 'block' in meta.columns

    if has_test_type:
        has_trial_type = 'trial_type' in meta.columns
        trial_types = ['Regular', 'Random'] if has_trial_type else ['All']

        print(f"\n  偵測到 'test_type' 欄位")

        # results 結構：{group_key: {trial_type: TFR}}
        # group_key 例如 "Learning_Block7-11"、"MotorTest_Block27-28"
        results = {}

        def _compute_and_save(cond_name, group_label, base_mask):
            group_results = {}
            for trial_type in trial_types:
                if trial_type == 'All':
                    sub_mask = base_mask
                else:
                    tt_mask  = meta['trial_type'].values == trial_type
                    sub_mask = base_mask & tt_mask

                sub_n = sub_mask.sum()
                if sub_n == 0:
                    print(f"    ⚠  {trial_type}: 沒有資料，跳過")
                    continue

                key = f"{cond_name}_{group_label}"
                print(f"\n  [{key} / {trial_type}]  {sub_n} trials")
                sub_indices = np.where(sub_mask)[0]

                resp_sub = response_epochs[sub_indices]
                stim_sub = epochs_stim[sub_indices]
                rt_sub   = rt_values[sub_mask]

                power = _run_alignment_and_ersp(
                    stim_sub, resp_sub, rt_sub,
                    response_times, event_id,
                    freqs, n_cycles, decim, n_jobs,
                    label=f"{key}/{trial_type}",
                )

                if output_dir and subject_id:
                    h5_dir = os.path.join(output_dir, 'h5')
                    os.makedirs(h5_dir, exist_ok=True)
                    default_fname = f"{subject_id}_Response_{cond_name}_{group_label}_{trial_type}_ERSP.h5"
                    h5_fname = input(f"\n請輸入 ERSP 檔名 [預設: {default_fname}]: ").strip() or default_fname
                    if not h5_fname.endswith('.h5'):
                        h5_fname += '.h5'
                    out_path = os.path.join(h5_dir, h5_fname)
                    power.save(out_path, overwrite=True)
                    print(f"  ✓ 已儲存: {out_path}")

                group_results[trial_type] = power
            return group_results

        # ── Learning ──
        learning_base = meta['phase'].values == 'Learning'
        groups = LEARNING_GROUPS if has_block else [None]
        for grp in groups:
            if grp is not None:
                blk_mask    = (meta['block'].values >= grp[0]) & (meta['block'].values <= grp[1])
                base_mask   = learning_base & blk_mask
                group_label = f"Block{grp[0]}-{grp[1]}"
            else:
                base_mask   = learning_base
                group_label = "All"
            if base_mask.sum() == 0:
                print(f"\n  ⚠  Learning {group_label}: 沒有資料，跳過")
                continue
            print(f"\n{'─'*60}")
            print(f"  Learning {group_label}  ({base_mask.sum()} trials)")
            print(f"{'─'*60}")
            results[f"Learning_{group_label}"] = _compute_and_save('Learning', group_label, base_mask)

        # ── Motor Test / Perceptual Test ──
        for test_type, cond_name in [('motor', 'MotorTest'), ('perceptual', 'PerceptualTest')]:
            test_base = (meta['phase'].values == 'Test') & (meta['test_type'].values == test_type)
            groups = TESTING_GROUPS if has_block else [None]
            for grp in groups:
                if grp is not None:
                    blk_mask    = (meta['block'].values >= grp[0]) & (meta['block'].values <= grp[1])
                    base_mask   = test_base & blk_mask
                    group_label = f"Block{grp[0]}-{grp[1]}"
                else:
                    base_mask   = test_base
                    group_label = "All"
                if base_mask.sum() == 0:
                    print(f"\n  ⚠  {cond_name} {group_label}: 沒有資料，跳過")
                    continue
                print(f"\n{'─'*60}")
                print(f"  {cond_name} {group_label}  ({base_mask.sum()} trials)")
                print(f"{'─'*60}")
                results[f"{cond_name}_{group_label}"] = _compute_and_save(cond_name, group_label, base_mask)

        total = sum(len(v) for v in results.values())
        print(f"\n{'='*70}")
        print(f"✓ Response ERSP 完成，共 {total} 個子條件")
        print(f"{'='*70}")

        # ── 繪圖 ──────────────────────────────────────────────────
        if output_dir and subject_id:
            try:
                from asrt.ersp_plots import (
                    plot_ersp_lum2023_style,
                    plot_learning_comparison,
                    plot_testing_comparison,
                )

                print(f"\n{'─'*70}")
                print(f"繪製 Response ERSP 圖片...")
                print(f"{'─'*70}")

                for key, trial_results in results.items():
                    power_dicts = {}

                    for trial_type, tfr in trial_results.items():
                        pd_ = _tfr_to_power_dict(tfr)
                        if not pd_:
                            print(f"  ⚠  {key}/{trial_type}: 找不到可用 ROI，略過")
                            continue
                        power_dicts[trial_type] = pd_

                        # 個別條件圖
                        plot_ersp_lum2023_style(
                            pd_,
                            f"{subject_id}_response_{key}_{trial_type}",
                            output_dir,
                        )

                    # Regular vs Random 比較圖
                    if 'Regular' in power_dicts and 'Random' in power_dicts:
                        # 解析 key: "Learning_Block7-11" / "MotorTest_Block27-28" / ...
                        parts       = key.split('_', 1)
                        cond_name   = parts[0]
                        group_label = parts[1] if len(parts) > 1 else None

                        if cond_name == 'Learning':
                            plot_learning_comparison(
                                {'Regular': power_dicts['Regular'],
                                 'Random':  power_dicts['Random']},
                                subject_id=subject_id,
                                lock_type='response',
                                output_dir=output_dir,
                                block_label=group_label,
                            )
                        elif cond_name == 'MotorTest':
                            plot_testing_comparison(
                                {'Regular': power_dicts['Regular'],
                                 'Random':  power_dicts['Random']},
                                subject_id=subject_id,
                                lock_type='response',
                                test_type='motor',
                                output_dir=output_dir,
                                block_label=group_label,
                            )
                        elif cond_name == 'PerceptualTest':
                            plot_testing_comparison(
                                {'Regular': power_dicts['Regular'],
                                 'Random':  power_dicts['Random']},
                                subject_id=subject_id,
                                lock_type='response',
                                test_type='perceptual',
                                output_dir=output_dir,
                                block_label=group_label,
                            )

            except Exception as _plot_err:
                print(f"  ⚠  繪圖時發生錯誤（不影響 .h5 資料）: {_plot_err}")
                import traceback; traceback.print_exc()

        return results


    else:
        # 原始單一條件路徑
        print("\n步驟 3-4：RT 對齊 + ERSP（全部 epochs）")
        power = _run_alignment_and_ersp(
            epochs_stim, response_epochs, rt_values,
            response_times, event_id,
            freqs, n_cycles, decim, n_jobs,
        )

        if output_dir and subject_id:
            h5_dir = os.path.join(output_dir, 'h5')
            os.makedirs(h5_dir, exist_ok=True)
            default_fname = f"{subject_id}_response_ersp_from_epochs-tfr.h5"
            h5_fname = input(f"\n請輸入 TFR 檔名 [預設: {default_fname}]: ").strip() or default_fname
            if not h5_fname.endswith('.h5'):
                h5_fname += '.h5'
            out_path = os.path.join(h5_dir, h5_fname)
            power.save(out_path, overwrite=True)
            print(f"\n✓ 已儲存: {out_path}")

        print("\n" + "="*70)
        print("Response ERSP 分析完成！")
        print("="*70)
        return power
