"""
ASRT Response ERSP 分析（從當前 epochs）

支援使用已經處理過的 epochs（包含極端值排除和 RT metadata）
"""

import os
import numpy as np
import mne

# ROI 定義（與 asrt/ersp.py 一致）
_ROI_GROUPS = {
    'Motor':                ['Fz', 'FCz', 'Cz', 'C3', 'C4'],
    'Motor_Frontal':        ['Fz', 'FCz'],
    'Motor_Central':        ['Cz', 'C3', 'C4'],
    'Motor_Parietal':       ['P3', 'Pz', 'P4'],
    'Motor_Occipital':      ['O1', 'Oz', 'O2'],
    'Perceptual':           ['O1', 'Oz', 'O2', 'P3', 'Pz', 'P4'],
    'Perceptual_Parietal':  ['P3', 'Pz', 'P4'],
    'Perceptual_Occipital': ['O1', 'Oz', 'O2'],
    'Perceptual_Frontal':   ['Fz', 'FCz'],
    'Perceptual_Central':   ['Cz', 'C3', 'C4'],
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


def _compute_pertrial_ersp(epochs_subset, freqs, n_cycles, decim, n_jobs,
                            label='', do_td_baseline=False, baseline_method='pre_stim'):
    prefix = f"  [{label}] " if label else "  "
    n_epochs = len(epochs_subset)
    sfreq = epochs_subset.info['sfreq']

    # === Mirror padding 設定 ===
    # 最低頻率的 wavelet 有效範圍 = n_cycles / (2π × f_min) × 3σ
    f_min = float(freqs[0])
    n_cyc_min = float(n_cycles[0]) if hasattr(n_cycles, '__len__') else float(n_cycles)
    sigma_t = n_cyc_min / (2.0 * np.pi * f_min)
    pad_duration = max(3.0 * sigma_t, 0.5)  # 至少 500ms
    pad_samples = int(np.ceil(pad_duration * sfreq))

    print(f"{prefix}Mirror padding: {pad_duration*1000:.0f} ms（各頻率 3σ，最低 4Hz 需 ~240ms）")

    # === 建立 padded epochs ===
    data = epochs_subset.get_data()  # (n_epochs, n_ch, n_times)
    padded_data = np.pad(data, ((0, 0), (0, 0), (pad_samples, pad_samples)), mode='edge')

    new_tmin = epochs_subset.tmin - pad_duration
    padded_epochs = mne.EpochsArray(
        padded_data,
        info=epochs_subset.info,
        events=epochs_subset.events,
        tmin=new_tmin,
        metadata=epochs_subset.metadata.reset_index(drop=True) if epochs_subset.metadata is not None else None,
        event_id=epochs_subset.event_id,
        verbose=False,
    )

    print(f"{prefix}計算逐 trial Morlet wavelet（{n_epochs} trials，padded）...")

    tfr_epochs = mne.time_frequency.tfr_morlet(
        padded_epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        decim=decim,
        n_jobs=n_jobs,
        average=False,
        verbose=False,
    )

    # crop 回原始時間範圍
    tfr_epochs.crop(tmin=epochs_subset.tmin, tmax=epochs_subset.tmax)

    times    = tfr_epochs.times
    ep_times = epochs_subset.times

    # === whole_epoch 模式：直接 average 後對整段做 baseline，不進 per-trial 迴圈 ===
    if baseline_method == 'whole_epoch':
        avg_tfr = tfr_epochs.average()
        avg_tfr.apply_baseline(mode='logratio', baseline=(None, None))
        print(f"{prefix}✓ ERSP 完成（whole-epoch baseline，average 後校正）")
        return avg_tfr

    corrected = np.zeros_like(tfr_epochs.data)
    n_short_baseline = 0
    for i in range(n_epochs):
        stim_sample = epochs_subset.metadata['stim_sample'].values[i]
        resp_sample = epochs_subset.events[i, 0]

        if stim_sample < 0:
            rt = 0.5
        else:
            rt = (resp_sample - stim_sample) / sfreq

        bl_start = -rt - 0.5
        bl_end   = -rt - 0.1

        if do_td_baseline:
            td_mask = (ep_times >= bl_start) & (ep_times <= bl_end)
            if td_mask.sum() == 0:
                td_mask = (ep_times >= bl_start - 0.2) & (ep_times <= bl_end)
            td_mean = epochs_subset._data[i][:, td_mask].mean(axis=-1, keepdims=True)
            epochs_subset._data[i] -= td_mean

        # 現有做法：per-trial pre-stimulus blank 期
        bl_mask = (times >= bl_start) & (times <= bl_end)
        if bl_mask.sum() == 0:
            bl_mask = (times >= bl_start - 0.2) & (times <= bl_end)
            n_short_baseline += 1
        bl_mean = tfr_epochs.data[i][:, :, bl_mask].mean(axis=-1, keepdims=True)

        bl_mean = np.where(bl_mean == 0, np.finfo(float).eps, bl_mean)
        corrected[i] = 10.0 * np.log10(tfr_epochs.data[i] / bl_mean)

    if n_short_baseline > 0:
        print(f"{prefix}⚠  {n_short_baseline} trials baseline 窗口太短，已自動擴展")

    tfr_epochs.data = corrected
    avg_tfr = tfr_epochs.average()

    print(f"{prefix}✓ ERSP 完成（per-trial logratio baseline + mirror padding）")
    return avg_tfr


def response_ersp_from_current_epochs(
    response_epochs,
    freqs=np.arange(4, 40, 1),
    n_cycles_func=None,
    decim=1,
    n_jobs=1,
    output_dir=r'C:\Experiment\Result\h5',
    subject_id=None,
    plot_dir=None,
    do_td_baseline=False,
    baseline_method='pre_stim',
):
    """
    從當前的 Response epochs 執行逐 trial ERSP 分析。

    流程：
    1. 確認 metadata 有 rt 欄位，排除 rt 為 NaN 或 rt < 0.15 s 的 trial
    2. 對每個 trial 執行 Morlet wavelet（average=False）
    3. 以各 trial 自己的 Blank 靜息期做 logratio baseline：
         baseline_start = -rt - 0.5 s（相對 response）
         baseline_end   = -rt - 0.1 s
    4. 平均所有校正後的 trial power
    5. 分 Learning / MotorTest / PerceptualTest 三個條件，
       並按 block grouping 細分
    6. 儲存為 .h5

    Parameters
    ----------
    response_epochs : mne.Epochs
        包含 metadata（需有 rt、phase、test_type、block 欄位）
    freqs : ndarray
    n_cycles_func : callable or None
    decim : int
    n_jobs : int
    output_dir : str
    subject_id : str

    Returns
    -------
    results : dict  {condition_key: AverageTFR}
    """

    print("\n" + "="*70)
    print("Response ERSP 分析（per-trial logratio baseline）")
    print("="*70)

    # ===== 步驟 1：確認參數 =====
    print("\n步驟 1：確認參數")

    if not hasattr(response_epochs, 'metadata') or response_epochs.metadata is None:
        raise ValueError("Response epochs 沒有 metadata！")

    if 'stim_sample' not in response_epochs.metadata.columns:
        raise ValueError("Response epochs 的 metadata 缺少 'stim_sample' 欄位，請重新建立 Epochs。")

    meta = response_epochs.metadata
    print(f"  Response epochs：{len(response_epochs)}")

    # ===== n_cycles =====
    if n_cycles_func is None:
        n_cycles = freqs / 2.0
    else:
        n_cycles = n_cycles_func(freqs)

    print(f"\n  頻率範圍：{freqs[0]}–{freqs[-1]} Hz")
    print(f"  n_cycles：{n_cycles[0]:.1f}–{n_cycles[-1]:.1f}")
    if baseline_method == 'whole_epoch':
        print(f"  Baseline：整段 epoch 平均（Lum et al. 2023）")
    else:
        print(f"  Baseline（per-trial）：-rt-0.5 → -rt-0.1 s（Blank 靜息期）")

    # Block grouping
    LEARNING_GROUPS = [(7, 11), (12, 16), (17, 21), (22, 26)]
    TESTING_GROUPS  = [(27, 28), (29, 30), (31, 32), (33, 34)]
    has_block     = 'block'     in meta.columns
    has_test_type = 'test_type' in meta.columns and 'phase' in meta.columns
    has_trial_type = 'trial_type' in meta.columns
    if has_trial_type:
        trial_types = sorted(response_epochs.metadata['trial_type'].dropna().unique().tolist())
    else:
        trial_types = ['All']

    results = {}

    def _process_and_save(cond_name, group_label, base_mask):
        """處理一個 condition+group 下的 Regular/Random（或 All）並存檔。"""
        group_results = {}
        for trial_type in trial_types:
            if trial_type == 'All':
                sub_mask = base_mask
            else:
                tt_mask  = meta['trial_type'].values == trial_type
                sub_mask = base_mask & tt_mask

            sub_n = sub_mask.sum()
            if sub_n == 0:
                print(f"    ⚠  {trial_type}：沒有資料，跳過")
                continue

            key = f"{cond_name}_{group_label}"
            print(f"\n  [{key} / {trial_type}]  {sub_n} trials")

            sub_indices = np.where(sub_mask)[0]
            epochs_sub  = response_epochs[sub_indices]

            power = _compute_pertrial_ersp(
                epochs_sub,
                freqs, n_cycles, decim, n_jobs,
                label=f"{key}/{trial_type}",
                do_td_baseline=do_td_baseline,
                baseline_method=baseline_method,
            )

            if output_dir and subject_id:
                os.makedirs(output_dir, exist_ok=True)
                h5_fname = f"{subject_id}_Response_{cond_name}_{group_label}_{trial_type}_ERSP.h5"
                out_path = os.path.join(output_dir, h5_fname)
                power.save(out_path, overwrite=True)
                print(f"  ✓ 已儲存：{out_path}")

            group_results[trial_type] = power
        return group_results

    # ===== 步驟 2：依條件分組計算 =====
    print("\n步驟 2：依條件分組計算 ERSP")

    if has_test_type:
        print("  偵測到 'test_type' 欄位，分三條件處理")

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
                print(f"\n  ⚠  Learning {group_label}：沒有資料，跳過")
                continue
            print(f"\n{'─'*60}")
            print(f"  Learning {group_label}  ({base_mask.sum()} trials)")
            print(f"{'─'*60}")
            results[f"Learning_{group_label}"] = _process_and_save('Learning', group_label, base_mask)

        # ── Motor Test / Perceptual Test ──
        for test_type_val, cond_name in [('motor', 'MotorTest'), ('perceptual', 'PerceptualTest')]:
            test_base = (
                (meta['phase'].values == 'Test') &
                (meta['test_type'].values == test_type_val)
            )
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
                    print(f"\n  ⚠  {cond_name} {group_label}：沒有資料，跳過")
                    continue
                print(f"\n{'─'*60}")
                print(f"  {cond_name} {group_label}  ({base_mask.sum()} trials)")
                print(f"{'─'*60}")
                results[f"{cond_name}_{group_label}"] = _process_and_save(cond_name, group_label, base_mask)

    else:
        # 沒有 test_type：當作單一條件處理
        print("  未偵測到 'test_type'，以單一條件處理全部 epochs")
        print(f"\n{'─'*60}")
        print(f"  All  ({len(response_epochs)} trials)")
        print(f"{'─'*60}")
        power = _compute_pertrial_ersp(
            response_epochs,
            freqs, n_cycles, decim, n_jobs,
            label="All",
            do_td_baseline=do_td_baseline,
            baseline_method=baseline_method,
        )
        if output_dir and subject_id:
            os.makedirs(output_dir, exist_ok=True)
            default_fname = f"{subject_id}_Response_All_ERSP.h5"
            h5_fname = input(f"\n請輸入 ERSP 檔名 [預設: {default_fname}]: ").strip() or default_fname
            if not h5_fname.endswith('.h5'):
                h5_fname += '.h5'
            out_path = os.path.join(output_dir, h5_fname)
            power.save(out_path, overwrite=True)
            print(f"  ✓ 已儲存：{out_path}")
        results['All'] = {'All': power}

    # ===== 步驟 3：繪圖 =====
    if plot_dir is None:
        from pathlib import Path
        plot_dir = str(Path(output_dir).parent) if output_dir else None

    if plot_dir and subject_id:
        try:
            from asrt.ersp_plots import (
                plot_ersp_lum2023_style,
                plot_learning_comparison,
                plot_testing_comparison,
            )

            print(f"\n{'─'*70}")
            print("繪製 Response ERSP 圖片...")
            print(f"{'─'*70}")

            for key, trial_results in results.items():
                power_dicts = {}

                for trial_type, tfr in trial_results.items():
                    pd_ = _tfr_to_power_dict(tfr)
                    if not pd_:
                        print(f"  ⚠  {key}/{trial_type}：找不到可用 ROI，略過")
                        continue
                    power_dicts[trial_type] = pd_

                    plot_ersp_lum2023_style(
                        pd_,
                        f"{subject_id}_response_{key}_{trial_type}",
                        plot_dir,
                    )

                cond1, cond2 = (trial_types[0], trial_types[1]) if len(trial_types) >= 2 else (None, None)
                if cond1 in power_dicts and cond2 in power_dicts:
                    parts       = key.split('_', 1)
                    cond_name   = parts[0]
                    group_label = parts[1] if len(parts) > 1 else None

                    if cond_name == 'Learning':
                        plot_learning_comparison(
                            {cond1: power_dicts[cond1],
                             cond2: power_dicts[cond2]},
                            subject_id=subject_id,
                            lock_type='response',
                            output_dir=plot_dir,
                            block_label=group_label,
                        )
                    elif cond_name == 'MotorTest':
                        plot_testing_comparison(
                            {cond1: power_dicts[cond1],
                             cond2: power_dicts[cond2]},
                            subject_id=subject_id,
                            lock_type='response',
                            test_type='motor',
                            output_dir=plot_dir,
                            block_label=group_label,
                        )
                    elif cond_name == 'PerceptualTest':
                        plot_testing_comparison(
                            {cond1: power_dicts[cond1],
                             cond2: power_dicts[cond2]},
                            subject_id=subject_id,
                            lock_type='response',
                            test_type='perceptual',
                            output_dir=plot_dir,
                            block_label=group_label,
                        )

        except Exception as _plot_err:
            print(f"  ⚠  繪圖時發生錯誤（不影響 .h5 資料）：{_plot_err}")
            import traceback; traceback.print_exc()

    total = sum(len(v) for v in results.values())
    print(f"\n{'='*70}")
    print(f"✓ Response ERSP 完成，共 {total} 個子條件")
    print(f"{'='*70}")

    return results


if __name__ == '__main__':
    import inspect
    for name, func in [
        ('_compute_pertrial_ersp', _compute_pertrial_ersp),
        ('response_ersp_from_current_epochs', response_ersp_from_current_epochs),
    ]:
        print(f'\n{"="*60}')
        print(f'函數：{name}')
        print(f'{"="*60}')
        sig = inspect.signature(func)
        print(f'參數：{sig}')
        src = inspect.getsource(func)
        print(src)
