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
    'Perceptual':           ['O1', 'Oz', 'O2', 'P3', 'Pz', 'P4'],
    'Perceptual_Parietal':  ['P3', 'Pz', 'P4'],
    'Perceptual_Occipital': ['O1', 'Oz', 'O2'],
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


def _compute_pertrial_ersp(epochs_subset, rt_subset, freqs, n_cycles, decim, n_jobs, label=''):
    """
    對一組 epochs 執行逐 trial Morlet wavelet，
    再套用各 trial 自己的 logratio baseline，最後平均。

    Baseline 窗口（相對於 response onset）：
        start = -rt - 0.5 s
        end   = -rt - 0.1 s
    這對應到 Stimulus 後約 700–1100 ms 的 Blank 靜息期。

    Parameters
    ----------
    epochs_subset : mne.Epochs
    rt_subset     : ndarray, shape (n_epochs,)  單位：秒
    freqs, n_cycles : ndarray
    decim, n_jobs : int
    label : str

    Returns
    -------
    avg_tfr : mne.time_frequency.AverageTFR
    """
    prefix = f"  [{label}] " if label else "  "
    n_epochs = len(epochs_subset)
    print(f"{prefix}計算逐 trial Morlet wavelet（{n_epochs} trials）...")

    tfr_epochs = mne.time_frequency.tfr_morlet(
        epochs_subset,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        decim=decim,
        n_jobs=n_jobs,
        average=False,
    )
    # tfr_epochs.data: (n_epochs, n_channels, n_freqs, n_times)

    times = tfr_epochs.times
    corrected = np.zeros_like(tfr_epochs.data)  # (n_epochs, n_ch, n_freqs, n_times)

    n_short_baseline = 0
    for i, rt in enumerate(rt_subset):
        bl_start = -rt - 0.5
        bl_end   = -rt - 0.1
        bl_mask  = (times >= bl_start) & (times <= bl_end)

        if bl_mask.sum() == 0:
            # fallback：向前擴展 200 ms
            bl_mask = (times >= bl_start - 0.2) & (times <= bl_end)
            n_short_baseline += 1

        # bl_mean shape: (n_ch, n_freqs, 1)
        bl_mean = tfr_epochs.data[i][:, :, bl_mask].mean(axis=-1, keepdims=True)
        # 避免除以零
        bl_mean = np.where(bl_mean == 0, np.finfo(float).eps, bl_mean)
        corrected[i] = 10.0 * np.log10(tfr_epochs.data[i] / bl_mean)

    if n_short_baseline > 0:
        print(f"{prefix}⚠  {n_short_baseline} trials baseline 窗口太短，已自動擴展")

    # 把校正後的資料寫回，再用 .average() 產生 AverageTFR
    tfr_epochs.data = corrected
    avg_tfr = tfr_epochs.average()

    print(f"{prefix}✓ ERSP 完成（per-trial logratio baseline）")
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

    # ===== 步驟 1：檢查 metadata 和 RT =====
    print("\n步驟 1：檢查 metadata 和 RT")

    if not hasattr(response_epochs, 'metadata') or response_epochs.metadata is None:
        raise ValueError("Response epochs 沒有 metadata！")

    meta = response_epochs.metadata
    if 'rt' not in meta.columns:
        raise ValueError("Metadata 中沒有 'rt' 欄位！請先執行 RT 對齊步驟")

    rt_values = meta['rt'].values.astype(float)

    # 排除 NaN
    valid_mask = ~np.isnan(rt_values)
    n_nan = np.sum(~valid_mask)
    if n_nan > 0:
        print(f"  排除 rt=NaN：{n_nan} trials")

    # 排除 RT < 0.15 s（過快，不可靠）
    valid_mask &= (rt_values >= 0.15)
    n_fast = np.sum(~valid_mask) - n_nan
    if n_fast > 0:
        print(f"  排除 rt<0.15s：{n_fast} trials")

    response_epochs = response_epochs[valid_mask]
    meta            = response_epochs.metadata
    rt_values       = meta['rt'].values.astype(float)

    print(f"  保留 epochs：{len(response_epochs)}")
    print(f"  RT 範圍：{rt_values.min()*1000:.1f} – {rt_values.max()*1000:.1f} ms")
    print(f"  RT 平均：{rt_values.mean()*1000:.1f} ± {rt_values.std()*1000:.1f} ms")

    # ===== n_cycles =====
    if n_cycles_func is None:
        n_cycles = freqs / 2.0
    else:
        n_cycles = n_cycles_func(freqs)

    print(f"\n  頻率範圍：{freqs[0]}–{freqs[-1]} Hz")
    print(f"  n_cycles：{n_cycles[0]:.1f}–{n_cycles[-1]:.1f}")
    print(f"  Baseline（per-trial）：-rt-0.5 → -rt-0.1 s（Blank 靜息期）")

    # Block grouping
    LEARNING_GROUPS = [(7, 11), (12, 16), (17, 21), (22, 26)]
    TESTING_GROUPS  = [(27, 28), (29, 30), (31, 32), (33, 34)]
    has_block     = 'block'     in meta.columns
    has_test_type = 'test_type' in meta.columns and 'phase' in meta.columns
    has_trial_type = 'trial_type' in meta.columns
    trial_types = ['Regular', 'Random'] if has_trial_type else ['All']

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

            sub_indices  = np.where(sub_mask)[0]
            epochs_sub   = response_epochs[sub_indices]
            rt_sub       = rt_values[sub_mask]

            power = _compute_pertrial_ersp(
                epochs_sub, rt_sub,
                freqs, n_cycles, decim, n_jobs,
                label=f"{key}/{trial_type}",
            )

            if output_dir and subject_id:
                os.makedirs(output_dir, exist_ok=True)
                default_fname = f"{subject_id}_Response_{cond_name}_{group_label}_{trial_type}_ERSP.h5"
                h5_fname = input(f"\n請輸入 ERSP 檔名 [預設: {default_fname}]: ").strip() or default_fname
                if not h5_fname.endswith('.h5'):
                    h5_fname += '.h5'
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
            response_epochs, rt_values,
            freqs, n_cycles, decim, n_jobs,
            label="All",
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

                if 'Regular' in power_dicts and 'Random' in power_dicts:
                    parts       = key.split('_', 1)
                    cond_name   = parts[0]
                    group_label = parts[1] if len(parts) > 1 else None

                    if cond_name == 'Learning':
                        plot_learning_comparison(
                            {'Regular': power_dicts['Regular'],
                             'Random':  power_dicts['Random']},
                            subject_id=subject_id,
                            lock_type='response',
                            output_dir=plot_dir,
                            block_label=group_label,
                        )
                    elif cond_name == 'MotorTest':
                        plot_testing_comparison(
                            {'Regular': power_dicts['Regular'],
                             'Random':  power_dicts['Random']},
                            subject_id=subject_id,
                            lock_type='response',
                            test_type='motor',
                            output_dir=plot_dir,
                            block_label=group_label,
                        )
                    elif cond_name == 'PerceptualTest':
                        plot_testing_comparison(
                            {'Regular': power_dicts['Regular'],
                             'Random':  power_dicts['Random']},
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
