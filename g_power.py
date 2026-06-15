"""
compute_gpower.py
=================
獨立計算 G*Power Cohen's d 的腳本。
不需要重新跑群體分析，直接讀取已存在的 h5 / pkl 檔案。

輸出：
  【學習階段】Block7-11 ~ Block22-26 平均
    - Response-locked │ Motor ROI       │ Theta (4–8 Hz)   │ −300 to +50 ms
    - Stimulus-locked │ Perceptual ROI  │ Alpha (8–13 Hz)  │ +100 to +300 ms

  【測驗階段】AllBlocks
    - MotorTest    │ Response-locked │ Motor ROI      │ Theta  ← 預期顯著
    - MotorTest    │ Stimulus-locked │ Perceptual ROI │ Alpha  ← 預期消失
    - PerceptualTest│ Stimulus-locked │ Perceptual ROI │ Alpha  ← 預期顯著
    - PerceptualTest│ Response-locked │ Motor ROI      │ Theta  ← 預期消失

使用方式：
  python compute_gpower.py

作者: Dillian (HE-JUN CHEN)
"""

import pickle
import warnings
import numpy as np
from pathlib import Path

# ══════════════════════════════════════════════════════════════════
#  ★ 使用前請修改以下路徑與受試者列表
# ══════════════════════════════════════════════════════════════════
H5_DIR      = r'C:\Experiment\Result\triplet\h5'   # response-locked .h5 目錄
PKL_DIR     = r'C:\Experiment\Result\h5'       # stimulus-locked .pkl 目錄
                                               # （stimulus AllBlocks .h5 也在此）

SUBJECT_IDS = [
    'sub0001', 'sub0002', 'sub0003', 'sub0004', 'sub0005',
    'sub0006', 'sub0007', 'sub0008', 'sub0009', 'sub00010',
]

ALPHA        = 0.05   # 顯著水準
POWER_TARGET = 0.80   # 目標統計力

COND_L = 'regular_high'   # 左側條件（預期效果大）
COND_R = 'random_low'     # 右側條件

# ══════════════════════════════════════════════════════════════════
#  ROI 定義（與 group_ersp_analysis.py 完全相同）
# ══════════════════════════════════════════════════════════════════
ROI_GROUPS = {
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

# ══════════════════════════════════════════════════════════════════
#  檔案讀取函數（從 group_ersp_analysis.py 提取核心邏輯）
# ══════════════════════════════════════════════════════════════════

def _load_pkl(filepath):
    """讀取 stimulus-locked .pkl，回傳 (ersp_2d, freqs, times, nave)。"""
    print(f"Reading {filepath} ...")   # ← 加這行
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['ersp'], data['freqs'], data['times'], int(data.get('nave', -1))


def _load_h5(filepath, roi_name):
    """
    讀取 MNE AverageTFR .h5，提取指定 ROI 的頻道並平均。
    回傳 (ersp_2d, freqs, times, nave)。
    """
    import mne
    roi_channels = None
    for key in ROI_GROUPS:
        if key.lower() == roi_name.lower():
            roi_channels = ROI_GROUPS[key]
            break
    if roi_channels is None:
        raise ValueError(f"未知 ROI: '{roi_name}'，可用: {list(ROI_GROUPS.keys())}")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tfr_list = mne.time_frequency.read_tfrs(str(filepath))
    tfr = tfr_list[0] if isinstance(tfr_list, list) else tfr_list

    ch_names = [ch.upper() for ch in tfr.ch_names]
    roi_idx  = [ch_names.index(ch.upper())
                for ch in roi_channels if ch.upper() in ch_names]
    if not roi_idx:
        raise ValueError(f"ROI '{roi_name}' 的所有頻道均不在檔案中。")

    power = tfr.data[roi_idx].mean(axis=0)   # (n_freqs, n_times)
    return power, tfr.freqs, tfr.times, int(tfr.nave)


# ══════════════════════════════════════════════════════════════════
#  Cohen's d 與 G*Power 計算
# ══════════════════════════════════════════════════════════════════

def _roi_mean(ersp, freqs, times, freq_range, time_range):
    """在特定 time-frequency 視窗內取平均 ERSP 值。"""
    f_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    t_mask = (times >= time_range[0]) & (times <= time_range[1])
    return float(ersp[np.ix_(f_mask, t_mask)].mean())


def _cohens_d_and_power(vals_l, vals_r, alpha, power_target):
    """
    計算 paired Cohen's d_z 及 G*Power 所需樣本數。

    Parameters
    ----------
    vals_l, vals_r : list[float]  每位受試者的條件均值
    alpha          : float        顯著水準
    power_target   : float        目標統計力

    Returns
    -------
    dict with keys: n, d, mean_diff, std_diff, n_required, power_at_pilot
    """
    from statsmodels.stats.power import TTestPower
    pwr = TTestPower()

    diffs  = np.array(vals_l) - np.array(vals_r)
    n      = len(diffs)
    mean_d = float(diffs.mean())
    std_d  = float(diffs.std(ddof=1))
    d      = mean_d / std_d if std_d > 1e-12 else 0.0

    n_req = int(np.ceil(
        pwr.solve_power(effect_size=abs(d), alpha=alpha,
                        power=power_target, alternative='larger')
    ))
    pwr_now = float(
        pwr.solve_power(effect_size=abs(d), alpha=alpha,
                        nobs=n, alternative='larger')
    )
    return dict(n=n, d=d, mean_diff=mean_d, std_diff=std_d,
                n_required=n_req, power_at_pilot=pwr_now)


# ══════════════════════════════════════════════════════════════════
#  學習階段：Block7-11 ~ Block22-26（四個 block 組的平均）
# ══════════════════════════════════════════════════════════════════

LEARNING_BLOCK_GROUPS = ['Block7-11', 'Block12-16', 'Block17-21', 'Block22-26']

LEARNING_CONFIGS = [
    dict(
        label     = 'Learning | Response | Motor      | Theta ERD',
        lock      = 'response',
        roi       = 'Motor',
        roi_lower = 'motor',
        freq      = (4,  8),
        time      = (-0.300, 0.050),
    ),
    dict(
        label     = 'Learning | Stimulus | Perceptual | Alpha ERS',
        lock      = 'stimulus',
        roi       = 'Perceptual',
        roi_lower = 'perceptual',
        freq      = (8, 13),
        time      = (0.100, 0.300),
    ),
]


def run_learning(cfg):
    """學習階段：讀取四個 block 組，每位受試者算 regular − random 差值。"""
    vals_l, vals_r = [], []
    for sid in SUBJECT_IDS:
        sub_l, sub_r = [], []
        for blk in LEARNING_BLOCK_GROUPS:
            if cfg['lock'] == 'stimulus':
                fp_l = (Path(PKL_DIR) /
                        f'{sid}_learning_stimulus_{cfg["roi_lower"]}_{COND_L}_{blk}_ersp.pkl')
                fp_r = (Path(PKL_DIR) /
                        f'{sid}_learning_stimulus_{cfg["roi_lower"]}_{COND_R}_{blk}_ersp.pkl')
                if not fp_l.exists() or not fp_r.exists():
                    continue
                try:
                    ersp_l, freqs, times, _ = _load_pkl(fp_l)
                    ersp_r, _,     _,     _ = _load_pkl(fp_r)
                except Exception as e:
                    print(f"    ⚠ {sid} {blk} pkl 讀取失敗: {e}")
                    continue
            else:
                fp_l = (Path(H5_DIR) /
                        f'{sid}_Response_Learning_{blk}_{COND_L}_ERSP.h5')
                fp_r = (Path(H5_DIR) /
                        f'{sid}_Response_Learning_{blk}_{COND_R}_ERSP.h5')
                if not fp_l.exists() or not fp_r.exists():
                    continue
                try:
                    ersp_l, freqs, times, _ = _load_h5(fp_l, cfg['roi'])
                    ersp_r, _,     _,     _ = _load_h5(fp_r, cfg['roi'])
                except Exception as e:
                    print(f"    ⚠ {sid} {blk} h5 讀取失敗: {e}")
                    continue
            sub_l.append(_roi_mean(ersp_l, freqs, times, cfg['freq'], cfg['time']))
            sub_r.append(_roi_mean(ersp_r, freqs, times, cfg['freq'], cfg['time']))
        if sub_l:
            vals_l.append(float(np.mean(sub_l)))
            vals_r.append(float(np.mean(sub_r)))
    return vals_l, vals_r


# ══════════════════════════════════════════════════════════════════
#  測驗階段：AllBlocks（每位受試者一個檔案）
# ══════════════════════════════════════════════════════════════════

TESTING_CONFIGS = [
    # ── 預期顯著（G*Power 的主要依據）──────────────────────────
    dict(
        label       = 'Testing | MotorTest    | Response | Motor      | Theta ERD  ← 預期顯著',
        lock        = 'response',
        cond_name   = 'MotorTest',
        roi         = 'Motor',
        freq        = (4,  8),
        time        = (-0.300, 0.050),
        expected    = 'significant',
    ),
    dict(
        label       = 'Testing | PerceptTest  | Stimulus | Perceptual | Alpha ERS  ← 預期顯著',
        lock        = 'stimulus',
        cond_name   = 'PerceptualTest',
        roi         = 'Perceptual',
        freq        = (8, 13),
        time        = (0.100, 0.300),
        expected    = 'significant',
    ),
    # ── 預期消失（雙重解離的另一半）────────────────────────────
    dict(
        label       = 'Testing | MotorTest    | Stimulus | Perceptual | Alpha ERS  ← 預期消失',
        lock        = 'stimulus',
        cond_name   = 'MotorTest',
        roi         = 'Perceptual',
        freq        = (8, 13),
        time        = (0.100, 0.300),
        expected    = 'absent',
    ),
    dict(
        label       = 'Testing | PerceptTest  | Response | Motor      | Theta ERD  ← 預期消失',
        lock        = 'response',
        cond_name   = 'PerceptualTest',
        roi         = 'Motor',
        freq        = (4,  8),
        time        = (-0.300, 0.050),
        expected    = 'absent',
    ),
]


def run_testing(cfg):
    """測驗階段：每位受試者讀一個 AllBlocks 檔案。"""
    lock_cap = cfg['lock'].capitalize()
    # response-locked → h5_dir；stimulus-locked → pkl_dir（但格式是 .h5）
    base_dir = Path(H5_DIR) if cfg['lock'] == 'response' else Path(PKL_DIR)

    vals_l, vals_r = [], []
    for sid in SUBJECT_IDS:
        fp_l = base_dir / f'{sid}_{lock_cap}_{cfg["cond_name"]}_AllBlocks_{COND_L}_ERSP.h5'
        fp_r = base_dir / f'{sid}_{lock_cap}_{cfg["cond_name"]}_AllBlocks_{COND_R}_ERSP.h5'
        if not fp_l.exists() or not fp_r.exists():
            continue
        try:
            ersp_l, freqs, times, _ = _load_h5(fp_l, cfg['roi'])
            ersp_r, _,     _,     _ = _load_h5(fp_r, cfg['roi'])
        except Exception as e:
            print(f"    ⚠ {sid} {cfg['cond_name']} AllBlocks 讀取失敗: {e}")
            continue
        vals_l.append(_roi_mean(ersp_l, freqs, times, cfg['freq'], cfg['time']))
        vals_r.append(_roi_mean(ersp_r, freqs, times, cfg['freq'], cfg['time']))
    return vals_l, vals_r


# ══════════════════════════════════════════════════════════════════
#  輸出格式化
# ══════════════════════════════════════════════════════════════════

def print_row(label, vals_l, vals_r, alpha, power_target, expected=''):
    n = len(vals_l)
    if n < 3:
        print(f"  ⚠  {label}")
        print(f"     有效受試者數不足（{n}），跳過\n")
        return None

    res = _cohens_d_and_power(vals_l, vals_r, alpha, power_target)
    flag = '★' if expected == 'significant' else ' '

    mean_l   = float(np.mean(vals_l))
    mean_r   = float(np.mean(vals_r))
    mean_diff = mean_l - mean_r
    state_l  = 'ERS(+)' if mean_l   > 0 else 'ERD(-)'
    state_r  = 'ERS(+)' if mean_r   > 0 else 'ERD(-)'
    state_d  = 'ERS(+)' if mean_diff > 0 else 'ERD(-)'

    print(f"  {flag} {label}")
    print(f"     {COND_L:>15s} = {mean_l:+.4f} dB [{state_l}]   "
          f"{COND_R:>12s} = {mean_r:+.4f} dB [{state_r}]   "
          f"diff = {mean_diff:+.4f} dB [{state_d}]")
    print(f"     N(pilot)={res['n']:2d}  │  "
          f"std_diff={res['std_diff']:.4f}  │  "
          f"Cohen's d={res['d']:+.3f}  │  "
          f"N required={res['n_required']:4d}  │  "
          f"power@pilot={res['power_at_pilot']:.3f}")
    print()

    res['mean_l']    = mean_l
    res['mean_r']    = mean_r
    res['mean_diff_abs'] = mean_diff
    res['label']     = label
    res['expected']  = expected
    return res


def print_summary_table(all_results):
    """所有條件跑完後印出彙整表格。"""
    print("\n" + "═"*110)
    print("  總結表格")
    print("═"*110)
    hdr = (f"  {'條件':<52}  {'reg_high':>9}  {'rand_low':>9}  "
           f"{'diff':>8}  {'d':>7}  {'N req':>6}  {'pwr':>6}  {'預期'}")
    print(hdr)
    print("─"*110)
    for res in all_results:
        if res is None:
            continue
        short = res['label'].replace('Testing | ', '').replace('Learning | ', 'Lrn | ')
        tag   = '★顯著' if res['expected'] == 'significant' else ('○消失' if res['expected'] == 'absent' else '      ')
        sl    = '+' if res['mean_l'] > 0 else '-'
        sr    = '+' if res['mean_r'] > 0 else '-'
        sd    = '+' if res['mean_diff_abs'] > 0 else '-'
        print(f"  {short:<52}  "
              f"{sl}{abs(res['mean_l']):.4f}   "
              f"{sr}{abs(res['mean_r']):.4f}   "
              f"{sd}{abs(res['mean_diff_abs']):.4f}  "
              f"{res['d']:+.3f}  "
              f"{res['n_required']:6d}  "
              f"{res['power_at_pilot']:.3f}  "
              f"{tag}")
    print("═"*110)
    print("  ※ diff = regular_high − random_low（正值 = regular 功率較高）")
    print("  ※ N required：one-tailed, α=0.05, power=0.80")
    print("═"*110 + "\n")


# ══════════════════════════════════════════════════════════════════
#  主程式
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "█"*72)
    print("  G*Power 樣本數估算")
    print(f"  one-tailed │ α={ALPHA} │ target power={POWER_TARGET}")
    print(f"  {COND_L}  vs  {COND_R}")
    print("█"*72 + "\n")

    all_results = []

    # ── 學習階段 ───────────────────────────────────────────────
    print("━"*72)
    print("【學習階段】  Block7-11 ~ Block22-26（四個 block 組平均）")
    print("━"*72)
    for cfg in LEARNING_CONFIGS:
        vl, vr = run_learning(cfg)
        res = print_row(cfg['label'], vl, vr, ALPHA, POWER_TARGET)
        all_results.append(res)

    # ── 測驗階段 ───────────────────────────────────────────────
    print("━"*72)
    print("【測驗階段】  AllBlocks")
    print("━"*72)
    for cfg in TESTING_CONFIGS:
        vl, vr = run_testing(cfg)
        res = print_row(cfg['label'], vl, vr, ALPHA, POWER_TARGET,
                        expected=cfg['expected'])
        all_results.append(res)

    # ── 彙整表格 ───────────────────────────────────────────────
    print_summary_table(all_results)
