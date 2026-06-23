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
    print(f"Reading {filepath} ...")   #
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
        label     = 'Learning | Stimulus | Perceptual | Alpha (post-stim 100-300ms)',
        lock      = 'stimulus',
        roi       = 'Perceptual',
        roi_lower = 'perceptual',
        freq      = (8, 13),
        time      = (0.100, 0.300),
    ),
    # ── 新增：pre-stimulus 窗(anticipatory alpha,對應 Bays 2015 的 ERS 階段)──
    # whole-epoch baseline 下,此窗若為正值=相對整段平均的 ERS(預期隨學習增大)
    dict(
        label     = 'Learning | Stimulus | Perceptual | Alpha (PRE-stim -500~-100ms)',
        lock      = 'stimulus',
        roi       = 'Perceptual',
        roi_lower = 'perceptual',
        freq      = (8, 13),
        time      = (-0.500, -0.100),
    ),
    # ── 可選:對齊 Bays 的刺激處理窗(250-500ms),預期為更深的 ERD ──
    dict(
        label     = 'Learning | Stimulus | Perceptual | Alpha (post-stim 250-500ms, Bays)',
        lock      = 'stimulus',
        roi       = 'Perceptual',
        roi_lower = 'perceptual',
        freq      = (8, 13),
        time      = (0.250, 0.500),
    ),
]


def run_learning(cfg):
    """
    學習階段：對每位受試者，在每個 block 組計算 RegH-RandL 差值，
    再以 block 序號（1,2,3,4）為 X、差值為 Y 做線性回歸，取個別斜率
    作為「序列學習的時間趨勢」（model changes across blocks）。
    斜率 < 0 代表 RegH 相對 RandL 的 Theta 在學習過程中逐漸更大的 ERD。
    回傳 slopes, per_sub_diffs, per_sub_means。
    """
    x = np.arange(1, len(LEARNING_BLOCK_GROUPS) + 1, dtype=float)
    x_c = x - x.mean()   # 中心化，讓截距等於整體均值

    slopes, per_sub_diffs, per_sub_means = [], [], []
    for sid in SUBJECT_IDS:
        diffs = []
        for blk in LEARNING_BLOCK_GROUPS:
            if cfg['lock'] == 'stimulus':
                fp_l = (Path(PKL_DIR) /
                        f'{sid}_learning_stimulus_{cfg["roi_lower"]}_{COND_L}_{blk}_ersp.pkl')
                fp_r = (Path(PKL_DIR) /
                        f'{sid}_learning_stimulus_{cfg["roi_lower"]}_{COND_R}_{blk}_ersp.pkl')
                if not fp_l.exists() or not fp_r.exists():
                    diffs.append(None); continue
                try:
                    ersp_l, freqs, times, _ = _load_pkl(fp_l)
                    ersp_r, _,     _,     _ = _load_pkl(fp_r)
                except Exception as e:
                    print(f"    \u26a0 {sid} {blk} pkl \u8b80\u53d6\u5931\u6557: {e}")
                    diffs.append(None); continue
            else:
                fp_l = (Path(H5_DIR) /
                        f'{sid}_Response_Learning_{blk}_{COND_L}_ERSP.h5')
                fp_r = (Path(H5_DIR) /
                        f'{sid}_Response_Learning_{blk}_{COND_R}_ERSP.h5')
                if not fp_l.exists() or not fp_r.exists():
                    diffs.append(None); continue
                try:
                    ersp_l, freqs, times, _ = _load_h5(fp_l, cfg['roi'])
                    ersp_r, _,     _,     _ = _load_h5(fp_r, cfg['roi'])
                except Exception as e:
                    print(f"    \u26a0 {sid} {blk} h5 \u8b80\u53d6\u5931\u6557: {e}")
                    diffs.append(None); continue
            d = (_roi_mean(ersp_l, freqs, times, cfg['freq'], cfg['time']) -
                 _roi_mean(ersp_r, freqs, times, cfg['freq'], cfg['time']))
            diffs.append(d)

        valid = [d is not None for d in diffs]
        if sum(valid) < 2:
            continue
        xv = x_c[valid]
        yv = np.array([d for d, m in zip(diffs, valid) if m])
        slope = float(np.dot(xv, yv) / np.dot(xv, xv))
        slopes.append(slope)
        per_sub_diffs.append([d if d is not None else np.nan for d in diffs])
        per_sub_means.append(float(np.nanmean(yv)))
    return slopes, per_sub_diffs, per_sub_means


# ══════════════════════════════════════════════════════════════════
#  測驗階段：AllBlocks（每位受試者一個檔案）
# ══════════════════════════════════════════════════════════════════

# 測驗階段配置：定義兩個「神經測量」各自的參數
# 每個測量都要計算 Task(Motor/Perceptual) × Condition(RegH/RandL) 的交互作用，
# 而不是分開算 Motor Test 跟 Perceptual Test 各自的 t-test。
# 原因：研究問題是「Motor 跟 Perceptual 的序列效果是否有差異」，
# 這是一個交互作用問題，不是兩個獨立的主效果問題。
# 用分開的 t-test 無法推論交互作用（即使一個顯著一個不顯著也不行）。
TESTING_NEURAL_CONFIGS = [
    dict(
        label    = 'Testing | Motor ROI | Theta ERD | Response-locked',
        lock     = 'response',
        roi      = 'Motor',
        freq     = (4,  8),
        time     = (-0.300, 0.050),
        # 預期方向：Motor Test 的 RegH-RandL 差值（負值=ERD）
        # 應該大於 Perceptual Test 的對應差值
        # → interaction = motor_diff - percept_diff 預期為負
        exp_dir  = 'negative',
    ),
    dict(
        label    = 'Testing | Perceptual ROI | Alpha (post-stim 100-300ms) | Stimulus-locked',
        lock     = 'stimulus',
        roi      = 'Perceptual',
        freq     = (8, 13),
        time     = (0.100, 0.300),
        exp_dir  = 'positive',
    ),
    # ── 新增:pre-stimulus 窗(Bays 2015 的 anticipatory ERS 階段)──
    # 與上面同一批檔案、同一套交互作用邏輯,只換時間窗。
    # 若 pre-stim 為 ERS(正)而 post-stim 為 ERD(負),即吻合 Bays 兩階段模型。
    dict(
        label    = 'Testing | Perceptual ROI | Alpha (PRE-stim -500~-100ms) | Stimulus-locked',
        lock     = 'stimulus',
        roi      = 'Perceptual',
        freq     = (8, 13),
        time     = (-0.500, -0.100),
        exp_dir  = 'positive',
    ),
    # ── 可選:對齊 Bays 的刺激處理窗(250-500ms),預期更深 ERD ──
    dict(
        label    = 'Testing | Perceptual ROI | Alpha (post-stim 250-500ms, Bays) | Stimulus-locked',
        lock     = 'stimulus',
        roi      = 'Perceptual',
        freq     = (8, 13),
        time     = (0.250, 0.500),
        exp_dir  = 'positive',
    ),
]


def _load_testing_val(sid, lock, cond_name, roi, freq, time):
    """
    讀取單一受試者、單一任務條件（MotorTest/PerceptualTest）、
    單一刺激條件（regular_high/random_low）的 ERSP 均值。
    回傳 (val_regh, val_randl) 或 (None, None)。
    """
    lock_cap = lock.capitalize()
    base_dir = Path(H5_DIR) if lock == 'response' else Path(PKL_DIR)
    fp_l = base_dir / f'{sid}_{lock_cap}_{cond_name}_AllBlocks_{COND_L}_ERSP.h5'
    fp_r = base_dir / f'{sid}_{lock_cap}_{cond_name}_AllBlocks_{COND_R}_ERSP.h5'
    if not fp_l.exists() or not fp_r.exists():
        return None, None
    try:
        ersp_l, freqs, times, _ = _load_h5(fp_l, roi)
        ersp_r, _,     _,     _ = _load_h5(fp_r, roi)
    except Exception as e:
        print(f"    ⚠ {sid} {cond_name} 讀取失敗: {e}")
        return None, None
    return (_roi_mean(ersp_l, freqs, times, freq, time),
            _roi_mean(ersp_r, freqs, times, freq, time))


def run_testing_interaction(cfg):
    """
    計算 Task × Condition 交互作用分數。

    對每位受試者：
      motor_diff   = ERSP(MotorTest, RegH)      - ERSP(MotorTest, RandL)
      percept_diff = ERSP(PerceptualTest, RegH) - ERSP(PerceptualTest, RandL)
      interaction  = motor_diff - percept_diff
        （如果 exp_dir=='positive' 則取 percept_diff - motor_diff，
          讓正值=預期方向，方便閱讀）

    回傳 interaction 分數的 list，以及各任務條件的原始均值（供描述統計用）。
    """
    interactions   = []
    motor_regh_all = []
    motor_rand_all = []
    perc_regh_all  = []
    perc_rand_all  = []

    for sid in SUBJECT_IDS:
        m_h, m_l = _load_testing_val(sid, cfg['lock'], 'MotorTest',
                                     cfg['roi'], cfg['freq'], cfg['time'])
        p_h, p_l = _load_testing_val(sid, cfg['lock'], 'PerceptualTest',
                                     cfg['roi'], cfg['freq'], cfg['time'])
        if any(v is None for v in [m_h, m_l, p_h, p_l]):
            continue

        motor_diff   = m_h - m_l
        percept_diff = p_h - p_l
        # 統一讓正值 = 預期方向（Motor ROI Theta: motor_diff < percept_diff → motor-percept < 0
        # 所以 Theta 用 motor-percept；Alpha 用 percept-motor，兩者預期方向都是負的）
        # 實際上為了可讀性，都用 motor_diff - percept_diff，用 exp_dir 標示方向
        interaction = motor_diff - percept_diff

        interactions.append(interaction)
        motor_regh_all.append(m_h)
        motor_rand_all.append(m_l)
        perc_regh_all.append(p_h)
        perc_rand_all.append(p_l)

    return (interactions,
            motor_regh_all, motor_rand_all,
            perc_regh_all,  perc_rand_all)


def _cohens_d_one_sample(vals, alpha, power_target):
    """
    單樣本（對照值=0）Cohen's d_z、95% CI 與 G*Power 樣本數。
    用於交互作用分數：H0: mean(interaction) = 0。
    CI 用 t 分布：mean ± t_{α/2, df=n-1} × SE。
    """
    from statsmodels.stats.power import TTestPower
    from scipy import stats as _stats
    pwr = TTestPower()
    arr    = np.array(vals)
    n      = len(arr)
    mean_v = float(arr.mean())
    std_v  = float(arr.std(ddof=1))
    se     = std_v / np.sqrt(n)
    d      = mean_v / std_v if std_v > 1e-12 else 0.0
    # 95% CI（兩尾 t，df = n-1）
    t_crit = float(_stats.t.ppf(0.975, df=n-1))
    ci_lo  = mean_v - t_crit * se
    ci_hi  = mean_v + t_crit * se
    n_req  = int(np.ceil(
        pwr.solve_power(effect_size=abs(d), alpha=alpha,
                        power=power_target, alternative='larger')
    ))
    pwr_now = float(
        pwr.solve_power(effect_size=abs(d), alpha=alpha,
                        nobs=n, alternative='larger')
    )
    return dict(n=n, d=d, mean=mean_v, std=std_v, se=se,
                ci_lo=ci_lo, ci_hi=ci_hi,
                n_required=n_req, power_at_pilot=pwr_now)


def print_interaction_row(cfg, interactions,
                          motor_regh, motor_rand,
                          perc_regh,  perc_rand):
    """印出交互作用分析結果。"""
    n = len(interactions)
    if n < 3:
        print(f"  ⚠  {cfg['label']} — 有效受試者不足（{n}），跳過\n")
        return None

    res = _cohens_d_one_sample(interactions, ALPHA, POWER_TARGET)

    mean_m_h = float(np.mean(motor_regh))
    mean_m_l = float(np.mean(motor_rand))
    mean_p_h = float(np.mean(perc_regh))
    mean_p_l = float(np.mean(perc_rand))
    mean_m_diff = mean_m_h - mean_m_l
    mean_p_diff = mean_p_h - mean_p_l
    mean_inter  = float(np.mean(interactions))   # = mean_m_diff - mean_p_diff

    print(f"  ★ {cfg['label']}")
    print(f"     MotorTest:      RegH={mean_m_h:+.4f}  RandL={mean_m_l:+.4f}  "
          f"diff={mean_m_diff:+.4f} dB")
    print(f"     PerceptualTest: RegH={mean_p_h:+.4f}  RandL={mean_p_l:+.4f}  "
          f"diff={mean_p_diff:+.4f} dB")
    print(f"     Interaction (MotorDiff - PerceptDiff) = {mean_inter:+.4f} dB  "
          f"[預期方向: {'< 0 (Motor ERD 更大)' if cfg['exp_dir']=='negative' else '> 0 (Percept ERS 更大)'}]")
    print(f"     N(pilot)={res['n']}  │  std={res['std']:.4f}  │  "
          f"Cohen's d_z={res['d']:+.3f}  │  "
          f"95% CI [{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}]  │  "
          f"N required={res['n_required']}  │  "
          f"power@pilot={res['power_at_pilot']:.3f}")
    print()

    res.update({
        'label': cfg['label'],
        'exp_dir': cfg['exp_dir'],
        'mean_m_diff': mean_m_diff,
        'mean_p_diff': mean_p_diff,
        'mean_interaction': mean_inter,
        # 保留給視覺化用
        'motor_regh': mean_m_h, 'motor_rand': mean_m_l,
        'perc_regh':  mean_p_h, 'perc_rand':  mean_p_l,
    })
    return res


# ══════════════════════════════════════════════════════════════════
#  輸出格式化
# ══════════════════════════════════════════════════════════════════

def print_learning_slope_row(label, slopes, per_sub_diffs, alpha, power_target):
    """
    Learning 階段斜率結果輸出。
    slopes: 每位受試者的線性回歸斜率（dB / block-step）。
    H0: mean(slope) = 0（RegH-RandL ERSP 沒有跨 block 趨勢）。
    補上 95% CI 與 G*Power 樣本數。
    """
    from scipy import stats as _stats
    from statsmodels.stats.power import TTestPower

    n = len(slopes)
    if n < 3:
        print(f"  \u26a0  {label} — 有效受試者不足（{n}），跳過\n")
        return None

    arr  = np.array(slopes)
    mean_s = float(arr.mean())
    std_s  = float(arr.std(ddof=1))
    se_s   = std_s / np.sqrt(n)
    d      = mean_s / std_s if std_s > 1e-12 else 0.0
    t_crit = float(_stats.t.ppf(0.975, df=n-1))
    ci_lo  = mean_s - t_crit * se_s
    ci_hi  = mean_s + t_crit * se_s
    t_stat, p_val = _stats.ttest_1samp(arr, 0)

    pwr_obj = TTestPower()
    n_req = int(np.ceil(
        pwr_obj.solve_power(effect_size=abs(d), alpha=alpha,
                            power=power_target, alternative='larger')
    ))
    pwr_now = float(
        pwr_obj.solve_power(effect_size=abs(d), alpha=alpha,
                            nobs=n, alternative='larger')
    )

    # 每個 block 組的跨人均值（供描述統計）
    per_block = np.nanmean(np.array(per_sub_diffs), axis=0) if per_sub_diffs else []
    blk_str = "  ".join([f"Blk{i+1}={v:+.4f}" for i, v in enumerate(per_block)])

    print(f"  \u2605 {label}")
    print(f"     Group means per block: {blk_str}")
    print(f"     Slope mean={mean_s:+.5f} dB/step  std={std_s:.5f}  SE={se_s:.5f}")
    print(f"     t({n-1})={t_stat:+.3f}  p={p_val:.3f}  (two-tailed, for reference)")
    print(f"     Cohen's d_z={d:+.3f}  95% CI [{ci_lo:+.5f}, {ci_hi:+.5f}]")
    print(f"     N required={n_req}  power@pilot={pwr_now:.3f}")
    print()

    return dict(label=label, type='learning_slope',
                n=n, mean_slope=mean_s, std_slope=std_s,
                d=d, ci_lo=ci_lo, ci_hi=ci_hi,
                t_stat=t_stat, p_val=p_val,
                n_required=n_req, power_at_pilot=pwr_now,
                per_block=list(per_block),
                per_sub_diffs=per_sub_diffs)


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
    """所有條件跑完後印出彙整表格（含 CI）。"""
    print("\n" + "═"*120)
    print("  總結表格")
    print("  ※ Learning：斜率檢定（H0: slope=0），d_z = mean(slope)/std(slope)")
    print("  ※ Testing ：Task × Condition 交互作用，d_z = mean(interaction)/std(interaction)")
    print("  ※ CI 均為 95%（t 分布，兩尾）")
    print("═"*120)
    for res in all_results:
        if res is None:
            continue
        if res.get('type') == 'learning_slope':
            short = res['label'].replace('Learning | ', 'Lrn | ')
            print(f"  {short}")
            print(f"    slope={res['mean_slope']:+.5f} dB/step  "
                  f"d_z={res['d']:+.3f}  "
                  f"95% CI [{res['ci_lo']:+.5f}, {res['ci_hi']:+.5f}]  "
                  f"N_req={res['n_required']}  pwr={res['power_at_pilot']:.3f}")
        elif 'mean_interaction' in res:
            short = res['label'].replace('Testing | ', 'Test | ')
            print(f"  {short}")
            print(f"    Motor diff={res['mean_m_diff']:+.4f}  "
                  f"Percept diff={res['mean_p_diff']:+.4f}  "
                  f"Interaction={res['mean_interaction']:+.4f}  "
                  f"d_z={res['d']:+.3f}  "
                  f"95% CI [{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}]  "
                  f"N_req={res['n_required']}  pwr={res['power_at_pilot']:.3f}")
        print()
    print("═"*120)
    print("  ※ N required：one-tailed, α=0.05, power=0.80")
    print("═"*120 + "\n")


def plot_results(all_results, testing_results=None, learning_results=None):
    """
    視覺化結果。

    Figure 1：學習階段 ERSP 長條圖（RegH vs RandL，不變）
    Figure 2：測驗階段交互作用圖
      左：Motor ROI Theta — MotorTest vs PerceptualTest 各自的 RegH-RandL diff，
          加上交互作用分數與 d_z
      右：Perceptual ROI Alpha — 同上
    Figure 3：d_z 跟統計力彙整（學習 + 測驗交互作用）
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("⚠  matplotlib not found. Please run: pip install matplotlib")
        return

    valid_all = [r for r in all_results if r is not None]
    if not valid_all:
        print("⚠  No valid results to plot.")
        return

    C_REG  = '#2C6FAC'
    C_RND  = '#AAAAAA'
    C_MTR  = '#E74C3C'
    C_PRC  = '#27AE60'
    C_NEUT = '#888780'

    # ════════════════════════════════════════════════════════════
    # Figure 1：學習階段長條圖（RegH vs RandL，保留原本邏輯）
    # ════════════════════════════════════════════════════════════
    lrn_results = learning_results if learning_results else [r for r in valid_all if r and r.get('type')=='learning_slope']
    if lrn_results:
        fig1, axes = plt.subplots(1, len(lrn_results), figsize=(8*len(lrn_results), 5), sharey=False)
        if len(lrn_results) == 1:
            axes = [axes]
        fig1.suptitle(
            'Learning Phase: RegH - RandL Slope across Blocks\n'
            '(model changes across blocks; slope < 0 = increasing ERD for RegH)',
            fontsize=11, fontweight='bold')
        blk_labels = [b.replace('Block','Blk') for b in LEARNING_BLOCK_GROUPS]
        x_blk = np.arange(1, len(LEARNING_BLOCK_GROUPS)+1)
        for ax, res in zip(axes, lrn_results):
            if res is None: continue
            pb = np.array(res['per_block'])
            # 個別受試者折線（淡色）
            for sub_d in res['per_sub_diffs']:
                yd = np.array(sub_d)
                ax.plot(x_blk, yd, color='#AAAAAA', linewidth=0.8, alpha=0.5)
            # 群體均值折線
            ax.plot(x_blk, pb, 'o-', color=C_MTR, linewidth=2.2, markersize=7, label='Group mean')
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
            # 回歸線
            x_c = x_blk - x_blk.mean()
            y_fit = res['mean_slope'] * x_c + np.nanmean(pb)
            ax.plot(x_blk, y_fit, color=C_PRC, linewidth=1.8, linestyle='--', label=f"slope={res['mean_slope']:+.5f}")
            ax.set_xticks(x_blk)
            ax.set_xticklabels(blk_labels, fontsize=9)
            ax.set_ylabel('RegH - RandL ERSP (dB)')
            ax.set_title(res['label'].replace('Learning | ', ''), fontsize=9)
            ax.legend(fontsize=8)
            ax.text(0.5, 0.02,
                    f"d_z={res['d']:+.3f}  95% CI [{res['ci_lo']:+.5f}, {res['ci_hi']:+.5f}]  N_req={res['n_required']}",
                    transform=ax.transAxes, ha='center', fontsize=8.5,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        fig1.tight_layout()
        fig1.savefig('gpower_learning.png', dpi=150, bbox_inches='tight')
        print("  ✓ Figure 1 saved → gpower_learning.png")

    # ════════════════════════════════════════════════════════════
    # Figure 2：測驗階段 Task × Condition 交互作用
    # ════════════════════════════════════════════════════════════
    tst_results = [r for r in valid_all if 'mean_interaction' in r]
    if tst_results:
        fig2, axes2 = plt.subplots(1, len(tst_results), figsize=(8*len(tst_results), 5))
        if len(tst_results) == 1:
            axes2 = [axes2]
        fig2.suptitle(
            'Testing Phase: Task × Condition Interaction\n'
            '(MotorTest_diff − PerceptualTest_diff, where diff = RegH − RandL)',
            fontsize=11, fontweight='bold')
        x = np.array([0, 1])
        W = 0.35
        for ax, res in zip(axes2, tst_results):
            # 左右兩群：MotorTest 跟 PerceptualTest 各自的 RegH/RandL
            ax.bar(x[0]-W/2, res['motor_regh'], width=W, color=C_REG, label='Regular High', zorder=3)
            ax.bar(x[0]+W/2, res['motor_rand'], width=W, color=C_RND, label='Random Low',   zorder=3)
            ax.bar(x[1]-W/2, res['perc_regh'],  width=W, color=C_REG, zorder=3)
            ax.bar(x[1]+W/2, res['perc_rand'],  width=W, color=C_RND, zorder=3)
            ax.axhline(0, color='black', linewidth=0.8)
            # 連線顯示交互作用
            ax.plot([x[0]-W/2, x[1]-W/2],
                    [res['motor_regh'], res['perc_regh']],
                    'o--', color=C_REG, linewidth=1.5, zorder=4)
            ax.plot([x[0]+W/2, x[1]+W/2],
                    [res['motor_rand'], res['perc_rand']],
                    's--', color=C_RND, linewidth=1.5, zorder=4)
            ax.set_xticks(x)
            ax.set_xticklabels(['Motor Test', 'Perceptual Test'])
            ax.set_ylabel('ERSP (dB)')
            ax.set_title(res['label'].replace('Testing | ', ''), fontsize=9)
            ax.legend(loc='upper right', fontsize=8)
            ax.text(0.5, 0.02,
                    f"Interaction = {res['mean_interaction']:+.4f} dB\n"
                    f"d_z = {res['d']:+.3f}  N_req = {res['n_required']}  "
                    f"pwr@N=10 = {res['power_at_pilot']:.3f}",
                    transform=ax.transAxes, ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        fig2.tight_layout()
        fig2.savefig('gpower_testing_interaction.png', dpi=150, bbox_inches='tight')
        print("  ✓ Figure 2 saved → gpower_testing_interaction.png")

    try:
        plt.show()
    except Exception:
        pass


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

    # ── 學習階段：回歸斜率（model changes across blocks）─────
    print("━"*72)
    print("【學習階段】  Block7-11 ~ Block22-26（線性回歸斜率，模型化跨 block 趨勢）")
    print("  ★ 每位受試者用 block 序號（1~4）回歸 RegH-RandL ERSP 差值，取斜率")
    print("━"*72)
    learning_results = []
    for cfg in LEARNING_CONFIGS:
        slopes, per_sub_diffs, per_sub_means = run_learning(cfg)
        res = print_learning_slope_row(cfg['label'], slopes, per_sub_diffs, ALPHA, POWER_TARGET)
        learning_results.append(res)
        all_results.append(res)

    # ── 測驗階段：Task × Condition 交互作用 ───────────────────
    print("━"*72)
    print("【測驗階段】  Task × Condition 交互作用  (Motor − Perceptual) × (RegH − RandL)")
    print("  ★ G*Power 應基於此交互作用的 d_z，而非 Motor Test 或 Perceptual Test 各自的 d")
    print("━"*72)
    testing_results = []
    for cfg in TESTING_NEURAL_CONFIGS:
        (interactions,
         motor_regh, motor_rand,
         perc_regh,  perc_rand) = run_testing_interaction(cfg)
        res = print_interaction_row(cfg, interactions,
                                    motor_regh, motor_rand,
                                    perc_regh,  perc_rand)
        testing_results.append(res)
        all_results.append(res)

    # ── 彙整表格 ───────────────────────────────────────────────
    print_summary_table(all_results)

    # ── 視覺化 ─────────────────────────────────────────────────
    plot_results(all_results, testing_results, learning_results)
