import numpy as np
import matplotlib.pyplot as plt
import mne
from pathlib import Path

def plot_ersp_lum2023_style(power_dict, subject_id, output_dir):
    """
    繪製 ERSP 圖（Lum et al. 2023 風格）
    
    特色：
    - Theta 和 Alpha ROI 並排顯示
    - dB 轉換後的功率
    - 時頻 heatmap with contours
    - 事件時間點標記（time=0）
    """
    
    n_rois = len(power_dict)
    
    fig, axes = plt.subplots(1, n_rois, figsize=(6*n_rois, 5))
    if n_rois == 1:
        axes = [axes]
    
    for idx, (roi_name, data) in enumerate(power_dict.items()):
        ax = axes[idx]
        
        power = data['power']
        times = data['times']
        freqs = data['freqs']
        
        # === Heatmap with contours ===
        im = ax.contourf(
            times * 1000,  # 轉成毫秒
            freqs,
            power,
            levels=20,
            cmap='RdBu_r',
            extend='both'
        )
        
        # 標記 time=0（刺激或反應時間點）
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Event onset')
        
        # 設定座標軸
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        ax.set_title(f'{roi_name} ROI', fontsize=14, fontweight='bold')
        ax.set_ylim([freqs[0], 30])
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', fontsize=11)
    
    plt.suptitle(f'ERSP - {subject_id} (Dillian\'s Parameters)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # === 儲存圖片 ===
    import os
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{subject_id}_ersp_lum2023_style.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ ERSP 圖片已儲存: {output_path}")
    plt.close(fig)

def plot_ersp_comparison(power_by_condition, subject_id, condition_labels, output_dir):
    """
    繪製條件比較圖（例如 Regular vs Random）
    
    Layout:
    Row 1: Condition 1 (Theta, Alpha)
    Row 2: Condition 2 (Theta, Alpha)
    Row 3: Difference (Theta, Alpha)
    """
    from matplotlib.gridspec import GridSpec
    
    conditions = list(power_by_condition.keys())
    roi_names = list(power_by_condition[conditions[0]].keys())
    n_rois = len(roi_names)
    
    fig = plt.figure(figsize=(6*n_rois, 12))
    gs = GridSpec(3, n_rois, figure=fig, hspace=0.3, wspace=0.3)
    
    for col_idx, roi_name in enumerate(roi_names):
        
        # === Row 1: Condition 1 ===
        ax1 = fig.add_subplot(gs[0, col_idx])
        data1 = power_by_condition[conditions[0]][roi_name]
        im1 = ax1.contourf(data1['times'] * 1000, data1['freqs'], data1['power'],
                          levels=20, cmap='RdBu_r', extend='both')
        ax1.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax1.set_title(f'{condition_labels[0]}\n{roi_name} ROI', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency (Hz)', fontsize=11)
        ax1.set_ylim([data1['freqs'][0], 30])
        plt.colorbar(im1, ax=ax1, label='Power (dB)')
        
        # === Row 2: Condition 2 ===
        ax2 = fig.add_subplot(gs[1, col_idx])
        data2 = power_by_condition[conditions[1]][roi_name]
        im2 = ax2.contourf(data2['times'] * 1000, data2['freqs'], data2['power'],
                          levels=20, cmap='RdBu_r', extend='both')
        ax2.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax2.set_title(f'{condition_labels[1]}\n{roi_name} ROI', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency (Hz)', fontsize=11)
        ax2.set_ylim([data2['freqs'][0], 30])
        plt.colorbar(im2, ax=ax2, label='Power (dB)')
        
        # === Row 3: Difference (Condition 2 - Condition 1) ===
        ax3 = fig.add_subplot(gs[2, col_idx])
        diff_power = data2['power'] - data1['power']
        im3 = ax3.contourf(data1['times'] * 1000, data1['freqs'], diff_power,
                          levels=20, cmap='RdBu_r', extend='both')
        ax3.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax3.set_title(f'Difference\n{roi_name} ROI', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time (ms)', fontsize=11)
        ax3.set_ylabel('Frequency (Hz)', fontsize=11)
        ax3.set_ylim([data1['freqs'][0], 30])
        plt.colorbar(im3, ax=ax3, label='Power diff (dB)')
    
    plt.suptitle(f'ERSP Comparison - {subject_id}', fontsize=16, fontweight='bold', y=0.995)
    
    # 儲存
    import os
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{subject_id}_ersp_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 比較圖已儲存: {output_path}")
    plt.close(fig)


def plot_learning_comparison(results, subject_id, lock_type, output_dir, block_label=None):
    """
    繪製 Learning 階段比較圖 (Regular vs Random)
    
    Layout (2 ROIs x 3 rows):
    Row 1: Regular (Theta, Alpha)
    Row 2: Random (Theta, Alpha)
    Row 3: Difference (Theta, Alpha)
    """
    from matplotlib.gridspec import GridSpec
    import os
    
    roi_names = list(results['Regular'].keys())
    n_rois = len(roi_names)
    
    fig = plt.figure(figsize=(6*n_rois, 12))
    gs = GridSpec(3, n_rois, figure=fig, hspace=0.3, wspace=0.3)
    
    for col_idx, roi_name in enumerate(roi_names):
        
        regular_data = results['Regular'][roi_name]
        random_data = results['Random'][roi_name]
        
        # === Row 1: Regular ===
        ax1 = fig.add_subplot(gs[0, col_idx])
        im1 = ax1.contourf(
            regular_data['times'] * 1000, 
            regular_data['freqs'], 
            regular_data['power'],
            levels=20, cmap='RdBu_r', extend='both'
        )
        ax1.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax1.set_title(f'Regular\n{roi_name} ROI', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency (Hz)', fontsize=11)
        ax1.set_ylim([regular_data['freqs'][0], 30])
        plt.colorbar(im1, ax=ax1, label='Power (dB)')
        
        # === Row 2: Random ===
        ax2 = fig.add_subplot(gs[1, col_idx])
        im2 = ax2.contourf(
            random_data['times'] * 1000, 
            random_data['freqs'], 
            random_data['power'],
            levels=20, cmap='RdBu_r', extend='both'
        )
        ax2.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax2.set_title(f'Random\n{roi_name} ROI', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency (Hz)', fontsize=11)
        ax2.set_ylim([random_data['freqs'][0], 30])
        plt.colorbar(im2, ax=ax2, label='Power (dB)')
        
        # === Row 3: Difference (Random - Regular) ===
        ax3 = fig.add_subplot(gs[2, col_idx])
        diff_power = regular_data['power'] - random_data['power']
        im3 = ax3.contourf(
            regular_data['times'] * 1000, 
            regular_data['freqs'], 
            diff_power,
            levels=20, cmap='RdBu_r', extend='both'
        )
        ax3.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax3.set_title(f'Difference (Regular - Random)\n{roi_name} ROI',
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time (ms)', fontsize=11)
        ax3.set_ylabel('Frequency (Hz)', fontsize=11)
        ax3.set_ylim([regular_data['freqs'][0], 30])
        plt.colorbar(im3, ax=ax3, label='Power diff (dB)')
    
    _block_str = f" | {block_label}" if block_label else ""
    plt.suptitle(
        f'ERSP Comparison - {subject_id}\n{lock_type.capitalize()}-locked | Learning Phase{_block_str}', 
        fontsize=16, fontweight='bold', y=0.995
    )
    
    # 儲存
    os.makedirs(output_dir, exist_ok=True)
    _block_suffix = f"_{block_label}" if block_label else ""
    filename = f'{subject_id}_{lock_type}_lock_learning{_block_suffix}_ersp_comparison.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Learning 比較圖已儲存: {output_path}")
    plt.close(fig)


def plot_testing_comparison(test_results, subject_id, lock_type, test_type, output_dir, block_label=None):
    """
    繪製 Testing 階段單一 test_type 的 Regular vs Random 比較圖
    
    Layout (2 ROIs x 3 rows):
    Row 1: Regular (Theta, Alpha)
    Row 2: Random (Theta, Alpha)
    Row 3: Difference (Theta, Alpha)
    """
    from matplotlib.gridspec import GridSpec
    import os
    
    roi_names = list(test_results['Regular'].keys())
    n_rois = len(roi_names)
    
    fig = plt.figure(figsize=(6*n_rois, 12))
    gs = GridSpec(3, n_rois, figure=fig, hspace=0.3, wspace=0.3)
    
    for col_idx, roi_name in enumerate(roi_names):
        
        regular_data = test_results['Regular'][roi_name]
        random_data = test_results['Random'][roi_name]
        
        # === Row 1: Regular ===
        ax1 = fig.add_subplot(gs[0, col_idx])
        im1 = ax1.contourf(
            regular_data['times'] * 1000, 
            regular_data['freqs'], 
            regular_data['power'],
            levels=20, cmap='RdBu_r', extend='both'
        )
        ax1.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax1.set_title(f'Regular\n{roi_name} ROI', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency (Hz)', fontsize=11)
        ax1.set_ylim([regular_data['freqs'][0], 30])
        plt.colorbar(im1, ax=ax1, label='Power (dB)')
        
        # === Row 2: Random ===
        ax2 = fig.add_subplot(gs[1, col_idx])
        im2 = ax2.contourf(
            random_data['times'] * 1000, 
            random_data['freqs'], 
            random_data['power'],
            levels=20, cmap='RdBu_r', extend='both'
        )
        ax2.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax2.set_title(f'Random\n{roi_name} ROI', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency (Hz)', fontsize=11)
        ax2.set_ylim([random_data['freqs'][0], 30])
        plt.colorbar(im2, ax=ax2, label='Power (dB)')
        
        # === Row 3: Difference (Random - Regular) ===
        ax3 = fig.add_subplot(gs[2, col_idx])
        diff_power = regular_data['power'] - random_data['power']
        im3 = ax3.contourf(
            regular_data['times'] * 1000, 
            regular_data['freqs'], 
            diff_power,
            levels=20, cmap='RdBu_r', extend='both'
        )
        ax3.axvline(0, color='black', linestyle='--', linewidth=1.5)
        ax3.set_title(f'Difference (Regular - Random)\n{roi_name} ROI',
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time (ms)', fontsize=11)
        ax3.set_ylabel('Frequency (Hz)', fontsize=11)
        ax3.set_ylim([regular_data['freqs'][0], 30])
        plt.colorbar(im3, ax=ax3, label='Power diff (dB)')
    
    _block_str = f" | {block_label}" if block_label else ""
    plt.suptitle(
        f'ERSP Comparison - {subject_id}\n{lock_type.capitalize()}-locked | Testing: {test_type.capitalize()}{_block_str}', 
        fontsize=16, fontweight='bold', y=0.995
    )
    
    # 儲存
    os.makedirs(output_dir, exist_ok=True)
    _block_suffix = f"_{block_label}" if block_label else ""
    filename = f'{subject_id}_{lock_type}_lock_testing_{test_type}{_block_suffix}_ersp_comparison.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Testing {test_type.capitalize()} 比較圖已儲存: {output_path}")
    plt.close(fig)