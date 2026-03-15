"""
工具函數模組 - Utilities Module

包含各種工具函數和幫助函數。
"""

import matplotlib.pyplot as plt


def select_subject(data_dict):
    """
    顯示可用的受試者清單並讓使用者選擇。
    
    參數:
        data_dict (dict): 包含受試者資料的字典
        
    返回:
        str: 所選受試者的ID，如果取消則返回None
    """
    print("\n可用的受試者清單 Display the participant list")
    subjects = list(data_dict.keys())
    for i, subject in enumerate(subjects, 1):
        print(f"{i}. {subject}")

    while True:
        try:
            choice = input("\n請選擇要處理的受試者編號 (按q退出): ")
            if choice.lower() == 'q':
                return None

            choice = int(choice)
            if 1 <= choice <= len(subjects):
                return subjects[choice - 1]
            else:
                print("沒有該受試者，請重新輸入")

        except ValueError:
            print("請輸入有效的數字")


def plot_raw_data(raw, title=None, scalings=None, block=True):
    """
    繪製Raw數據。
    
    參數:
        raw (mne.io.Raw): 要繪製的Raw物件
        title (str): 標題
        scalings (dict): 縮放因子
        block (bool): 是否阻塞
    """
    if scalings is None:
        scalings = 'auto'
    
    fig = raw.plot(scalings=scalings, title=title, block=block)
    return fig


def plot_electrodes(raw, show_names=True, block=True):
    """
    繪製電極位置。
    
    參數:
        raw (mne.io.Raw): 包含電極位置的Raw物件
        show_names (bool): 是否顯示電極名稱
        block (bool): 是否阻塞
    """
    fig = raw.plot_sensors(show_names=show_names)
    plt.show(block=block)
    return fig


def plot_psd(raw, fmax=50, block=True):
    """
    繪製功率頻譜密度。
    
    參數:
        raw (mne.io.Raw): 要分析的Raw物件
        fmax (float): 最大頻率
        block (bool): 是否阻塞
    """
    fig = raw.plot_psd(fmax=fmax)
    plt.show(block=block)
    return fig


def set_matplotlib_properties(family='Microsoft JhengHei'):
    """
    設定Matplotlib字體。
    
    參數:
        family (str): 字體名稱
    """
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = [family]
    matplotlib.rcParams['axes.unicode_minus'] = False
    print(f"設定Matplotlib字體為: {family}")


def get_channels_by_regions(raw, region='frontal'):
    """
    根據腦區獲取通道。
    
    參數:
        raw (mne.io.Raw): 包含通道的Raw物件
        region (str): 腦區，可選'frontal', 'central', 'temporal', 'parietal', 'occipital'
        
    返回:
        list: 該腦區的通道名稱列表
    """
    # 通道-腦區映射
    region_mapping = {
        'frontal': ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8'],
        'central': ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'],
        'temporal': ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8'],
        'parietal': ['CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8'],
        'occipital': ['PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2']
    }
    
    if region not in region_mapping:
        raise ValueError(f"不支持的腦區: {region}")
    
    # 找出與raw.ch_names的交集
    available_channels = [ch for ch in region_mapping[region] if ch in raw.ch_names]
    
    return available_channels