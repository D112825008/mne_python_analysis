"""
使用者互動提示模組 - User Prompts Module

包含所有使用者輸入、驗證和格式化函數

所有函數均從 main.py 的 input() 和驗證邏輯中抽取
"""

import sys


def ask_save_confirmation(file_type='raw'):
    """
    詢問是否儲存檔案
    
    來源：main.py process_eeg_data() 第 2444-2445 行（選項 17-18）
    抽取：儲存確認邏輯
    
    Parameters
    ----------
    file_type : str
        檔案類型 ('raw' 或 'epochs')
    
    Returns
    -------
    bool
        True 表示要儲存，False 表示不儲存
    """
    prompt = f"\n是否要儲存 {file_type} 資料? (y/n) [預設 n]: "
    choice = input(prompt).strip().lower()
    return choice == 'y'


def ask_continue(message="是否繼續?"):
    """
    通用的繼續確認提示
    
    來源：main.py 多處使用的繼續確認邏輯
    抽取：通用確認邏輯
    
    Parameters
    ----------
    message : str
        提示訊息
    
    Returns
    -------
    bool
        True 表示繼續，False 表示取消
    """
    prompt = f"\n{message} (y/n) [預設 y]: "
    choice = input(prompt).strip().lower()
    return choice != 'n'


def validate_channels(input_channels, available_channels):
    """
    驗證電極名稱是否有效
    
    來源：main.py 多處使用的電極驗證邏輯
    抽取：電極驗證邏輯
    
    Parameters
    ----------
    input_channels : list
        使用者輸入的電極名稱列表
    available_channels : list
        可用的電極名稱列表
    
    Returns
    -------
    valid_channels : list
        有效的電極名稱列表
    invalid_channels : list
        無效的電極名稱列表
    """
    valid_channels = [ch for ch in input_channels if ch in available_channels]
    invalid_channels = [ch for ch in input_channels if ch not in available_channels]
    
    return valid_channels, invalid_channels


def validate_numeric_input(input_str, min_val=None, max_val=None):
    """
    驗證數值輸入
    
    來源：main.py 多處使用的數值驗證邏輯
    抽取：數值驗證邏輯
    
    Parameters
    ----------
    input_str : str
        使用者輸入的字串
    min_val : float or None
        最小值限制
    max_val : float or None
        最大值限制
    
    Returns
    -------
    float or None
        驗證後的數值，如果無效則返回 None
    """
    try:
        value = float(input_str)
        
        if min_val is not None and value < min_val:
            return None
        if max_val is not None and value > max_val:
            return None
        
        return value
    except ValueError:
        return None


def format_duration(seconds):
    """
    格式化時長顯示（HH:MM:SS）
    
    來源：main.py calculate_experiment_duration() 第 128-131 行
    抽取：時長格式化邏輯
    
    Parameters
    ----------
    seconds : float
        時長（秒）
    
    Returns
    -------
    str
        格式化的時長字串（HH:MM:SS）
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def ask_file_path(file_format):
    """
    詢問檔案路徑
    
    來源：main.py select_data_source() 第 269-272 行
    抽取：檔案路徑輸入邏輯
    
    Parameters
    ----------
    file_format : str
        檔案格式名稱
    
    Returns
    -------
    str
        使用者輸入的檔案路徑
    """
    if file_format == 'BIDS':
        path = input("\n請輸入 BIDS 資料夾路徑: ").strip()
    else:
        path = input(f"\n請輸入 {file_format} 檔案完整路徑: ").strip()
    
    return path


def ask_threshold(default=3):
    """
    詢問閾值設定
    
    來源：main.py asrt_artifact_rejection() 中的閾值輸入邏輯
    抽取：閾值輸入邏輯
    
    Parameters
    ----------
    default : float
        預設閾值
    
    Returns
    -------
    float
        使用者設定的閾值
    """
    while True:
        threshold_input = input(f"\n請設定閾值 (建議 3-5σ) [預設 {default}]: ").strip()
        
        if threshold_input == "":
            return float(default)
        
        try:
            threshold = float(threshold_input)
            if 0 < threshold <= 10:
                return threshold
            else:
                print("⚠️  閾值應在 0-10 之間，請重新輸入")
        except ValueError:
            print("⚠️  請輸入有效的數字")


def ask_filename(default_name, description='檔案'):
    """
    詢問儲存檔名，按 Enter 使用預設名稱
    """
    user_input = input(f"\n請輸入{description}檔名 [預設: {default_name}]: ").strip()
    return user_input if user_input else default_name


def ask_yes_no(prompt, default='n'):
    """
    通用的 Yes/No 提示
    
    來源：main.py 多處使用的 y/n 確認邏輯
    抽取：通用 y/n 確認邏輯
    
    Parameters
    ----------
    prompt : str
        提示訊息
    default : str
        預設選項 ('y' 或 'n')
    
    Returns
    -------
    bool
        True 表示 yes，False 表示 no
    """
    default_str = 'y' if default == 'y' else 'n'
    full_prompt = f"{prompt} (y/n) [預設 {default_str}]: "
    choice = input(full_prompt).strip().lower()
    
    if choice == "":
        return default == 'y'
    
    return choice == 'y'