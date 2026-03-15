"""
ICA分析模組 - ICA Analysis Module

包含ICA分析與互動式檢視功能，用於去除EEG雜訊。
版本: 4.0 (整合新版ICANavigator視窗與自動化分類)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, TextBox
import mne
import mne_icalabel


# ================================================================
# ICA Viewer - 新版多頁面互動式介面
# ================================================================
class ICANavigatorWindow:
    """
    ICA Component 導航視窗
    顯示單一component的詳細資料，可切換到前一個/後一個component，
    並支援多頁視圖 (Topography, Properties, Overlay, Sources, Scores)
    """

    def __init__(self, raw, ica, ic_labels=None, eog_scores=None):
        self.raw = raw
        self.ica = ica
        self.ic_labels = ic_labels
        self.current_idx = 0
        self.n_components = ica.n_components
        self.current_page = 0  # 0=第1頁（概覽）, 1=第2頁（詳細屬性）
        
        # 獲取變異量解釋率
        self.var_ratio = ica.get_explained_variance_ratio(raw)
        
        # 追蹤每個component的properties圖是否已開啟
        self.properties_figs = {}  # {component_idx: figure_object}
        self.overlay_figs = {}      # 第3頁：plot_overlay
        self.sources_figs = {}      # 第4頁：plot_sources
        self.scores_figs = {}       # 第5頁：plot_scores
        self.eog_scores = eog_scores  # EOG artifact分數
        
        # 創建主視窗
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('ICA Component Viewer - Page 1/5', fontsize=16, fontweight='bold')
        
        # 創建子圖區域
        gs = GridSpec(4, 2, figure=self.fig, hspace=0.4, wspace=0.3)
        
        # 地形圖 (左上)
        self.ax_topo = self.fig.add_subplot(gs[0, 0])
        
        # 時間序列 (右上，跨2列)
        self.ax_timeseries = self.fig.add_subplot(gs[0:2, 1])
        
        # 功率頻譜 (左中)
        self.ax_psd = self.fig.add_subplot(gs[1, 0])
        
        # 源波形 (中下，跨2欄)
        self.ax_source = self.fig.add_subplot(gs[2, :])
        
        # 資訊文字區域 (底部左側)
        self.ax_info = self.fig.add_subplot(gs[3, 0])
        self.ax_info.axis('off')
        
        # 按鈕區域 (底部右側)
        self.ax_buttons = self.fig.add_subplot(gs[3, 1])
        self.ax_buttons.axis('off')
        
        # 創建導航按鈕
        self._create_buttons()
        
        # 顯示第一個component
        self.update_display()
        
        plt.show(block=False)
    
    def _create_buttons(self):
        """創建導航按鈕（包含分頁功能）"""
        # Component 導航按鈕
        # 前一個成分
        ax_prev = plt.axes([0.28, 0.05, 0.07, 0.035])
        self.btn_prev = Button(ax_prev, 'Prev')
        self.btn_prev.on_clicked(self._prev_component)
        
        # Jump to component (TextBox) - 縮小寬度
        from matplotlib.widgets import TextBox
        ax_jump = plt.axes([0.38, 0.05, 0.05, 0.035])
        self.txt_jump = TextBox(ax_jump, 'IC:', initial='0')
        self.txt_jump.on_submit(self._jump_to_component)
        
        # 後一個成分
        ax_next = plt.axes([0.46, 0.05, 0.07, 0.035])
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self._next_component)
        
        # 分隔線（視覺上的分隔）
        
        # 頁面導航按鈕 - 調整尺寸
        # Previous page
        ax_prev_page = plt.axes([0.58, 0.05, 0.10, 0.035])
        self.btn_prev_page = Button(ax_prev_page, 'Prev Page')
        self.btn_prev_page.on_clicked(self._prev_page)
        
        # Next page
        ax_next_page = plt.axes([0.69, 0.05, 0.10, 0.035])
        self.btn_next_page = Button(ax_next_page, 'Next Page')
        self.btn_next_page.on_clicked(self._next_page)
        
        # 關閉按鈕
        ax_close = plt.axes([0.82, 0.05, 0.07, 0.035])
        self.btn_close = Button(ax_close, 'Close')
        self.btn_close.on_clicked(self._close_window)
    
    def _jump_to_component(self, text):
        try:
            comp_idx = int(text)
            if 0 <= comp_idx < self.n_components:
                self.current_idx = comp_idx
                self.update_display()
            else:
                print(f"Component index must be between 0 and {self.n_components-1}")
        except ValueError:
            print("Please enter a valid number")
    
    def _prev_component(self, event):
        """切換到前一個component"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
    
    def _next_component(self, event):
        """切換到後一個component"""
        if self.current_idx < self.n_components - 1:
            self.current_idx += 1
            self.update_display()
    
    def _prev_page(self, event):
        """切換到上一頁"""
        if self.current_page > 0:
            self.current_page -= 1
            self.update_display()
    
    def _next_page(self, event):
        """切換到下一頁"""
        if self.current_page < 4:  # 目前有5頁
            self.current_page += 1
            self.update_display()
    
    def _close_window(self, event):
        """關閉視窗"""
        plt.close(self.fig)
    
    def update_display(self):
        """更新顯示當前component的詳細資料（根據頁面）"""
        comp_idx = self.current_idx
        
        # 更新視窗標題（包含頁面資訊）
        self.fig.suptitle(f'ICA Component Viewer - Component {comp_idx}/{self.n_components-1} - Page {self.current_page+1}/5',
                         fontsize=16, fontweight='bold')
        
        if self.current_page == 0:
            # 第1頁：概覽（原有的4個子圖）
            self._display_page1(comp_idx)
        elif self.current_page == 1:
            # 第2頁：詳細屬性（MNE plot_properties）
            self._display_page2(comp_idx)
        elif self.current_page == 2:
            # 第3頁：疊加比較（plot_overlay）
            self._display_page3(comp_idx)
        elif self.current_page == 3:
            # 第4頁：源信號檢視（plot_sources）
            self._display_page4(comp_idx)
        elif self.current_page == 4:
            # 第5頁：分數視圖（plot_scores）
            self._display_page5(comp_idx)
        
        # 重繪
        self.fig.canvas.draw()
    
    def _display_page1(self, comp_idx):
        """顯示第1頁：概覽"""
        # 清除所有子圖
        self.ax_topo.clear()
        self.ax_timeseries.clear()
        self.ax_psd.clear()
        self.ax_source.clear()
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # 1. 地形圖
        self.ica.plot_components(
            picks=comp_idx,
            ch_type='eeg',
            axes=self.ax_topo,
            show=False
        )
        self.ax_topo.set_title(f'Component {comp_idx} Topography', fontweight='bold')
        
        # 2. 獲取component的時間序列數據
        sources = self.ica.get_sources(self.raw).get_data()
        times = self.ica.get_sources(self.raw).times
        
        # 只顯示前10秒的數據
        max_time = min(10, times[-1])
        mask = times <= max_time
        
        self.ax_timeseries.plot(times[mask], sources[comp_idx, mask])
        self.ax_timeseries.set_xlabel('Time (s)')
        self.ax_timeseries.set_ylabel('Amplitude')
        self.ax_timeseries.set_title('Time Series (First 10s)', fontweight='bold')
        self.ax_timeseries.grid(True, alpha=0.3)
        
        # 3. 功率頻譜圖
        from mne.time_frequency import psd_array_welch
        psds, freqs = psd_array_welch(
            sources[comp_idx:comp_idx+1, :],
            sfreq=self.raw.info['sfreq'],
            fmin=0.5,
            fmax=50,
            verbose=False
        )
        
        # effective window size
        if len(freqs) > 1:
            freq_resolution = freqs[1] - freqs[0]
            self.effective_window_size = 1.0 / freq_resolution
        else:
            self.effective_window_size = 0
        
        self.ax_psd.plot(freqs, 10 * np.log10(psds[0]))
        self.ax_psd.set_xlabel('Frequency (Hz)')
        self.ax_psd.set_ylabel('Power (dB)')
        self.ax_psd.set_title('Power Spectral Density', fontweight='bold')
        self.ax_psd.grid(True, alpha=0.3)
        
        # 4. 源波形
        max_source_time = min(30, times[-1])
        mask_source = times <= max_source_time
        
        self.ax_source.plot(times[mask_source], sources[comp_idx, mask_source])
        self.ax_source.set_xlabel('Time (s)')
        self.ax_source.set_ylabel('Amplitude')
        self.ax_source.set_title('Source Waveform (first 30s)', fontweight='bold')
        self.ax_source.grid(True, alpha=0.3)
        
        # 5. 顯示資訊文字
        info_text = self._get_component_info(comp_idx)
        
        # window size 加入顯示
        if hasattr(self, 'effective_window_size'):
            info_text += f"\n\nEffective Window Size:\n{self.effective_window_size:.3f} (s)"
            
        self.ax_info.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    def _display_page2(self, comp_idx):
        """顯示第2頁：詳細屬性（MNE plot_properties）"""
        # 清除所有子圖
        self.ax_topo.clear()
        self.ax_timeseries.clear()
        self.ax_psd.clear()
        self.ax_source.clear()
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # 顯示說明文字
        self.ax_topo.text(0.5, 0.5, 'Detailed Properties View\n(MNE plot_properties)\n\nOpening separate window...', 
                         ha='center', va='center', fontsize=14, 
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        self.ax_topo.axis('off')
        
        # 關閉其他子圖的坐標軸
        self.ax_timeseries.axis('off')
        self.ax_psd.axis('off')
        self.ax_source.axis('off')
        
        # 檢查當前component的properties圖是否已開啟
        fig_already_open = False
        if comp_idx in self.properties_figs:
            try:
                # 檢查figure是否還存在（沒有被關閉）
                fig = self.properties_figs[comp_idx]
                if isinstance(fig, list):
                    # plot_properties返回的是一個figure列表
                    # 檢查第一個figure是否還存在
                    if len(fig) > 0 and plt.fignum_exists(fig[0].number):
                        fig_already_open = True
                        # 將已存在的視窗提到前面
                        for f in fig:
                            if plt.fignum_exists(f.number):
                                f.canvas.manager.show()
                                f.canvas.draw()
                elif plt.fignum_exists(fig.number):
                    fig_already_open = True
                    fig.canvas.manager.show()
                    fig.canvas.draw()
            except:
                # 如果檢查時出錯，清除這個記錄
                del self.properties_figs[comp_idx]
                fig_already_open = False
        
        # 顯示資訊
        info_text = f"Component {comp_idx}\n\n"
        if fig_already_open:
            info_text += "Properties window already open!\n\n"
            info_text += "The existing window has been\nbrought to front.\n\n"
            info_text += "(Close it to create a new one\nnext time)"
        else:
            info_text += "Page 2 will automatically open\nMNE's plot_properties window\n\n"
            info_text += "Please check the pop-up window\nfor complete detailed analysis"
        
        self.ax_info.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
                         family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        # 只有在圖尚未開啟時才創建新的properties圖
        if not fig_already_open:
            try:
                # 調用 MNE 的 plot_properties
                fig_properties = self.ica.plot_properties(self.raw, picks=[comp_idx], 
                                                          psd_args={'fmax': 50}, reject=None, show=False)
                # 儲存figure引用
                self.properties_figs[comp_idx] = fig_properties
                plt.show(block=False)
                print(f"已為 Component {comp_idx} 開啟 properties 視窗")
            except Exception as e:
                print(f"Error displaying plot_properties: {str(e)}")
        else:
            print(f"Component {comp_idx} 的 properties 視窗已經開啟，已將其提到前面")
    
    def _display_page3(self, comp_idx):
        """顯示第3頁：疊加比較（MNE plot_overlay）"""
        # 清除所有子圖
        self.ax_topo.clear()
        self.ax_timeseries.clear()
        self.ax_psd.clear()
        self.ax_source.clear()
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # 顯示說明文字
        self.ax_topo.text(0.5, 0.5, 'Overlay Comparison View\n(MNE plot_overlay)\n\nOpening separate window...', 
                         ha='center', va='center', fontsize=14, 
                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        self.ax_topo.axis('off')
        
        # 關閉其他子圖的坐標軸
        self.ax_timeseries.axis('off')
        self.ax_psd.axis('off')
        self.ax_source.axis('off')
        
        # 檢查當前component的overlay圖是否已開啟
        fig_already_open = False
        if comp_idx in self.overlay_figs:
            try:
                # 檢查figure是否還存在（沒有被關閉）
                fig = self.overlay_figs[comp_idx]
                if isinstance(fig, list):
                    # plot_overlay返回的可能是一個figure列表
                    if len(fig) > 0 and plt.fignum_exists(fig[0].number):
                        fig_already_open = True
                        # 將已存在的視窗提到前面
                        for f in fig:
                            if plt.fignum_exists(f.number):
                                f.canvas.manager.show()
                                f.canvas.draw()
                elif plt.fignum_exists(fig.number):
                    fig_already_open = True
                    fig.canvas.manager.show()
                    fig.canvas.draw()
            except:
                # 如果檢查時出錯，清除這個記錄
                del self.overlay_figs[comp_idx]
                fig_already_open = False
        
        # 顯示資訊
        info_text = f"Component {comp_idx}\n\n"
        info_text += "=== Overlay Comparison ===\n\n"
        if fig_already_open:
            info_text += "Overlay window already open!\n\n"
            info_text += "The existing window has been\nbrought to front.\n\n"
            info_text += "(Close it to create a new one\nnext time)"
        else:
            info_text += "This view shows:\n"
            info_text += "• Red: Original signal\n"
            info_text += "• Black: Signal after\n  removing this component\n\n"
            info_text += "Use this to evaluate\nremoval effect"
        
        self.ax_info.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                         family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        # 只有在圖尚未開啟時才創建新的overlay圖
        if not fig_already_open:
            try:
                # 調用 MNE 的 plot_overlay
                fig_overlay = self.ica.plot_overlay(self.raw, picks=[comp_idx], show=False)
                # 儲存figure引用
                self.overlay_figs[comp_idx] = fig_overlay
                plt.show(block=False)
                print(f"已為 Component {comp_idx} 開啟 overlay 視窗")
            except Exception as e:
                print(f"Error displaying plot_overlay: {str(e)}")
                # 在資訊區顯示錯誤
                error_text = f"Error:\n{str(e)}"
                self.ax_info.clear()
                self.ax_info.axis('off')
                self.ax_info.text(0.1, 0.5, error_text, fontsize=10, verticalalignment='center',
                                 family='monospace', color='red')
        else:
            print(f"Component {comp_idx} 的 overlay 視窗已經開啟，已將其提到前面")
    
    def _display_page4(self, comp_idx):
        """顯示第4頁：源信號檢視（MNE plot_sources）"""
        # 清除所有子圖
        self.ax_topo.clear()
        self.ax_timeseries.clear()
        self.ax_psd.clear()
        self.ax_source.clear()
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # 顯示說明文字
        self.ax_topo.text(0.5, 0.5, 'Interactive Source View\n(MNE plot_sources)\n\nOpening separate window...', 
                         ha='center', va='center', fontsize=14, 
                         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        self.ax_topo.axis('off')
        
        # 關閉其他子圖的坐標軸
        self.ax_timeseries.axis('off')
        self.ax_psd.axis('off')
        self.ax_source.axis('off')
        
        # 檢查當前component的sources圖是否已開啟
        fig_already_open = False
        if comp_idx in self.sources_figs:
            try:
                # 檢查figure是否還存在（沒有被關閉）
                fig = self.sources_figs[comp_idx]
                if isinstance(fig, list):
                    # plot_sources返回的可能是一個figure列表
                    if len(fig) > 0 and plt.fignum_exists(fig[0].number):
                        fig_already_open = True
                        # 將已存在的視窗提到前面
                        for f in fig:
                            if plt.fignum_exists(f.number):
                                f.canvas.manager.show()
                                f.canvas.draw()
                elif plt.fignum_exists(fig.number):
                    fig_already_open = True
                    fig.canvas.manager.show()
                    fig.canvas.draw()
            except:
                # 如果檢查時出錯，清除這個記錄
                del self.sources_figs[comp_idx]
                fig_already_open = False
        
        # 顯示資訊
        info_text = f"Component {comp_idx}\n\n"
        info_text += "=== Interactive Sources ===\n\n"
        if fig_already_open:
            info_text += "Sources window already open!\n\n"
            info_text += "The existing window has been\nbrought to front.\n\n"
            info_text += "(Close it to create a new one\nnext time)"
        else:
            info_text += "This interactive view allows:\n"
            info_text += "• Scroll through time\n"
            info_text += "• Zoom in/out\n"
            info_text += "• Inspect entire recording\n\n"
            info_text += "Similar to raw.plot()\ninterface"
        
        self.ax_info.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                         family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        # 只有在圖尚未開啟時才創建新的sources圖
        if not fig_already_open:
            try:
                # 調用 MNE 的 plot_sources
                # 可以選擇顯示當前component或相鄰的幾個components
                # 這裡我們顯示當前component前後各1個（如果存在）
                picks_to_show = []
                if comp_idx > 0:
                    picks_to_show.append(comp_idx - 1)
                picks_to_show.append(comp_idx)
                if comp_idx < self.n_components - 1:
                    picks_to_show.append(comp_idx + 1)
                
                fig_sources = self.ica.plot_sources(self.raw, picks=picks_to_show, 
                                                    show=False, title=f'ICA Sources (focus on IC{comp_idx})')
                # 儲存figure引用
                self.sources_figs[comp_idx] = fig_sources
                plt.show(block=False)
                print(f"已為 Component {comp_idx} 開啟 sources 視窗（顯示 IC{picks_to_show}）")
            except Exception as e:
                print(f"Error displaying plot_sources: {str(e)}")
                # 在資訊區顯示錯誤
                error_text = f"Error:\n{str(e)}"
                self.ax_info.clear()
                self.ax_info.axis('off')
                self.ax_info.text(0.1, 0.5, error_text, fontsize=10, verticalalignment='center',
                                 family='monospace', color='red')
        else:
            print(f"Component {comp_idx} 的 sources 視窗已經開啟，已將其提到前面")
    
    def _display_page5(self, comp_idx):
        """顯示第5頁：分數視圖（MNE plot_scores）"""
        # 清除所有子圖
        self.ax_topo.clear()
        self.ax_timeseries.clear()
        self.ax_psd.clear()
        self.ax_source.clear()
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # 顯示說明文字
        self.ax_topo.text(0.5, 0.5, 'Artifact Detection Scores\n(MNE plot_scores)\n\nOpening separate window...', 
                         ha='center', va='center', fontsize=14, 
                         bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))
        self.ax_topo.axis('off')
        
        # 關閉其他子圖的坐標軸
        self.ax_timeseries.axis('off')
        self.ax_psd.axis('off')
        self.ax_source.axis('off')
        
        # 檢查是否有EOG scores資料
        if self.eog_scores is None:
            # 沒有scores資料，顯示警告
            info_text = f"Component {comp_idx}\n\n"
            info_text += "=== Artifact Scores ===\n\n"
            info_text += "⚠️ No EOG scores available\n\n"
            info_text += "EOG artifact detection\nwas not performed or\nno EOG channels found.\n\n"
            info_text += "This page requires EOG\nchannels in your data."
            
            self.ax_info.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                             family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
            print(f"無法顯示 Component {comp_idx} 的 scores：沒有EOG資料")
            return
        
        # 檢查當前component的scores圖是否已開啟
        fig_already_open = False
        if comp_idx in self.scores_figs:
            try:
                # 檢查figure是否還存在（沒有被關閉）
                fig = self.scores_figs[comp_idx]
                if isinstance(fig, list):
                    # plot_scores返回的可能是一個figure列表
                    if len(fig) > 0 and plt.fignum_exists(fig[0].number):
                        fig_already_open = True
                        # 將已存在的視窗提到前面
                        for f in fig:
                            if plt.fignum_exists(f.number):
                                f.canvas.manager.show()
                                f.canvas.draw()
                elif plt.fignum_exists(fig.number):
                    fig_already_open = True
                    fig.canvas.manager.show()
                    fig.canvas.draw()
            except:
                # 如果檢查時出錯，清除這個記錄
                del self.scores_figs[comp_idx]
                fig_already_open = False
        
        # 顯示資訊
        info_text = f"Component {comp_idx}\n\n"
        info_text += "=== EOG Artifact Scores ===\n\n"
        if fig_already_open:
            info_text += "Scores window already open!\n\n"
            info_text += "The existing window has been\nbrought to front.\n\n"
            info_text += "(Close it to create a new one\nnext time)"
        else:
            info_text += "This view shows:\n"
            info_text += "• EOG correlation scores\n"
            info_text += "• Artifact detection\n"
            info_text += "• Statistical analysis\n\n"
            info_text += "Higher scores indicate\nstronger EOG correlation"
            
            # 加入當前component的分數資訊
            if comp_idx < len(self.eog_scores):
                score_value = self.eog_scores[comp_idx]
                info_text += f"\n\nCurrent IC Score:\n{score_value:.4f}"
        
        self.ax_info.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                         family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        # 只有在圖尚未開啟時才創建新的scores圖
        if not fig_already_open:
            try:
                # 調用 MNE 的 plot_scores
                fig_scores = self.ica.plot_scores(self.eog_scores, picks=[comp_idx], 
                                                  show=False, title='EOG Artifact Scores')
                # 儲存figure引用
                self.scores_figs[comp_idx] = fig_scores
                plt.show(block=False)
                print(f"已為 Component {comp_idx} 開啟 scores 視窗（EOG score: {self.eog_scores[comp_idx]:.4f}）")
            except Exception as e:
                print(f"Error displaying plot_scores: {str(e)}")
                # 在資訊區顯示錯誤
                error_text = f"Error:\n{str(e)}"
                self.ax_info.clear()
                self.ax_info.axis('off')
                self.ax_info.text(0.1, 0.5, error_text, fontsize=10, verticalalignment='center',
                                 family='monospace', color='red')
        else:
            print(f"Component {comp_idx} 的 scores 視窗已經開啟，已將其提到前面")
    
    def _get_component_info(self, comp_idx):
        """獲取component的資訊文字"""
        info_lines = [f"Component: {comp_idx}"]
        
        # 變異量解釋率
        if isinstance(self.var_ratio, dict) and comp_idx in self.var_ratio:
            var = self.var_ratio[comp_idx]
            info_lines.append(f"Explain Varible: {var:.4f} ({var*100:.2f}%)")
        
        # ICLabel分類
        if self.ic_labels is not None:
            if 'labels' in self.ic_labels and len(self.ic_labels['labels']) > comp_idx:
                label = self.ic_labels['labels'][comp_idx]
                info_lines.append(f"ICLabel: {label}")
                
                if 'y_pred_proba' in self.ic_labels and len(self.ic_labels['y_pred_proba']) > comp_idx:
                    proba = self.ic_labels['y_pred_proba'][comp_idx]
                    # 檢查proba是否為陣列還是標量
                    if isinstance(proba, (list, np.ndarray)) and len(np.array(proba).shape) > 0:
                        max_idx = np.argmax(proba)
                        max_proba = proba[max_idx]
                        info_lines.append(f"Probability: {max_proba:.2%}")
                    elif np.isscalar(proba):
                        # 如果是標量，直接使用
                        info_lines.append(f"Probability: {proba:.2%}")
        
        return '\n'.join(info_lines)
        

def perform_ica(raw):
    """執行ICA分析並讓使用者選擇要排除的成分"""
    
    # === 新增：顯示當前電極資訊 ===
    print("\n" + "="*60)
    print("ICA 分析準備")
    print("="*60)
    
    # 顯示所有電極
    print(f"\n當前電極資訊 (共 {len(raw.ch_names)} 個)：")
    
    # 分類顯示電極
    eeg_channels = mne.pick_types(raw.info, eeg=True, eog=False, exclude=[])
    eog_channels = mne.pick_types(raw.info, eeg=False, eog=True, exclude=[])
    other_channels = [i for i in range(len(raw.ch_names)) 
                     if i not in eeg_channels and i not in eog_channels]
    
    print(f"\nEEG 電極 ({len(eeg_channels)} 個):")
    eeg_names = [raw.ch_names[i] for i in eeg_channels]
    for i in range(0, len(eeg_names), 8):
        print("  " + ", ".join(eeg_names[i:i+8]))
    
    if len(eog_channels) > 0:
        print(f"\nEOG 電極 ({len(eog_channels)} 個):")
        eog_names = [raw.ch_names[i] for i in eog_channels]
        print("  " + ", ".join(eog_names))
    else:
        print("\n⚠️  未檢測到 EOG 電極")
    
    if len(other_channels) > 0:
        print(f"\n其他電極 ({len(other_channels)} 個):")
        other_names = [raw.ch_names[i] for i in other_channels]
        print("  " + ", ".join(other_names))
    
    # 顯示當前的壞通道
    if raw.info['bads']:
        print(f"\n當前標記的壞通道: {', '.join(raw.info['bads'])}")
    else:
        print("\n當前沒有標記壞通道")
    
    # === 新增：詢問是否排除某些電極 ===
    print("\n" + "="*60)
    print("選擇參與 ICA 的電極")
    print("="*60)
    print("\n說明：")
    print("  - ICA 通常使用所有 EEG 電極")
    print("  - 可以排除外圍電極（如：Fp1, Fp2, F7, F8）以減少 artifact 影響")
    print("  - EOG 電極通常用於偵測眼動，但不參與 ICA 分解")
    print("  - 壞通道會自動排除")
    
    exclude_from_ica = input("\n是否要排除某些電極不參與 ICA? (y/n) [預設 n]: ").strip().lower()
    
    excluded_channels = []
    if exclude_from_ica == 'y':
        print("\n建議排除的電極類型：")
        print("  - 外圍電極：Fp1, Fp2, F7, F8, T7, T8, P7, P8, O1, O2")
        print("  - 或自定義")
        
        exclude_choice = input("\n請選擇:\n1. 排除外圍電極\n2. 手動輸入要排除的電極\n3. 不排除\n請輸入 (1/2/3) [預設 3]: ").strip()
        
        if exclude_choice == '1':
            # 預定義的外圍電極
            peripheral = ['Fp1', 'Fp2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'O1', 'O2']
            excluded_channels = [ch for ch in peripheral if ch in raw.ch_names]
            if excluded_channels:
                print(f"\n將排除: {', '.join(excluded_channels)}")
            else:
                print("\n⚠️  未找到預定義的外圍電極")
        
        elif exclude_choice == '2':
            exclude_input = input("\n請輸入要排除的電極名稱（用空格分隔）: ").strip()
            if exclude_input:
                excluded_channels = exclude_input.split()
                valid_excluded = [ch for ch in excluded_channels if ch in raw.ch_names]
                invalid_excluded = [ch for ch in excluded_channels if ch not in raw.ch_names]
                
                if valid_excluded:
                    excluded_channels = valid_excluded
                    print(f"\n將排除: {', '.join(excluded_channels)}")
                
                if invalid_excluded:
                    print(f"\n⚠️  以下電極不存在: {', '.join(invalid_excluded)}")
    
    # 首先進行資料處理
    raw_copy = raw.copy()
    raw_copy.set_eeg_reference('average')
    
    # === 詢問使用者關於濾波的選擇 ===
    current_highpass = raw_copy.info.get('highpass', 0)
    current_lowpass = raw_copy.info.get('lowpass', None)
    print(f"\n當前資料濾波: High-pass {current_highpass} Hz, Low-pass {current_lowpass} Hz")
    
    # 詢問使用者
    if current_highpass < 1.0:
        print("\n" + "="*60)
        print("ICA Fitting 濾波設定")
        print("="*60)
        print("\n說明：")
        print("  1. 使用當前濾波進行 ICA (較快，但品質可能較差)")
        print("  2. 創建 1-100 Hz 副本用於 ICA fitting (推薦，品質較好)")
        print("\n選項 2 說明:")
        print("  - 在 1-100 Hz 副本上進行 ICA fitting")
        print("  - 但最終資料仍保持原始濾波設定")
        print("  - 這樣既保證 ICA 品質，又保留低頻資訊")
        
        choice = input("\n請選擇 (1/2) [預設 2]: ").strip()
        
        if choice == '2' or choice == '':
            print("\n✓ 將創建 1-100 Hz 副本用於 ICA fitting")
            use_highpass_copy = True
        else:
            print("\n✓ 使用當前濾波進行 ICA")
            use_highpass_copy = False
    
    else:
        print(f"\n✓ 當前 high-pass {current_highpass} Hz 已符合 ICA 要求")
        use_highpass_copy = False
    
    # === 修正：確定參與 ICA 的電極 ===
    # 排除：壞通道 + 使用者指定的電極 + EOG 電極（用於偵測，不參與分解）
    picks = mne.pick_types(raw_copy.info,
                          eeg=True,
                          eog=False,  # EOG 不參與 ICA
                          exclude='bads')
    
    # 進一步排除使用者指定的電極
    if excluded_channels:
        picks = [p for p in picks if raw_copy.ch_names[p] not in excluded_channels]
    
    print(f"\n✓ 實際參與 ICA 的電極數: {len(picks)}")
    print(f"  參與 ICA: {', '.join([raw_copy.ch_names[p] for p in picks[:10]])}...")
    
    # 讓使用者決定要使用的ICA component數量
    n_channels = len(picks)
    max_safe = n_channels - 1  # 至少少 1 個以避免數值問題
    recommended = min(max_safe, int(n_channels * 0.95))

    print(f"\n目前參與 ICA 的電極數: {n_channels}")
    print(f"\n建議 component 數量範圍:")
    print(f"  • 保守 (70%): {int(n_channels * 0.7)}")
    print(f"  • 平衡 (85%): {int(n_channels * 0.85)}")
    print(f"  • 積極 (95%): {recommended} ← 推薦")
    print(f"  • 最大值:     {max_safe} (避免數值不穩定)")
    print(f"\n⚠️  注意：使用 {n_channels} 可能導致混合矩陣不穩定")

    while True:
        try:
            n_components_input = input(f"\n請輸入 component 數量 [預設: {recommended}]: ").strip()
            
            if n_components_input == "":
                n_components = recommended
                break
            else:
                n_components = int(n_components_input)
                if 0 < n_components < n_channels:  # 修改：< 而非 <=
                    break
                elif n_components == n_channels:
                    confirm = input(f"\n⚠️  使用 {n_channels} 可能不穩定，確定繼續？(y/n): ").strip().lower()
                    if confirm == 'y':
                        break
                    else:
                        continue
                else:
                    print(f"⚠️  component數必須在 1 到 {n_channels-1} 之間")
        except ValueError:
            print("⚠️  請輸入有效的數字")
        
        print(f"\n執行 ICA 分析 (使用 {n_components} 個 components)...")
    
    # === 根據使用者選擇決定是否創建High pass副本 ===
    if use_highpass_copy:
        # 創建副本並濾波到 1-100 Hz
        print("\n創建 1-100 Hz 副本用於 ICA fitting...")
        raw_for_fitting = raw_copy.copy()
        raw_for_fitting.filter(l_freq=1, h_freq=100, picks=picks)
        raw_for_iclabel = raw_for_fitting
    else:
        # 直接使用當前濾波
        raw_for_fitting = raw_copy
        raw_for_iclabel = raw_copy
    
    # 應用高通濾波器(1Hz)來改善ICA效果
#    raw_copy.filter(l_freq=1, h_freq=100, picks=picks)
    
    # Setting ICA parameter
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=42,
        method='picard',
        fit_params=dict(ortho=False, extended=True), 
        max_iter=800,
    )
    
    
    
    # 直接使用已經過濾的腦波 Use EEG wave which already do the ica analysis
    reject = dict(eeg=500e-6)  # 500 ÂµV threshold
    ica.fit(raw_for_fitting, picks=picks, reject=reject)
    print("✓ ICA fitting 完成")
    
    # 計算EOG artifact分數（如果有EOG通道且參與了ICA）
    eog_scores = None
    try:
        # 檢查是否有EOG通道
        eog_channels = mne.pick_types(raw_copy.info, eeg=False, eog=True)
        
        # ========== 修正：檢查 EOG 通道是否參與了 ICA ==========
        eog_in_ica = False
        eog_in_ica_names = []
        
        if len(eog_channels) > 0:
            # 檢查 EOG 通道是否在 picks 中（參與了 ICA）
            for ch_idx in eog_channels:
                ch_name = raw_copy.ch_names[ch_idx]
                if ch_idx in picks:
                    eog_in_ica = True
                    eog_in_ica_names.append(ch_name)
            
            if eog_in_ica:
                print(f"\n找到 {len(eog_in_ica_names)} 個參與ICA的EOG通道: {eog_in_ica_names}")
                print("計算EOG artifact分數...")
                eog_scores = ica.score_sources(raw_copy, target='eog', score_func='correlation')
                print(f"✓ EOG分數計算完成")
                
                # 顯示高分的components
                threshold = 0.3
                high_score_comps = [i for i, score in enumerate(eog_scores) if abs(score) > threshold]
                if high_score_comps:
                    print(f"  → EOG相關性較高的components (|score| > {threshold}): {high_score_comps}")
            else:
                print(f"\n⚠️  偵測到 {len(eog_channels)} 個EOG通道，但未參與ICA")
                print("  跳過EOG artifact自動偵測（您可稍後手動標記眼動components）")
        else:
            print("\n⚠️  未找到EOG通道，跳過EOG artifact偵測")
        # =====================================================
        
    except Exception as e:
        print(f"\n❌ 計算EOG分數時發生錯誤: {str(e)}")
        print("  提示：可能是 EOG 通道未參與 ICA 或資料格式問題")
        eog_scores = None
    
    # 使用ICLabel進行自動分類
    ic_labels = mne_icalabel.label_components(raw_for_iclabel, ica, method="iclabel")
    
    
    # 檢查ICLabel的用字
    #print("ICLabel 字典中的鍵:", list(ic_labels.keys()))
    #print("labels 的類型:", type(ic_labels["labels"]))
    #print("labels 的內容:", ic_labels["labels"])
    #print("y_pred_proba 的類型:", type(ic_labels["y_pred_proba"]))
    #print("y_pred_proba 的內容:", ic_labels["y_pred_proba"][:5])
    
    # 取得ICLabel分類結果
    labels = ic_labels["labels"]
    probabilities =ic_labels["y_pred_proba"]
    
    
    # 印出ICLabel分類結果
    print("\nICLabel分類結果 ICLabel classification results:")
    for i, label in enumerate(labels):
        print(f"IC {i}: {label}")
        
    # 自動辨識眼動artifacts跟肌電artifacts
    eye_idx = [i for i, label in enumerate(labels) if 'eye' in label.lower()]
    muscle_idx = [i for i, label in enumerate(labels) if 'muscle' in label.lower()]
    
    # 顯示自動檢測的結果
    if eye_idx:
        print(f"\n自動檢測到的眼動相關成分 Auto-detected eye components: {eye_idx}")
    if muscle_idx:
        print(f"\n自動檢測到的肌肉相關成分 Auto-detected muscle components: {muscle_idx}")
        
   # 使用互動式ICA檢視器 - 導航視窗模式
    print("\n啟動互動式ICA檢視器（導航視窗模式）...")
    print("使用「前一個」「後一個」按鈕來切換component")
    
    # 初始化 Navigator（會在循環中檢查是否需要重新創建）
    navigator = None
      
    while True:
        # 檢查 Navigator 是否存在或已被關閉，如果需要就重新創建
        if navigator is None or not plt.fignum_exists(navigator.fig.number):
            print("創建/重新創建 ICA Navigator 視窗...")
            navigator = ICANavigatorWindow(raw_copy, ica, ic_labels, eog_scores)
        
        # 顯示ICA Display ICA components
        print("\n顯示ICA圖片... Display ICA plot...")
        fig = ica.plot_components(
            picks=range(n_components),
            ch_type='eeg',
            title='ICA components',
            show=False
        )
        plt.show(block=False)
        
        # ===== 已整合到 ICANavigatorWindow 的第2頁，此處不再需要獨立呼叫 =====
        print("\nTip: Use the 'Next Page ▶' button in ICA Navigator to view detailed properties")
        print("\nICA 成分解釋方差比例:")
        print(f"實際使用的 ICA 成分數: {ica.n_components}")
    
         # 修正：使用 PCA 解釋變異量
        if hasattr(ica, 'pca_explained_variance_'):
            pca_var = ica.pca_explained_variance_
            total_var = np.sum(pca_var)
        
            # 計算每個 component 的變異量比例
            var_ratio_list = [(i, pca_var[i] / total_var) for i in range(len(pca_var))]
            var_ratio_list.sort(key=lambda x: x[1], reverse=True)
        
            # 顯示排序後的變異量
            print("\n依變異量排序的 ICA 成分:")
            total_variance = 0.0
            for comp_idx, ratio in var_ratio_list:
                print(f"Component {comp_idx}: {ratio:.3f}")
                total_variance += ratio
            print(f"\n總解釋方差比例: {total_variance:.3f}")  # 應該接近 1.0
        
            # 建立累積變異量圖表
            cumulative_var = np.cumsum([ratio for _, ratio in var_ratio_list])
        
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(var_ratio_list)), [ratio for _, ratio in var_ratio_list])
            plt.plot(range(len(var_ratio_list)), cumulative_var, 'r-', marker='o')
            plt.axhline(y=0.95, color='g', linestyle='--')
            plt.xlabel('ICA Component Number')
            plt.ylabel('Explained Variance Ratio')
            plt.title('ICA Component Explained Variance Ratio')
            plt.legend(['Cumulative Variance', 'Individual Variance', '95% Variance'])
            plt.tight_layout()
            plt.show(block=False)
        
        else:
            print("⚠️  無法取得 PCA 解釋變異量資訊")
            var_ratio_list = []
            
        # ===== 新增：等待用戶確認已查看完畢 =====
        print("\n" + "="*70)
        print("Please review the ICA Navigator window and charts above.")
        print("When you're ready to select components to exclude, press ENTER to continue...")
        print("="*70)
        input()  # 等待用戶按 ENTER
        # ===== 等待確認結束 =====
 
        # 讓使用者選擇要排除的ICA components Make user select which component want to delete 
        print("\n請選擇要排除的成分 Select which component you want to delete:")
        print("輸入格式:用逗號分開數字(例如:2,3,11) Input formal: Use comma as the interval to separate the number(e.g.2,3,11)")
        print("輸入 'skip'跳過ICA處理 Input 'skip' ICA analysis")
        print("輸入 'show'重新顯示ICA components Input 'show' re-display ICA components")
        print("輸入 'auto'使用IClabel自動檢測到的眼動和肌電成分 (如果有檢測到的話) Input 'auto' use the auto detected eog components (if have)")
        print("Tip: Use the 'Next Page' button in ICA Navigator to view detailed properties of each component")
        
        choice = input("請輸入:").strip().lower()
        
        # 處理空輸入
        if choice == '':
            print("未輸入任何內容，請重新輸入 No input detected, please try again")
            continue
        
        if choice == 'skip':
            return raw
        elif choice == 'show':
            continue
        elif choice == 'auto':
            exclude_idx = list(set(eye_idx + muscle_idx))
            if exclude_idx:
                print(f"將自動排除以下IClabel檢測到的成份: {exclude_idx}")
            else:
                print("ICLabel未檢測到明顯的artifact成份，請手動選擇要排除的成分")
                continue
        else:
            try:
                # 排除指定的components Delete the component you want
                exclude_idx = [int(x.strip()) for x in choice.split(',') if x.strip()]
                
                # 檢查是否有輸入有效的成分編號
                if not exclude_idx:
                    print("未輸入有效的成分編號，請重新輸入 No valid component numbers entered, please try again")
                    continue
                    
                # 檢查輸入的編號是否在有效範圍內
                invalid_idx = [idx for idx in exclude_idx if idx < 0 or idx >= n_components]
                if invalid_idx:
                    print(f"以下編號超出範圍 (0-{n_components-1}): {invalid_idx}")
                    print("請重新輸入 Please try again")
                    continue
                    
            except ValueError:
                print("輸入格式錯誤，請重新輸入 Input wrong format, please input again")
                continue
        
        # 確認選擇 Confirm the selection
        print(f"\n選擇要排除的components Select the component you want to delete:{exclude_idx}")
        confirm = input("確認要排除指定的components嗎? Are you sure you want to delete this components?(y/n):").lower()
            
        if confirm == 'y':
            # 應用 ICA apply ICA
            ica.exclude = exclude_idx
            raw_cleaned = raw.copy()
            ica.apply(raw_cleaned)
            print("ICA 處理完成! ICA finish!")
            
            if use_highpass_copy:
                print(f"✓ 最終資料保持原始濾波: High-pass {current_highpass} Hz")
                
            # 顯示處理前跟處理後的比較 Display before and after ica analysis
            print("\n顯示處理前後的比較圖 Display before and after ica analysis...")
            fig = mne.viz.plot_ica_overlay(ica, raw)
            plt.show(block=True)
            
            # 儲存ICLabel分類結果
            icalabel_results = {
                'IC_labels': labels,
                'IC_probabilities': probabilities.tolist(),
                'eye_components': eye_idx,
                'muscle_components': muscle_idx
                }
            
            # 儲存ICA資訊前確保variance_ratio是數值
            safe_variance_ratios = {}
            if var_ratio_list:
                for comp_idx, ratio in var_ratio_list:
                    safe_variance_ratios[comp_idx] = float(ratio)
            
            # 儲存ICA資訊 Save ICA information
            ica_info = {
                'n_components': n_components,
                'excluded_components': exclude_idx,
                'variance_explained': safe_variance_ratios,
                'iclabel_results': icalabel_results
            }

            # === 儲存 ICA 後的資料 ===
            save_choice = input("\n是否要儲存 ICA 處理後的資料? (y/n) [預設 n]: ").strip().lower()
            if save_choice == 'y':
                print("\n請選擇儲存格式:")
                print("1. FIF (.fif) - MNE-Python 原生格式")
                print("2. MAT (.mat) - MATLAB 格式")
                print("3. FIF + MAT 都儲存")
                fmt_choice = input("\n請選擇 (1/2/3) [預設 1]: ").strip()

                import os, scipy.io
                formats = ['mat'] if fmt_choice == '2' else (['fif', 'mat'] if fmt_choice == '3' else ['fif'])

                if 'fif' in formats:
                    default_fif = 'ica_cleaned-raw.fif'
                    fif_file = input(f"\n請輸入 FIF 檔名 [預設: {default_fif}]: ").strip() or default_fif
                    if not (fif_file.endswith('.fif') or fif_file.endswith('.fif.gz')):
                        fif_file += '.fif'
                    raw_cleaned.save(fif_file, overwrite=True)
                    print(f"✓ 已儲存: {fif_file}")

                if 'mat' in formats:
                    default_mat = 'ica_cleaned-raw.mat'
                    mat_file = input(f"\n請輸入 MAT 檔名 [預設: {default_mat}]: ").strip() or default_mat
                    if not mat_file.endswith('.mat'):
                        mat_file += '.mat'
                    scipy.io.savemat(mat_file, {
                        'data': raw_cleaned.get_data(),
                        'ch_names': raw_cleaned.ch_names,
                        'sfreq': raw_cleaned.info['sfreq'],
                        'times': raw_cleaned.times,
                        'excluded_components': exclude_idx,
                    })
                    print(f"✓ 已儲存: {mat_file}")

            return raw_cleaned, ica_info # 不返回 raw_copy