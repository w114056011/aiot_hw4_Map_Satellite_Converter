# AIOT HW4 - 地圖與衛星圖像轉換器專案報告

## 1. 專案概述 (Project Overview)
本專案實作了一個基於深度學習的圖像轉換應用程式，能夠在「地圖 (Map)」與「衛星圖像 (Satellite)」之間進行風格轉換。系統核心採用 **CycleGAN** 模型，這是一種非配對 (Unpaired) 的圖像對圖像轉換技術，特別適合處理地圖與衛星圖這類難以取得完美像素級配對資料的任務。

應用程式使用 **Streamlit** 框架建構，並整合了 **Folium** 互動式地圖，提供使用者友善的介面來選取地理區域並進行即時轉換。

## 2. 系統架構 (System Architecture)

### 2.1 技術堆疊 (Tech Stack)
- **前端介面**: Streamlit
- **深度學習框架**: PyTorch
- **地圖整合**: Folium, streamlit-folium
- **圖像處理**: Pillow (PIL), NumPy
- **模型來源**: pytorch-CycleGAN-and-pix2pix (Berkeley AI Research)

### 2.2 模型架構 (Model Architecture)
本專案使用 CycleGAN 的生成器 (Generator) 進行推論。
- **架構類型**: ResNet-based Generator
- **網路深度**: 9 個 ResNet Blocks (適用於 256x256 解析度)
- **關鍵組件**:
    - **Downsampling**: 2 層卷積層，將特徵圖縮小。
    - **ResNet Blocks**: 9 層殘差塊，負責特徵轉換與風格遷移。
    - **Upsampling**: 2 層轉置卷積層 (Transposed Conv)，將圖像還原至原始尺寸。
- **特殊配置**: 
    - 為了相容於 Berkeley 釋出的預訓練權重 (`map2sat.pth`, `sat2map.pth`)，模型中的 Normalization 層被特別調整為 `BatchNorm2d(affine=False, track_running_stats=True)`，而非標準 CycleGAN 常見的 InstanceNorm。

## 3. 主要功能 (Key Features)

### 3.1 雙向轉換
支援兩種轉換模式：
1. **地圖轉衛星 (Map → Satellite)**: 將抽象的線條地圖轉換為擬真的衛星空照圖。
2. **衛星轉地圖 (Satellite → Map)**: 將衛星照片轉換為結構化的地圖樣式。

### 3.2 互動式地圖整合
為了解決使用者難以取得高品質輸入圖源的問題，系統內建互動式地圖功能：
- **多圖層支援**: 整合 Google Maps、OpenStreetMap、Esri World Imagery 等圖層。
- **無標籤優化**: 特別加入 **CartoDB No Labels** 與 **Esri 街道圖 (少標註)**，避免地名文字干擾 GAN 模型的生成效果。
- **即時預覽**: 使用者可在地圖上導航至感興趣的區域。

### 3.3 自動化模型管理
- 系統會自動檢查 `checkpoints/` 目錄下是否存在所需的預訓練模型。
- 若模型缺失，會自動從 `efrosgans.eecs.berkeley.edu` 下載，無需手動配置。

### 3.4 圖像處理管線
1. **預處理 (Preprocessing)**:
    - 自動裁切圖像中央的正方形區域。
    - 縮放至模型輸入尺寸 (256x256)。
    - 正規化像素值至 [-1, 1] 區間。
2. **推論 (Inference)**:
    - 支援 GPU (CUDA) 加速，若無 GPU 則自動切換至 CPU。
3. **後處理 (Postprocessing)**:
    - 反正規化並轉換回 RGB 圖像格式供下載。

## 4. 實作細節與挑戰 (Implementation Details & Challenges)

### 4.1 模型權重相容性問題
**挑戰**: 原始 CycleGAN 論文通常使用 Instance Normalization，但官方釋出的 `.pth` 預訓練權重在載入時出現 `state_dict` 鍵值不匹配的問題 (Unexpected keys)。
**解決方案**: 經過分析權重檔結構，發現該預訓練模型在訓練時使用的是 Batch Normalization 且未啟用可學習參數 (`affine=False`)。因此，我們在 `ResnetBlock` 與 `ResnetGenerator` 中重寫了網路定義，精確匹配了預訓練權重的結構，成功解決載入錯誤。

### 4.2 文字標籤干擾
**挑戰**: 一般 Google Maps 截圖包含大量文字 (路名、地標)，這些非自然特徵會導致 GAN 生成出奇怪的偽影 (Artifacts)。
**解決方案**: 在 Folium 地圖中引入 **CartoDB Voyager No Labels** 圖層，提供乾淨、無文字的底圖，顯著提升了「地圖轉衛星」的生成品質。

## 5. 安裝與執行 (Installation & Usage)

### 環境需求
- Python 3.8+
- 相關套件詳見 `requirements.txt`

### 安裝步驟
```bash
pip install -r requirements.txt
```

### 執行程式
```bash
streamlit run app.py
```

## 6. 結論 (Conclusion)
本專案成功整合了深度學習模型與現代化 Web 介面，提供了一個直觀的工具來展示 CycleGAN 的強大能力。透過解決模型架構匹配與輸入資料品質 (無標籤地圖) 的問題，大幅提升了使用者的體驗與生成結果的可用性。
