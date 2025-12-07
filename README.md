# 🗺️ Map ↔ Satellite Converter

使用 **CycleGAN** 深度學習模型進行地圖與衛星圖像之間的雙向轉換。

## 🔍Demo Site
You can try the live application here:[https://5114056011-aiot-hw4.streamlit.app/](https://5114056011-aiot-hw4.streamlit.app/)

## ✨ 功能特色

- **雙向轉換**：地圖 → 衛星圖像 / 衛星圖像 → 地圖
- **支援多種地圖來源**：Google Maps、Apple Maps、OpenStreetMap 截圖
- **互動式地圖**：內建 Folium 互動地圖，可直接選擇區域
- **CycleGAN 模型**：對非標準輸入有更好的泛化能力
- **GPU 加速**：支援 CUDA 加速推論

## 🚀 安裝

### 1. 安裝相依套件

```bash
pip install -r requirements.txt
```

### 2. 執行應用程式

```bash
streamlit run app.py
```

## 📖 使用方式

1. 啟動應用程式後，在瀏覽器中開啟顯示的網址（通常是 `http://localhost:8501`）
2. 在側邊欄選擇轉換方向（地圖→衛星 或 衛星→地圖）
3. 如果模型尚未下載，點擊「下載模型」按鈕
4. 使用以下任一方式輸入圖像：
   - **上傳圖像**：直接上傳地圖或衛星圖像截圖
   - **互動地圖**：在內建地圖上導航，擷取截圖後上傳
5. 點擊「開始轉換」按鈕
6. 查看並下載結果

## 🔬 模型說明

本專案使用 **CycleGAN**（Cycle-Consistent Adversarial Networks）模型，
這是一種用於非配對圖像轉換的深度學習模型。

**為什麼選擇 CycleGAN？**
- ✅ 對 Google Maps 風格的截圖效果更好
- ✅ 不需要嚴格配對的訓練資料
- ✅ 支援雙向轉換

**預訓練模型來源**：[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## 💡 使用提示

1. **截圖建議**：
   - 使用正方形或接近正方形的截圖
   - 避免包含 UI 元素（搜尋框、按鈕等）
   - 選擇 zoom level 15-18 的範圍效果最佳

2. **最佳實踐**：
   - 使用純地圖視圖，減少標註和圖標
   - 選擇有明顯道路和建築物的區域
   - 保持適中的縮放級別

## 📦 相依套件

- streamlit
- torch
- torchvision
- numpy
- Pillow
- folium
- streamlit-folium

## 📚 參考資料

- [CycleGAN 論文](https://arxiv.org/abs/1703.10593)
- [pix2pix 論文](https://arxiv.org/abs/1611.07004)
- [pytorch-CycleGAN-and-pix2pix GitHub](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
