"""
Map ↔ Satellite Converter - Streamlit 應用程式
支援互動式地圖選擇和 Google Maps 截圖轉換

使用 CycleGAN 模型進行衛星圖像與地圖之間的轉換
基於 pytorch-CycleGAN-and-pix2pix 專案
"""

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import urllib.request
import os
from io import BytesIO

# =====================================================
# CycleGAN ResNet Generator 模型架構
# =====================================================

class ResnetBlock(nn.Module):
    """ResNet Block for CycleGAN"""
    def __init__(self, dim, use_bias=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            nn.BatchNorm2d(dim, affine=False, track_running_stats=True),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            nn.BatchNorm2d(dim, affine=False, track_running_stats=True)
        )

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """ResNet-based Generator for CycleGAN (與預訓練模型相容)"""
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        super(ResnetGenerator, self).__init__()
        
        use_bias = True
        
        # Initial conv block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            nn.BatchNorm2d(ngf, affine=False, track_running_stats=True),
            nn.ReLU(True)
        ]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                nn.BatchNorm2d(ngf * mult * 2, affine=False, track_running_stats=True),
                nn.ReLU(True)
            ]

        # ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, use_bias=use_bias)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                   kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                nn.BatchNorm2d(int(ngf * mult / 2), affine=False, track_running_stats=True),
                nn.ReLU(True)
            ]
        
        # Output conv
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0, bias=use_bias),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


# =====================================================
# 工具函式
# =====================================================

def download_cyclegan_model(model_name, save_dir="checkpoints"):
    """下載 CycleGAN 預訓練模型"""
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_name}_cyclegan.pth")
    
    if not os.path.exists(model_path):
        url = f"http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/{model_name}.pth"
        st.info(f"正在下載 CycleGAN 模型: {model_name}...")
        
        try:
            urllib.request.urlretrieve(url, model_path)
            st.success(f"模型下載完成！")
        except Exception as e:
            st.error(f"下載失敗: {e}")
            return None
    
    return model_path


@st.cache_resource
def load_cyclegan_model(model_path, _device_str):
    """載入 CycleGAN 生成器模型"""
    device = torch.device(_device_str)
    
    # 建立與預訓練模型相容的 ResNet generator
    model = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=9)
    
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
    except:
        state_dict = torch.load(model_path, map_location=device)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def preprocess_image(image, target_size=256):
    """預處理輸入圖像"""
    # 保持正方形比例
    w, h = image.size
    size = min(w, h)
    left = (w - size) // 2
    top = (h - size) // 2
    image = image.crop((left, top, left + size, top + size))
    
    # 調整大小
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # 轉換為 numpy 陣列
    img_array = np.array(image).astype(np.float32)
    
    # 正規化到 [-1, 1]
    img_array = (img_array / 255.0 - 0.5) / 0.5
    
    # 轉換為 PyTorch tensor [B, C, H, W]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor


def postprocess_image(tensor):
    """後處理輸出 tensor 為圖像"""
    img = tensor.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
    img = (img * 0.5 + 0.5) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def run_inference(model, input_image, device, size=256):
    """執行推論"""
    with torch.no_grad():
        input_tensor = preprocess_image(input_image, size).to(device)
        output_tensor = model(input_tensor)
        output_image = postprocess_image(output_tensor)
    return output_image


# =====================================================
# Streamlit 應用程式
# =====================================================

def main():
    st.set_page_config(
        page_title="Map ↔ Satellite Converter",
        page_icon="🗺️",
        layout="wide"
    )
    
    st.title("🗺️ 地圖 ↔ 衛星圖像轉換器")
    st.markdown("""
    使用 **CycleGAN** 深度學習模型進行地圖與衛星圖像之間的轉換。
    
    ✨ **使用方式**：
    1. 選擇轉換方向
    2. 在互動地圖中導航到目標區域，擷取截圖
    3. 上傳截圖進行 AI 轉換
    """)
    
    st.divider()
    
    # 側邊欄設定
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # 選擇轉換方向
        direction = st.radio(
            "選擇轉換方向",
            ["🗺️ → 🛰️ 地圖轉衛星", "🛰️ → 🗺️ 衛星轉地圖"],
            index=0
        )
        
        if "地圖轉衛星" in direction:
            model_name = "map2sat"
            input_type = "地圖"
            output_type = "衛星圖像"
        else:
            model_name = "sat2map"
            input_type = "衛星圖像"
            output_type = "地圖"
        
        st.divider()
        
        # 圖像大小設定
        output_size = st.selectbox(
            "輸出圖像大小",
            [256, 512],
            index=0,
            help="較大的圖像需要更多處理時間"
        )
        
        # 設備選擇
        if torch.cuda.is_available():
            device_choice = st.selectbox("選擇裝置", ["CUDA (GPU)", "CPU"])
        else:
            st.info("未偵測到 GPU，使用 CPU")
            device_choice = "CPU"
        
        st.divider()
        
        # 模型狀態
        st.header("📦 模型狀態")
        checkpoints_dir = "checkpoints"
        model_path = os.path.join(checkpoints_dir, f"{model_name}_cyclegan.pth")
        
        if os.path.exists(model_path):
            st.success(f"✅ {model_name} 模型已就緒")
        else:
            st.warning(f"⚠️ 需要下載 {model_name} 模型")
            if st.button(f"📥 下載 {model_name} 模型", use_container_width=True):
                download_cyclegan_model(model_name, checkpoints_dir)
                st.rerun()
        
        st.divider()
        
        # 使用說明
        with st.expander("📖 使用說明"):
            st.markdown("""
            1. **選擇轉換方向**
            2. **確保模型已下載**
            3. **使用無標籤地圖圖層**
            4. **擷取地圖截圖**
            5. **上傳並轉換**
            
            **建議使用：**
            - CartoDB 無標籤圖層
            - Esri/Google 衛星圖
            """)
    
    # 主要內容區 - 使用 tabs
    tab1, tab2 = st.tabs(["🗺️ 互動地圖", "❓ 使用說明"])
    
    with tab1:
        st.subheader("🗺️ 互動式地圖")
        st.markdown("""
        使用下方的互動地圖導航到您想要轉換的區域，然後擷取螢幕截圖並上傳。
        """)
        
        # 嘗試使用 Folium
        try:
            import folium
            from streamlit_folium import st_folium
            
            col_map, col_result = st.columns([2, 1])
            
            with col_map:
                # 預設位置（台中興大）
                default_lat = 24.1215
                default_lon = 120.6756
                
                # 建立地圖
                m = folium.Map(
                    location=[default_lat, default_lon],
                    zoom_start=16,
                    tiles=None
                )
                
                # 添加無標註地圖圖層（避免文字影響模型）
                folium.TileLayer(
                    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
                    attr='Esri',
                    name='Esri 街道地圖（少標註）',
                    overlay=False
                ).add_to(m)
                
                folium.TileLayer(
                    tiles='https://{s}.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}{r}.png',
                    attr='CartoDB',
                    name='CartoDB 無標籤',
                    overlay=False
                ).add_to(m)
                
                folium.TileLayer(
                    tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                    attr='Google',
                    name='Google 衛星（無標籤）',
                    overlay=False
                ).add_to(m)
                
                folium.TileLayer(
                    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                    attr='Esri',
                    name='Esri 衛星圖',
                    overlay=False
                ).add_to(m)
                
                folium.LayerControl().add_to(m)
                
                # 顯示地圖
                map_data = st_folium(m, width=700, height=500)
                
                st.info("💡 請使用 **CartoDB 無標籤** 或 **衛星圖** 圖層，避免文字標註影響轉換效果。使用 **Win+Shift+S** 擷取截圖。")
            
            with col_result:
                st.subheader("📤 上傳截圖")
                
                map_screenshot = st.file_uploader(
                    "上傳地圖截圖",
                    type=["jpg", "jpeg", "png"],
                    key="map_screenshot"
                )
                
                if map_screenshot is not None:
                    map_image = Image.open(map_screenshot).convert("RGB")
                    st.image(map_image, caption="截圖預覽", use_container_width=True)
                    
                    if os.path.exists(model_path):
                        if st.button("🚀 轉換截圖", type="primary", use_container_width=True):
                            with st.spinner("轉換中..."):
                                try:
                                    device_str = "cuda" if "CUDA" in device_choice else "cpu"
                                    model, device = load_cyclegan_model(model_path, device_str)
                                    output_image = run_inference(model, map_image, device, output_size)
                                    st.image(output_image, caption="轉換結果", use_container_width=True)
                                    st.session_state['map_output'] = output_image
                                except Exception as e:
                                    st.error(f"錯誤: {e}")
                        
                        if 'map_output' in st.session_state:
                            buf = BytesIO()
                            st.session_state['map_output'].save(buf, format="PNG")
                            buf.seek(0)
                            st.download_button("💾 下載", data=buf, file_name="converted.png", mime="image/png")
                    else:
                        st.warning("請先下載模型")
                        
        except ImportError:
            st.warning("📦 需要安裝額外套件來使用互動地圖功能")
            st.code("pip install folium streamlit-folium", language="bash")
            
            # 備用方案：嵌入式地圖
            st.markdown("### 🗺️ 備用方案：嵌入式地圖")
            st.markdown("您可以在下方地圖中導航，然後使用截圖工具擷取。")
            
            # 嵌入 OpenStreetMap
            iframe_html = '''
            <iframe 
                width="100%" 
                height="500" 
                frameborder="0" 
                scrolling="no" 
                marginheight="0" 
                marginwidth="0" 
                src="https://www.openstreetmap.org/export/embed.html?bbox=120.6556%2C24.1015%2C120.6956%2C24.1415&amp;layer=mapnik"
                style="border: 1px solid #ccc; border-radius: 8px;">
            </iframe>
            '''
            st.components.v1.html(iframe_html, height=520)
            
            st.info("💡 使用 **Win+Shift+S** 擷取上方地圖區域的截圖")
            
            # 上傳截圖
            map_screenshot = st.file_uploader(
                "上傳地圖截圖",
                type=["jpg", "jpeg", "png"],
                key="map_screenshot_backup"
            )
            
            if map_screenshot is not None:
                col_a, col_b = st.columns(2)
                with col_a:
                    map_image = Image.open(map_screenshot).convert("RGB")
                    st.image(map_image, caption="輸入截圖", use_container_width=True)
                
                with col_b:
                    if os.path.exists(model_path):
                        if st.button("🚀 轉換", type="primary", use_container_width=True):
                            with st.spinner("轉換中..."):
                                try:
                                    device_str = "cuda" if "CUDA" in device_choice else "cpu"
                                    model, device = load_cyclegan_model(model_path, device_str)
                                    output_image = run_inference(model, map_image, device, output_size)
                                    st.image(output_image, caption="轉換結果", use_container_width=True)
                                except Exception as e:
                                    st.error(f"錯誤: {e}")
                    else:
                        st.warning("請先下載模型")
    
    with tab2:
        st.markdown("""
        ### 📸 如何擷取地圖截圖
        
        **方法 1: 使用 Google Maps**
        1. 開啟 [Google Maps](https://www.google.com/maps)
        2. 切換到「地圖」或「衛星」檢視
        3. 縮放到想要的區域
        4. 使用截圖工具擷取（Windows: **Win+Shift+S**）
        
        **方法 2: 使用 OpenStreetMap**
        1. 開啟 [OpenStreetMap](https://www.openstreetmap.org)
        2. 導航到目標區域
        3. 擷取螢幕截圖
        
        **💡 提示**：為了最佳效果，建議：
        - 使用正方形或接近正方形的截圖
        - 避免包含 UI 元素（搜尋框、按鈕等）
        - 選擇 zoom level 15-18 的範圍
        """)
        
        st.divider()
        
        st.markdown("""
        ### 🔬 關於 CycleGAN 模型
        
        **CycleGAN** 是一種用於非配對圖像轉換的深度學習模型。與 pix2pix 不同，
        CycleGAN 不需要嚴格配對的訓練資料，因此對於不同來源的地圖截圖有更好的泛化能力。
        
        **為什麼選擇 CycleGAN？**
        - ✅ 對各種地圖風格有較好的泛化能力
        - ✅ 不需要嚴格配對的訓練資料
        - ✅ 支援雙向轉換（地圖↔衛星）
        
        **模型來源**: [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
        
        ### ⚠️ 注意事項
        
        1. **輸入圖像品質**：較清晰的截圖會產生較好的結果
        2. **圖像比例**：建議使用正方形或接近正方形的截圖
        3. **縮放級別**：zoom 15-18 的地圖效果最佳
        4. **避免 UI 元素**：截圖時避免包含搜尋框、按鈕等介面元素
        
        ### 🎯 最佳實踐
        
        - 使用純地圖視圖，減少標註和圖標
        - 選擇有明顯道路和建築物的區域
        - 保持適中的縮放級別
        """)
    
    # 頁尾
    st.divider()
    st.caption("🤖 基於 CycleGAN 模型 | [GitHub](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)")


if __name__ == "__main__":
    main()
