# image_processor.py

import cv2
import numpy as np

# --- 內部 Helper Function (色彩空間轉換公式) ---
def _process_intensity(img, func, **kwargs):
    """
    色彩轉換公式 (RGB to YCrCb):
    Y  = 0.299*R + 0.587*G + 0.114*B  (亮度 Luminance)
    Cr = 0.713*(R - Y) + 128          (紅色色差 Chroma Red)
    Cb = 0.564*(B - Y) + 128          (藍色色差 Chroma Blue)
    
    我們只處理 Y 通道，因為人眼對亮度變化最敏感。
    """
    if len(img.shape) == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0]
        processed_y = func(y, **kwargs)
        ycrcb[:, :, 0] = processed_y
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        return func(img, **kwargs)

# --- 空間域濾波 (Spatial Domain Filtering) ---

def average_filter(img, kernel_size=5):
    # 公式: G(x,y) = (1 / k^2) * Σ Σ I(x+i, y+j)
    # 說明: 計算 kernel 範圍內所有像素的平均值
    avg_blur = cv2.blur(img, (kernel_size, kernel_size))
    return np.uint8(avg_blur)

def median_filter(img, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 公式: G(x,y) = Median{ I(x+i, y+j) }
    # 說明: 取 kernel 範圍內的中位數 (去除椒鹽雜訊最有效)
    med_blur = cv2.medianBlur(img, kernel_size)
    return np.uint8(med_blur)

def gauss_blur(img, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 公式: G(x,y) = (1 / 2πσ^2) * exp( -(x^2 + y^2) / 2σ^2 )
    # 說明: 權重隨距離中心越遠而呈現常態分佈遞減
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=0)
    return np.uint8(blurred)

def sobel_sharp(img, k=0.5):
    return _process_intensity(img, _sobel_sharp_core, k=k)

def _sobel_sharp_core(gray, k):
    gray_float = gray.astype(np.float32)

    # 梯度公式 (Gradient):
    # Gx = [-1 0 1; -2 0 2; -1 0 1] * I  (水平梯度)
    # Gy = [-1 -2 -1; 0 0 0; 1 2 1] * I  (垂直梯度)
    sobelx = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)

    # 梯度強度公式 (Gradient Magnitude):
    # G = sqrt( Gx^2 + Gy^2 )
    magnitude = cv2.magnitude(sobelx, sobely)
    
    # 銳化疊加公式:
    # Output = Original + k * Magnitude
    sharpened = gray_float + k * magnitude
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)

# --- 頻率域濾波 (Frequency Domain Filtering) ---

def fft_denoise(img, radius=30):
    return _process_intensity(img, _fft_denoise_core, radius=radius)

def _fft_denoise_core(gray, radius):
    # 1. 傅立葉轉換: F(u,v) = DFT(f(x,y))
    gray_float = gray.astype(np.float32)
    dft = cv2.dft(gray_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 2. 定義理想低通濾波器 (Ideal Low Pass Filter, ILPF)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # 公式: H(u,v) = 1 if D(u,v) <= radius else 0
    # D(u,v) 是頻率點到中心的距離
    mask = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, (1, 1), -1)

    # 3. 頻域濾波: G(u,v) = F(u,v) * H(u,v)
    fshift = dft_shift * mask
    
    # 4. 逆轉換: g(x,y) = IDFT(G(u,v))
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    
    return np.clip(img_back, 0, 255).astype(np.uint8)

def fft_sharp(img, radius=30, strength=0.5):
    return _process_intensity(img, _fft_sharp_core, radius=radius, strength=strength)

def _fft_sharp_core(gray, radius, strength):
    gray_float = gray.astype(np.float32)
    dft = cv2.dft(gray_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    x = np.linspace(-ccol, ccol - 1, cols)
    y = np.linspace(-crow, crow - 1, rows)
    X, Y = np.meshgrid(x, y)
    
    # 頻率距離公式: D^2 = u^2 + v^2
    D2 = X**2 + Y**2

    # 高斯高通濾波器公式 (Gaussian High Pass Filter, GHPF):
    # H_hp(u,v) = 1 - exp( -D^2 / (2 * radius^2) )
    # 說明: 1 減去 低通 = 高通
    mask_gaussian = 1 - np.exp(-D2 / (2 * (radius**2)))
    mask = np.stack([mask_gaussian, mask_gaussian], axis=2)

    # 頻域濾波
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    
    # 取出高頻邊緣訊號 (Edges)
    img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    # High Boost 銳化公式:
    # Output = Original + strength * HighPass_Signal
    sharpened = gray_float + (strength * img_back * 2.0) 
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def gauss_fft(img, cutoff=30):
    return _process_intensity(img, _gauss_fft_core, cutoff=cutoff)

def _gauss_fft_core(gray, cutoff):
    gray_float = gray.astype(np.float32)
    dft = cv2.dft(gray_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    x = np.linspace(-ccol, ccol - 1, cols)
    y = np.linspace(-crow, crow - 1, rows)
    X, Y = np.meshgrid(x, y)
    D2 = X**2 + Y**2

    # 高斯低通濾波器公式 (Gaussian Low Pass Filter, GLPF):
    # H_lp(u,v) = exp( -D^2 / (2 * cutoff^2) )
    mask_gaussian = np.exp(-D2 / (2 * (cutoff**2)))
    mask = np.stack([mask_gaussian, mask_gaussian], axis=2)

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    
    img_smooth = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    
    return np.clip(img_smooth, 0, 255).astype(np.uint8)