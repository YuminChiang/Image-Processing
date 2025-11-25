# image_processor.py

import cv2
import numpy as np

def average_filter(img, kernel_size=5):
    avg_blur = cv2.blur(img, (kernel_size, kernel_size))
    return np.uint8(avg_blur)

def median_filter(img, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1

    med_blur = cv2.medianBlur(img, kernel_size)
    return np.uint8(med_blur)

def fft_denoise(img, D0=30):
    # 1. 確保輸入為 2D
    if img.ndim == 3:
        img = img[:, :, 0]
    gray = img.astype(np.float32)

    # 2. FFT 轉換
    dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 3. 建立高斯低通濾波器 (Gaussian Low Pass Filter)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # 建立網格座標 (修正為穩健寫法)
    x = np.arange(cols) - ccol
    y = np.arange(rows) - crow
    X, Y = np.meshgrid(x, y)
    
    # 計算距離中心的距離平方 D^2
    D2 = X**2 + Y**2
    
    # --- 關鍵差異點 ---
    # 低通濾波公式: H = exp(-D^2 / (2*D0^2))
    # 離中心越遠 (高頻)，數值越接近 0 (被刪除)
    H = np.exp(-D2 / (2 * (D0**2)))
    
    # 4. 應用濾波器 (雙通道相乘)
    mask = np.stack([H, H], axis=2)
    fshift = dft_shift * mask
    
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    
    return img_back


def sobel_sharp(img, k=0.5):
    gray = img[:, :, 0] if img.ndim == 3 else img
    gray = gray.astype(np.float32)

    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    high_freq = sobelx + sobely
    sharp = gray + k * high_freq

    res = np.clip(sharp, 0, 255).astype(np.uint8)
    return res

def fft_sharp(img, D0=40, k=1.5):
    if img.ndim == 3:
        img = img[:, :, 0]
    gray = img.astype(np.float32)

    dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Gaussian High-pass Filter
    M, N = gray.shape
    u = np.arange(M)
    v = np.arange(N)
    V, U = np.meshgrid(v, u)

    # 修正中心點計算：這樣更精確對應 fftshift 後的中心
    D = np.sqrt((U - M/2)**2 + (V - N/2)**2)
    
    H_LP = np.exp(-(D**2) / (2 * (D0**2)))
    H_HP = 1 - H_LP
    
    H = 1 + k * H_HP 

    dft_filtered = dft_shift * H[:, :, np.newaxis]
    f_ishift = np.fft.ifftshift(dft_filtered)
  
    img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)

    return img_back

def gauss_blur(img, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=0)
    return np.uint8(blurred)

def gauss_fft(img, cutoff=45):
    if img.ndim == 3:
        img = img[:, :, 0]
    gray = img.astype(np.float32)
    dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
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