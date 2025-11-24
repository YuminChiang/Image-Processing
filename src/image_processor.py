# image_processor.py

import cv2
import numpy as np

def _process_intensity(img, func, **kwargs):
    if len(img.shape) == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0]
        processed_y = func(y, **kwargs)
        ycrcb[:, :, 0] = processed_y
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        return func(img, **kwargs)

def average_filter(img, kernel_size=5):
    avg_blur = cv2.blur(img, (kernel_size, kernel_size))
    return np.uint8(avg_blur)

def median_filter(img, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1

    med_blur = cv2.medianBlur(img, kernel_size)
    return np.uint8(med_blur)

def gauss_blur(img, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=0)
    return np.uint8(blurred)

def sobel_sharp(img, k=0.5):
    return _process_intensity(img, _sobel_sharp_core, k=k)

def _sobel_sharp_core(gray, k):
    # gray_float = gray.astype(np.float32)

    # sobelx = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    # sobely = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)

    # magnitude = cv2.magnitude(sobelx, sobely)
 
    # sharpened = gray_float + k * magnitude
    
    # return np.clip(sharpened, 0, 255).astype(np.uint8)
    gray = gray.astype(np.float32)

    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    high_freq = sobelx + sobely

    sharpened = gray + k * high_freq

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
    return _process_intensity(img, _fft_sharp_core, D0=radius, k=strength)

def _fft_sharp_core(img, D0=40, k=1.5):
    gray = gray.astype(np.float32)

    # 1. Forward FFT using cv2.dft
    dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 2. Construct Gaussian HPF from textbook
    M, N = gray.shape
    u = np.arange(M)
    v = np.arange(N)
    V, U = np.meshgrid(v, u)

    D = np.sqrt((U - M/2)**2 + (V - N/2)**2)

    # H_HP = 1 - H_LP
    H_LP = np.exp(-(D**2) / (2 * (D0**2)))
    H_HP = 1 - H_LP

    # High-boost sharpening: H = 1 + k * H_HP
    H = 1 + k * H_HP

    # Make 2-channel filter for OpenCV
    H2 = np.repeat(H[:, :, np.newaxis], 2, axis=2)

    # 3. Apply filter
    G = dft_shift * H2

    # 4. Inverse FFT
    G_ishift = np.fft.ifftshift(G)
    img_back = cv2.idft(G_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # 5. Clip & return
    return np.uint8(np.clip(img_back, 0, 255))

    

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