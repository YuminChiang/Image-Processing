# Image Processing

A Python-based image processing application with a graphical user interface built using PyQt5.  This tool provides various image manipulation capabilities including filtering, transformations, and enhancement operations.

## Features

### Image Operations

- **Load and Save Images**: Support for common image formats (JPG, PNG, etc.)
- **Image Filtering**:
    - Gaussian Blur
    - Median Filter
    - Bilateral Filter
- **Image Transformations**:
    - Rotation
    - Scaling
    - Translation
    - Affine Transformations
- **Image Enhancement**:
    - Histogram Equalization
    - Contrast Adjustment
    - Brightness Control
- **Edge Detection**:
    - Canny Edge Detection
    - Sobel Edge Detection
- **Morphological Operations**:
    - Erosion
    - Dilation
    - Opening
    - Closing

## Project Structure

```
Image-Processing/
├── src/
│   ├── main.py       # Application entry point
│   ├── ui.py         # PyQt5 UI implementation
│   ├── ui.ui         # Qt Designer UI file
│   └── imgproc.py    # Core image processing functions
├── assets/
│   ├── image1.jpg    # Sample images
│   └── image2.jpg
├── .gitignore
└── README.md

```

## Requirements

- Python 3.x
- PyQt5
- OpenCV (cv2)
- NumPy
- Pillow (PIL)

## Installation

1. Clone the repository:

```bash
git clone <https://github.com/YuminChiang/Image-Processing>. git
cd Image-Processing
```

1. Install required dependencies:

```bash
pip install PyQt5 opencv-python numpy Pillow
```

## Usage

Run the application:

```bash
python src/main.py
```

### Using the GUI

1. **Load Image**: Click the "Load" button to open an image file
2. **Apply Operations**: Select desired image processing operations from the menu
3. **Adjust Parameters**: Use sliders or input fields to adjust operation parameters
4. **Save Result**: Click "Save" to export the processed image

## Core Modules

### [main.py](http://main.py/)

The entry point of the application that initializes the GUI and handles the main event loop.

### [ui.py](http://ui.py/)

Contains the PyQt5-based user interface implementation with all widgets, layouts, and event handlers.

### [imgproc.py](http://imgproc.py/)

Core image processing module containing functions for:

- Image filtering and smoothing
- Geometric transformations
- Color space conversions
- Edge detection algorithms
- Morphological operations

## Sample Images

The `assets` folder contains sample images (`image1.jpg`, `image2.jpg`) for testing the application's functionality.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](https://www.notion.so/yumin920421/LICENSE).

## Author

[YuminChiang](https://github.com/YuminChiang)

## Acknowledgments

- Built with OpenCV for robust image processing capabilities
- PyQt5 for the modern and intuitive user interface
- Sample images provided for demonstration purposes
