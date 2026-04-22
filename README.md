# SmartLens 

SmartLens is an AI-powered image analysis system that combines multiple computer vision techniques into a single application.


## Features

- 🧠 Object Detection using YOLOv8
- 🔤 Text Extraction (OCR)
- 📦 Barcode & QR Code Scanning
- 🌐 Language Translation


## Tech Stack

- Python
- Flask
- OpenCV
- YOLOv8 (Ultralytics)
- Tesseract OCR


## Project Structure

SmartLens/
│
├── app.py
├── analyze.py
├── detect.py
├── ocr.py
├── barcode.py
├── translator.py
├── requirements.txt
│
├── static/
│   ├── style.css
│   └── script.js
│
├── templates/
│   └── index.html
│
├── README.md
├── .gitignore


## Installation

1. Clone the repository:
   git clone https://github.com/Tanya-869/SmartLens.git

2. Navigate to project folder:
   cd SmartLens

3. Install dependencies:
   pip install -r requirements.txt

4. Run the application:
   python app.py


## Note

- YOLO model weights (.pt files) are not included due to size limitations.
- You can download them from:
  https://github.com/ultralytics/ultralytics


## Screenshots

(Add your screenshots here)

## Author

Tanya
