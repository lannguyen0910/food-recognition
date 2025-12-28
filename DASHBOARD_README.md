# Food Nutrition CV Dashboard

A modern, single-page web application for food detection and nutrition analysis using custom YOLO models and LLaMA AI.

![Dashboard Preview](https://via.placeholder.com/800x400/1E1F47/FFFFFF?text=Food+Nutrition+CV+Dashboard)

## Features

### 🎨 Modern Dashboard Interface
- **Single-page design** - No scrolling, everything fits snugly
- **Dark theme** inspired by inventory management systems
- **Real-time stats** - Track scans, detected foods, and health scores
- **History sidebar** - View all previous scans with thumbnails

### 🧠 AI-Powered Analysis
- **Custom YOLO Model** - Trained on ~100 food classes from Roboflow
- **LLaMA Nutrition Insights** - AI-generated nutrition information via Hugging Face
- **Health Scoring** - 0-100 score based on nutritional balance
- **Personalized Recommendations** - AI suggestions for healthier eating

### 📊 Nutrition Tracking
- Calories, Protein, Carbs, Fats
- Vitamins & Minerals
- Health assessment
- Balance evaluation

## Quick Start

### 1. Installation

```bash
# Clone or navigate to the project
cd food-recognition

# Install dependencies
pip install -r requirements_dashboard.txt
```

Create `requirements_dashboard.txt`:
```txt
flask>=2.3.0
flask-cors>=4.0.0
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
transformers>=4.30.0
huggingface-hub>=0.16.0
torch>=2.0.0
```

### 2. Download/Train YOLO Model

#### Option A: Train Your Own Model

```bash
# Download Roboflow dataset first
# Then run training script
python train_yolo_model.py --epochs 50 --validate
```

#### Option B: Use Pretrained YOLOv8

The app will automatically download YOLOv8n if no custom model is found.

### 3. Set Up LLaMA (Hugging Face)

```bash
# Get your Hugging Face API key from https://huggingface.co/settings/tokens
export HUGGINGFACE_API_KEY="your-hf-api-key-here"

# Optional: Use API instead of local model (recommended)
export USE_LLAMA_API="true"
```

**Note:** Using the Hugging Face Inference API is recommended as it doesn't require downloading the full LLaMA model (~13GB).

### 4. Run the Application

```bash
python app_dashboard.py
```

Or with custom settings:

```bash
python app_dashboard.py --host 0.0.0.0 --port 8080 --debug
```

### 5. Access the Dashboard

Open your browser and navigate to:
```
http://localhost:5000
```

## Usage Guide

### Analyzing Food

1. **Upload Image**
   - Click the upload area or drag & drop an image
   - Supports JPG, PNG, BMP formats
   - Max file size: 60MB

2. **Adjust Settings**
   - **Confidence**: Detection confidence threshold (0-100%)
   - **Model**: Choose between custom model or YOLOv8 variants

3. **Analyze**
   - Click "Analyze Food" button
   - Wait for detection to complete (2-5 seconds)
   - View results in the nutrition panel

4. **Review Results**
   - See detected foods with bounding boxes
   - Check nutrition information for each item
   - View total calories and macros
   - Get AI-powered health score and recommendations

5. **History**
   - All analyses are saved in the sidebar
   - Click any history item to view past results
   - Stored locally in browser (localStorage)

## Project Structure

```
food-recognition/
├── app_dashboard.py              # Main Flask application
├── train_yolo_model.py           # YOLO training script
├── templates/
│   └── dashboard.html            # Main dashboard UI
├── static/
│   ├── css/
│   │   └── dashboard.css         # Dashboard styles
│   ├── js/
│   │   └── dashboard.js          # Dashboard interactions
│   ├── uploads/                  # Uploaded images
│   └── detections/               # Detection results
├── backend/
│   ├── llama_nutrition.py        # LLaMA integration
│   └── health_analyzer.py        # Health analysis (Claude)
├── datasets/
│   └── food-dataset/             # Roboflow dataset
└── runs/
    └── train/
        └── food_detection/
            └── weights/
                ├── best.pt       # Trained model
                └── last.pt       # Last checkpoint
```

## Training Custom YOLO Model

### 1. Prepare Dataset

Download a food dataset from [Roboflow Universe](https://universe.roboflow.com/):

```bash
# Example: Food-101 or custom food dataset
# Export in YOLO format
# Extract to ./datasets/food-dataset/
```

Dataset structure:
```
datasets/food-dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 2. Update Class Names

Edit `train_yolo_model.py` and update the `names` list with your actual food classes:

```python
'names': [
    'apple', 'banana', 'pizza', 'burger', # ... your classes
]
```

### 3. Train Model

```bash
# Full training
python train_yolo_model.py --epochs 100 --img-size 640 --batch-size 16

# Quick training (for testing)
python train_yolo_model.py --epochs 10

# With validation
python train_yolo_model.py --epochs 50 --validate

# Export to ONNX
python train_yolo_model.py --epochs 50 --export onnx
```

### 4. Monitor Training

Results are saved to `runs/train/food_detection/`:
- Loss curves
- Precision/Recall
- Confusion matrix
- Sample predictions

### 5. Use Trained Model

```bash
export YOLO_MODEL_PATH="./runs/train/food_detection/weights/best.pt"
python app_dashboard.py
```

## LLaMA Integration

### Using Hugging Face API (Recommended)

```bash
# Set API key
export HUGGINGFACE_API_KEY="hf_xxxxxxxxxxxx"
export USE_LLAMA_API="true"

# Run app
python app_dashboard.py
```

**Advantages:**
- No local model download (saves ~13GB)
- Faster inference
- No GPU required

### Using Local Model (Advanced)

```bash
# Disable API
export USE_LLAMA_API="false"

# Run app (will download model on first run)
python app_dashboard.py
```

**Requirements:**
- ~16GB RAM minimum
- GPU recommended (8GB+ VRAM)
- ~13GB disk space

## API Endpoints

### POST /api/detect-food

Detect food and generate nutrition insights.

**Request:**
```
FormData:
  file: <image file>
  confidence: <0-100>
  model: <"best" | "yolov8n" | "yolov8s">
```

**Response:**
```json
{
  "success": true,
  "detection_image": "/static/detections/detection_image.jpg",
  "detected_foods": [
    {
      "name": "apple",
      "confidence": 95.2
    }
  ],
  "nutrition_data": {
    "foods": [{
      "name": "apple",
      "calories": 95,
      "protein": 0.5,
      "carbs": 25,
      "fat": 0.3,
      "vitamins": ["Vitamin C", "Vitamin A"],
      "minerals": ["Potassium"],
      "benefits": "Rich in fiber and antioxidants"
    }],
    "total_calories": 95,
    "total_protein": 0.5,
    "total_carbs": 25,
    "total_fat": 0.3,
    "health_score": 85,
    "balance_assessment": "Healthy, low-calorie snack...",
    "recommendations": [
      "Great source of fiber",
      "Consider adding protein"
    ]
  }
}
```

## Configuration

### Environment Variables

```bash
# Hugging Face API
export HUGGINGFACE_API_KEY="your-key"
export USE_LLAMA_API="true"  # or "false" for local model

# YOLO Model
export YOLO_MODEL_PATH="./runs/train/food_detection/weights/best.pt"

# Server
export FLASK_HOST="0.0.0.0"
export FLASK_PORT="5000"
export FLASK_DEBUG="false"
```

### Model Selection Priority

1. Custom model at `YOLO_MODEL_PATH`
2. `./runs/train/food_detection/weights/best.pt`
3. Fallback to `yolov8n.pt` (auto-download)

## Customization

### Change Color Scheme

Edit [static/css/dashboard.css](static/css/dashboard.css#L3-L16):

```css
:root {
    --primary: #7C5CFF;       /* Main accent color */
    --success: #4ECB71;       /* Success/positive */
    --warning: #FFB547;       /* Warning/caution */
    --danger: #FF6B6B;        /* Error/negative */
    --bg-dark: #1E1F47;       /* Main background */
    --bg-darker: #16172E;     /* Darker backgrounds */
    /* ... */
}
```

### Adjust YOLO Parameters

Edit [app_dashboard.py](app_dashboard.py#L71-L75):

```python
results = yolo_model.predict(
    source=image,
    conf=confidence,      # Confidence threshold
    iou=0.45,            # Add IOU threshold
    max_det=100,         # Max detections
    classes=[0,1,2],     # Specific classes only
    verbose=False
)
```

### Customize LLaMA Prompt

Edit [backend/llama_nutrition.py](backend/llama_nutrition.py#L67-L97) to change the nutrition analysis prompt.

## Troubleshooting

### Issue: Model Not Found

**Error:** `Custom model not found. Using YOLOv8n pretrained model.`

**Solution:**
1. Train a custom model: `python train_yolo_model.py`
2. Or set path to existing model: `export YOLO_MODEL_PATH="path/to/model.pt"`

### Issue: LLaMA API Error

**Error:** `HUGGINGFACE_API_KEY environment variable required`

**Solution:**
1. Get API key from https://huggingface.co/settings/tokens
2. Set environment variable: `export HUGGINGFACE_API_KEY="your-key"`

### Issue: Out of Memory (Local LLaMA)

**Error:** `CUDA out of memory` or `Killed`

**Solution:**
1. Use API instead: `export USE_LLAMA_API="true"`
2. Or use smaller model (edit `model_name` in llama_nutrition.py)
3. Or increase system RAM/swap

### Issue: Low Detection Accuracy

**Solutions:**
1. Train with more epochs: `--epochs 100`
2. Use larger YOLO model: YOLOv8m or YOLOv8l
3. Increase dataset size and quality
4. Adjust confidence threshold
5. Check dataset labels are correct

### Issue: Slow Inference

**Solutions:**
1. Use smaller YOLO model (YOLOv8n)
2. Reduce image size: `--img-size 320`
3. Use GPU if available
4. Enable TensorRT or ONNX export

## Performance

### Expected Performance

- **YOLO Inference:** 50-200ms (depends on model and hardware)
- **LLaMA API:** 1-3 seconds
- **LLaMA Local:** 5-15 seconds
- **Total Analysis:** 2-5 seconds (with API)

### Hardware Recommendations

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 5GB

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB
- GPU: NVIDIA with 4GB+ VRAM
- Storage: 20GB SSD

## Credits

- **YOLO:** [Ultralytics](https://github.com/ultralytics/ultralytics)
- **LLaMA:** [Meta AI](https://ai.meta.com/llama/) via [Hugging Face](https://huggingface.co/)
- **UI Design:** Inspired by modern inventory dashboards
- **Original Food Recognition:** [kaylode](https://github.com/kaylode/) and [lannguyen](https://github.com/lannguyen0910/)

## License

This project is for educational and demonstration purposes.

## Support

For issues or questions:
1. Check this README
2. Review error messages
3. Check model and API key configuration
4. Ensure all dependencies are installed

## Future Enhancements

- [ ] Barcode scanning
- [ ] Meal planning
- [ ] Nutrition goals tracking
- [ ] Multi-language support
- [ ] Mobile app
- [ ] Batch processing
- [ ] Export reports (PDF, CSV)
- [ ] Social sharing
- [ ] Recipe suggestions
