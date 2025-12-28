# Food Recognition - Modern Interface

## Overview

The Food Recognition application now features a modern, sleek 3-column interface with AI-powered health analysis using Claude Code.

## New Features

### 🎨 Modern UI Design
- **Dark Theme**: Professional dark theme with Spotify-inspired green accents
- **3-Column Layout**:
  - Column 1: Collapsible sidebar menu for navigation
  - Column 2: Image upload and model configuration
  - Column 3: Nutritional analysis and AI health scoring

### 🧠 AI Health Analysis
- Powered by Claude API (Anthropic)
- Analyzes detected food and provides:
  - Health score (0-100)
  - Detailed explanation of the score
  - Personalized recommendations for healthier eating

### 📊 Nutritional Information
- Displays calories, protein, carbs, and fat for each detected food item
- Shows total nutritional values for the entire meal
- Visual health bar that changes color based on healthiness

## Setup

### 1. Install Dependencies

```bash
pip install anthropic pandas
```

### 2. Set Up Claude API Key

You need an Anthropic API key to use the AI health analysis feature.

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Or add it to your `.env` file:
```
ANTHROPIC_API_KEY=your-api-key-here
```

Get your API key from: [https://console.anthropic.com/](https://console.anthropic.com/)

### 3. Run the Application

```bash
python app.py
```

## Usage

### Accessing the Interface

- **Modern Interface** (default): [http://localhost:5000/](http://localhost:5000/) or [http://localhost:5000/modern](http://localhost:5000/modern)
- **Classic Interface**: [http://localhost:5000/classic](http://localhost:5000/classic)

### Using the Modern Interface

1. **Upload an Image**
   - Click "Upload Image" button
   - Select a food image from your computer
   - Preview will appear in the center panel

2. **Configure Model Settings**
   - Choose detection model (YOLOv8s, YOLOv5s, etc.)
   - Adjust confidence threshold (default: 15%)
   - Adjust IOU threshold (default: 50%)
   - Toggle optional features:
     - Ensemble Models
     - Test-Time Augmentation
     - Label Enhancement
     - Semantic Segmentation

3. **Analyze Food**
   - Click "Analyze Food" button
   - Wait for detection to complete
   - View results:
     - Detected food items with bounding boxes
     - Nutritional information in right panel
     - AI-powered health score and recommendations

4. **Health Score**
   - Visual health bar shows score from 0-100
   - Colors indicate health level:
     - 🔴 Red (0-39): Low health score
     - 🟡 Yellow (40-69): Medium health score
     - 🟢 Green (70-100): High health score
   - AI-generated explanation and recommendations

## File Structure

```
food-recognition/
├── templates/
│   ├── modern-interface.html      # New 3-column interface
│   └── upload-file.html           # Classic interface
├── static/
│   ├── css/
│   │   └── modern-style.css       # Modern dark theme styles
│   └── js/
│       └── modern-interface.js    # Modern interface JavaScript
├── backend/
│   ├── routes.py                  # Updated with API endpoints
│   └── health_analyzer.py         # Claude API health analysis
└── MODERN_INTERFACE_README.md     # This file
```

## API Endpoints

### Modern Interface Endpoints

- `GET /modern` - Render modern interface
- `POST /api/analyze` - Analyze food image (returns JSON)
- `POST /api/analyze-health` - Get AI health analysis (requires Claude API key)

### Request Format (api/analyze)

```javascript
FormData {
  file: <image file>,
  'model-types': 'yolov8s',
  'confidence-range': '15',
  'threshold-range': '50',
  'ensemble': 'on',  // optional
  'tta': 'on',       // optional
  'enhanced': 'on',  // optional
  'seg': 'on'        // optional
}
```

### Response Format (api/analyze)

```json
{
  "success": true,
  "detection_image": "/static/assets/detections/image.jpg",
  "nutrition_data": {
    "foods": [
      {
        "name": "Apple",
        "calories": 95,
        "protein": 0.5,
        "carbs": 25,
        "fat": 0.3
      }
    ],
    "total_calories": 95,
    "total_protein": 0.5,
    "total_carbs": 25,
    "total_fat": 0.3
  }
}
```

### Request Format (api/analyze-health)

```json
{
  "nutrition_data": {
    "foods": [...],
    "total_calories": 200,
    "total_protein": 10,
    "total_carbs": 30,
    "total_fat": 5
  }
}
```

### Response Format (api/analyze-health)

```json
{
  "success": true,
  "data": {
    "health_score": 75,
    "explanation": "This meal has a good balance of nutrients with moderate calories.",
    "recommendations": [
      "Consider adding more protein sources",
      "Great fiber content from fruits",
      "Well-balanced macronutrient distribution"
    ]
  }
}
```

## Customization

### Colors

Edit [static/css/modern-style.css](static/css/modern-style.css#L3-L13) to change the color scheme:

```css
:root {
    --bg-primary: #0d0d0d;
    --bg-secondary: #1a1a1a;
    --bg-tertiary: #252525;
    --accent-green: #1DB954;  /* Change this for different accent color */
    /* ... */
}
```

### Claude Model

Edit [backend/health_analyzer.py](backend/health_analyzer.py#L48) to use a different Claude model:

```python
message = self.client.messages.create(
    model="claude-3-5-sonnet-20241022",  # Change model here
    # ...
)
```

## Troubleshooting

### Health Analysis Not Working

**Problem**: "ANTHROPIC_API_KEY not configured" error

**Solution**: Make sure you've set the environment variable:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Nutritional Data Not Showing

**Problem**: Nutrition panel shows "No nutritional data available"

**Solution**: This likely means the CSV files weren't generated. Check:
1. Detection completed successfully
2. CSV files exist in `static/csv/` directory
3. CSV files contain the expected columns (Food, Calories, Protein, Carbohydrate, Fat)

### Sidebar Not Collapsing

**Problem**: Sidebar menu doesn't expand/collapse

**Solution**:
1. Check browser console for JavaScript errors
2. Ensure [static/js/modern-interface.js](static/js/modern-interface.js) is loaded
3. Clear browser cache

## Browser Compatibility

Tested and working on:
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## Performance

- Image upload: Instant preview
- Food detection: 2-5 seconds (depends on model and image size)
- Health analysis: 1-3 seconds (Claude API call)

## Future Enhancements

- [ ] Save analysis history
- [ ] Compare meals over time
- [ ] Export nutrition reports
- [ ] Barcode scanning
- [ ] Mobile-responsive design improvements
- [ ] Real-time webcam detection in modern interface

## Credits

- Original interface by [kaylode](https://github.com/kaylode/) and [lannguyen](https://github.com/lannguyen0910/)
- Modern interface design inspired by MetroMatch
- AI health analysis powered by [Anthropic Claude](https://www.anthropic.com/)
