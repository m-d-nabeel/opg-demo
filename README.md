# Age Prediction Visualization Tool

A Python-based web application built with FastAPI and Jinja2 templates to visualize the pipeline of an age prediction model using precomputed predictions and masks.

## Features

- Image selection from dataset
- Visualization of original and masked images
- Animated model inference simulation
- Display of true labels and predictions
- Toggle between original and masked views
- Summary statistics (accuracy, correct/incorrect counts)

## Requirements

- Python 3.8+
- FastAPI
- Jinja2
- Pandas
- OpenCV
- NumPy
- Pillow

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd opg-age-prediction
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

The application expects the following structure:
- `dataset/` - Contains original images (PNG files) and mask files (JSON)
- `predictions.csv` - CSV file with image filenames and predicted age categories

## Running the Application

Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

Then open your browser and navigate to:
```
http://localhost:8000
```

## How It Works

1. **Image Selection**: Choose an image from the dataset
2. **Original Image**: View the unprocessed image
3. **Masked Image**: See the image with facial feature masks applied
4. **Model Inference**: Watch a simulated processing animation
5. **Results**: Compare the true age category with the model's prediction

## Age Categories

- Age_0-6: 0-6 years old
- Age_7-14: 7-14 years old
- Age_15-18: 15-18 years old
- Age_19-25: 19-25 years old
- Age_26-40: 26-40 years old
- Age_41+: 41+ years old 