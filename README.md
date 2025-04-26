# Counting On You - Virtual Fitting Room

A modern virtual fitting room application with a Chanel-inspired design, built with FastAPI, React, and Tailwind CSS.

## Features

- Elegant, Chanel-inspired UI design
- Image upload with drag-and-drop functionality
- Body measurement calculation
- Responsive design for all devices

## Tech Stack

- **Backend**: FastAPI, Python
- **Frontend**: React, Tailwind CSS
- **Computer Vision**: MediaPipe, OpenCV

## Setup Instructions

### Backend Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the FastAPI server:
   ```
   uvicorn app.main:app --reload
   ```

### Frontend Setup

1. Install Node.js dependencies:
   ```
   npm install
   ```

2. Start the development server:
   ```
   npm start
   ```

3. For production build:
   ```
   npm run build
   ```

## Usage

1. Open your browser and navigate to `http://localhost:8000`
2. Upload an image by dragging and dropping or clicking the upload area
3. Enter your height in centimeters
4. Click "Calculate Measurements" to get your body measurements

## API Endpoints

- `POST /api/v1/measurements/upload`: Upload an image and get body measurements

## License

MIT # Virtual-Fitting-room
