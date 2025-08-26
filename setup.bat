@echo off

echo Setting up Traffic Sign Recognition System...

REM Create virtual environment
echo Creating virtual environment...
python -m venv traffic_sign_env

REM Activate virtual environment
echo Activating virtual environment...
call traffic_sign_env\Scripts\activate

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Create necessary directories
echo Creating directories...
mkdir data 2>nul
mkdir outputs 2>nul
mkdir models 2>nul
mkdir evaluation_results 2>nul

echo Setup complete!
echo.
echo To get started:
echo 1. Activate the environment: traffic_sign_env\Scripts\activate
echo 2. Download data: python train.py --download_data
echo 3. Train model: python train.py --architecture custom --epochs 50
echo 4. Evaluate model: python evaluate.py --model_path outputs/final_model_custom.h5
echo 5. Real-time detection: python realtime_detection.py --model_path outputs/final_model_custom.h5

pause
