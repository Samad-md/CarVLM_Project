# CarVLM FastAPI Backend

## 1) Copy your trained model
Copy all files from your Google Drive folder:

`MyDrive/CarVLM_Project/best_blip_vehicle_model/`

into:

`CarVLM_FastAPI_Backend/model/`

## 2) Open in VS Code
Open the extracted `CarVLM_FastAPI_Backend` folder.

## 3) Create environment (Windows)
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Recommended: Python 3.10 or 3.11.

## 4) Run backend
```powershell
uvicorn main:app --reload
```

Then open:

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

## 5) Test an image
At `/docs`, open `POST /predict` → **Try it out** → upload a car image → **Execute**.

Or:
```powershell
python test_api.py "D:\path\to\car.jpg"
```

## Endpoints
- `GET /`
- `GET /health`
- `GET /classes`
- `POST /predict`

## Important
This backend is for your current 10-class closed-set prototype.
`label_match_score` is text similarity to a known class label, not calibrated visual confidence.
