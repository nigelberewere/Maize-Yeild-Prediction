# Web UI Usage

## Starting the Web Application

### Prerequisites
```powershell
pip install -r requirements.txt
```

### Run the Web Server
```powershell
python app.py
```

Then open your browser to: **http://127.0.0.1:5000**

## Features

✨ **Modern Web Interface**
- Clean, responsive design with Bootstrap styling
- Real-time prediction results
- Input validation with helpful hints
- Mobile-friendly layout

📊 **Easy Input**
- Annual Rainfall (mm) — Zimbabwe average: 694-850 mm
- Average Temperature (°C) — Zimbabwe average: 20-24°C  
- Fertilizer Use (kg/ha) — Zimbabwe typical: 30-120 kg/ha
- Area Harvested (hectares) — National average: 350k-420k ha

🎯 **Instant Results**
- Expected Yield in kg/hectare
- Total Production in tonnes
- Formatted for easy reading

## Technical Details

- **Backend:** Flask (Python)
- **Frontend:** HTML5, Bootstrap 5, Vanilla JavaScript
- **Model:** Linear Regression trained on Zimbabwe maize data
- **API Endpoint:** POST `/predict` (JSON request/response)

## Example API Call

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"rainfall": 700, "temperature": 22, "fertilizer": 80, "area": 380000}'
```

Response:
```json
{
  "success": true,
  "yield_kg_per_ha": 1345.92,
  "production_tonnes": 511000,
  "area_hectares": 380000
}
```

## Deployment

For production, use a WSGI server like Gunicorn:
```powershell
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```
