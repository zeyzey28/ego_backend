# Accessibility Analysis Backend

FastAPI backend for User Panel and Municipality Panel.

## Setup

### 1. Create Virtual Environment

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Server

```bash
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server will start at: http://localhost:8000

API Documentation: http://localhost:8000/docs

## API Endpoints

### User Panel

#### GET /nearest-stops
Get nearest 3 bus stops for a location.

**Parameters:**
- `lat` (float): Latitude
- `lon` (float): Longitude

**Example:**
```bash
curl "http://localhost:8000/nearest-stops?lat=39.92&lon=32.85"
```

**Response:**
```json
{
  "grid_id": 242,
  "slope_score": 87.5,
  "nearest_stops": [
    {
      "stop_id": 1750,
      "stop_name": "DURAK ADI",
      "lat": 39.92,
      "lon": 32.85,
      "distance_m": 150.5,
      "duration_min": 1.79
    }
  ]
}
```

### Municipality Panel

#### POST /complaints
Create a new complaint.

**Form Data:**
- `category` (string): Complaint category
- `description` (string): Description
- `lat` (float): Latitude
- `lon` (float): Longitude
- `photo` (file, optional): Photo upload

**Example:**
```bash
curl -X POST "http://localhost:8000/complaints" \
  -F "category=merdiven_kirik" \
  -F "description=Merdiven kırık, tehlikeli" \
  -F "lat=39.92" \
  -F "lon=32.85" \
  -F "photo=@photo.jpg"
```

**Response:**
```json
{
  "status": "ok",
  "id": 1
}
```

#### GET /complaints
Get all complaints.

**Example:**
```bash
curl "http://localhost:8000/complaints"
```

### Utility Endpoints

#### GET /categories
Get available complaint categories.

#### GET /grid/{grid_id}
Get info for a specific grid.

## Urgency Mapping

| Category | Urgency |
|----------|---------|
| boru_patlamasi | 🔴 red |
| su_baskini | 🔴 red |
| yangin | 🔴 red |
| merdiven_kirik | 🟡 yellow |
| kaldirim_bozuk | 🟡 yellow |
| rampa_eksik | 🟡 yellow |
| isik_yanmiyor | 🟢 green |
| cop_toplama | 🟢 green |
| diger | 🟢 green |

## Project Structure

```
backend/
├── main.py           # FastAPI application
├── geo.py            # Geographic utilities
├── requirements.txt  # Python dependencies
├── complaints.json   # Complaints database
├── photos/           # Uploaded photos
└── README.md         # This file
```

## Data Files (parent directory)

- `grid_access_only.geojson` - Grid polygons
- `grid_nearest_3stops.json` - Nearest stops per grid
- `bus_stops_list.json` - Bus stop details
- `grid_slope_score.json` - Slope scores per grid

