# Immune Knowledge Graph Analysis

A web application for analyzing immune related genes and pathways relationships with literature-based knowledge graphs.

## Project Structure
```
immune-knowledgegraph.github.io/
├── api/               # Flask backend
│   ├── app.py        # Main application code
│   └── requirements.txt
├── frontend/         # Frontend files
│   └── index.html    # Main webpage
└── data/            # Network data
    └── ...network_data.csv
```

## Setup

### Backend Setup
1. Navigate to the api directory:
```bash
cd api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Flask server:
```bash
python app.py
```

### Frontend Setup
1. Open `frontend/index.html` in a web browser
2. Enter gene names to analyze
3. View results in the table

## Usage
1. Enter gene names (one per line) in the text area
2. Click "Analyze" to see pathway relationships
3. Results will show pathways with their scores and statistical significance

## API Endpoints
- POST `/analyze`: Accepts a list of genes and returns pathway analysis results
