#!/bin/bash
echo "============================================================"
echo " Southern Company Network Lifecycle Dashboard"
echo "============================================================"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
echo ""

# Run ETL pipeline
echo "Running data pipeline..."
python etl_pipeline.py
echo ""

# Launch dashboard
echo "Launching dashboard..."
echo "Open http://localhost:8050 in your browser"
echo ""
python app.py
