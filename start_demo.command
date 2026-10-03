#!/bin/bash
# Double-click to start the Graph2Drug web demo, then open http://localhost:8501
cd "$(dirname "$0")"
source .venv/bin/activate
(sleep 4 && open http://localhost:8501) &
streamlit run app.py --server.port 8501
