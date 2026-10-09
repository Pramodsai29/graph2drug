#!/bin/bash
# Double-click to start the Graph2Drug web demo, then open http://localhost:8501
cd "$(dirname "$0")"
# use the first environment that has the demo's packages (streamlit, shap)
for v in .venv "$HOME/Documents/graph2drug/.venv"; do
  if "$v/bin/python" -c "import streamlit, shap" 2>/dev/null; then source "$v/bin/activate"; break; fi
done
(sleep 4 && open http://localhost:8501) &
streamlit run app.py --server.port 8501
