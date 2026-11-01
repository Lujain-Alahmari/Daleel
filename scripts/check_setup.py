from pathlib import Path
import sys
r=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(r/"src"))
from rag_dashboard_backend import verify_core
verify_core(r)
for name in ["google.genai","torch","torchvision","transformers","streamlit","faiss"]:
 __import__(name)
print("Setup imports and core hashes passed; no API calls.")
