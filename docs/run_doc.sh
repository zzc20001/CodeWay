#!/usr/bin/bash
sphinx-build -M html . _build/
python -m http.server --directory _build/html &
open http://127.0.0.1:8000/
