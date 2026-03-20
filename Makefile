.PHONY: data train optimize dashboard test clean all

PYTHON ?= python

# --- Full Pipeline ---
all: data train optimize test

# --- Step 1: Generate synthetic hotel dataset ---
data:
	$(PYTHON) -m src.generate_data

# --- Steps 2 & 3: Fit log-log elasticity + LightGBM models ---
train: data
	$(PYTHON) -m src.log_log_model
	$(PYTHON) -m src.ml_model

# --- Steps 4, 5 & 6: Rate optimizer + scenarios + last-minute engine ---
optimize: train
	$(PYTHON) -m src.optimizer
	$(PYTHON) -m src.scenarios
	$(PYTHON) -m src.last_minute

# --- Step 7: Launch Streamlit dashboard ---
dashboard:
	streamlit run dashboard/app.py

# --- Step 8: Run tests ---
test:
	$(PYTHON) -m pytest tests/ -v

# --- Cleanup ---
clean:
	rm -rf data/*.csv models/*.csv models/*.json models/*.pkl
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
