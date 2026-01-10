.PHONY: setup bootstrap train serve all clean retrain

PYTHON=python
VENV=venv
ACTIVATE=. $(VENV)/bin/activate;
PYTHONPATH_ENV=PYTHONPATH=.

# -------------------------------------------------
# Setup environment
# -------------------------------------------------
setup:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) pip install --upgrade pip
	$(ACTIVATE) pip install -r requirements.txt

# -------------------------------------------------
# Phase 1: Bootstrap training (ONLY if no champion)
# -------------------------------------------------
bootstrap:
	mkdir -p data/models
	$(ACTIVATE) $(PYTHONPATH_ENV) $(PYTHON) src/models/bootstrap_train.py


# -------------------------------------------------
# Phase 3–4: Retrain from feedback (manual/cron)
# -------------------------------------------------
train:
	$(ACTIVATE) $(PYTHONPATH_ENV) $(PYTHON) src/models/retrain_pipeline.py

retrain:
	$(ACTIVATE) $(PYTHONPATH_ENV) $(PYTHON) src/jobs/retrain_job.py

# -------------------------------------------------
# Serve API
# -------------------------------------------------
serve:
	$(ACTIVATE) $(PYTHONPATH_ENV) $(PYTHON) -m uvicorn src.serving.api:app \
		--host 0.0.0.0 --port 8000 --reload

# -------------------------------------------------
# Full system bring-up (SAFE & CORRECT)
# -------------------------------------------------
all: setup bootstrap serve

# -------------------------------------------------
# Cleanup
# -------------------------------------------------
clean:
	rm -rf $(VENV)
	rm -rf data/models/*.pkl
	rm -rf data/models/lstm_model
	rm -rf data/stats/*.pkl
