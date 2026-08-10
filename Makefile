.PHONY: all setup data train test clean

all: setup data train test

setup:
	uv pip install pip-tools
	uv pip compile requirements.in -o requirements.txt
	uv pip install -r requirements.txt
	uv pip install -e .

# CPU-only stages: prepare_dataset -> make_folds -> fold_calibration -> build_fold_ultralytics.
# No GPU needed; safe to run on a machine without CUDA.
data:
	dvc repro build_fold_ultralytics

# GPU stage: trains YOLOv8-seg per fold. Requires `data` to have been run first
# (locally or by whoever produced outputs/ultralytics_folds).
train:
	dvc repro train_folds

test:
	pytest tests/

clean:
	rm -rf __pycache__ .pytest_cache ml-env
