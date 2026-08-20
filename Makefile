.PHONY: all setup data train_detector train_head tune_fheo clean

all: data train_detector train_head

setup:
	uv pip install pip-tools
	uv pip compile requirements.in -o requirements.txt
	uv pip install -r requirements.txt
	uv pip install -e .

data:
	dvc repro build_fold_ultralytics

train_detector:
	dvc repro train_detector

train_head:
	dvc repro train_head

tune_fheo:
	dvc repro tune_fheo

clean:
	rm -rf __pycache__ .pytest_cache ml-env
