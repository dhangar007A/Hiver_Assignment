.PHONY: setup data index eval

setup:
	pip install -r requirements.txt

data:
	python src/ingest.py
	python src/preprocess.py

index:
	python src/retrieval_index.py

eval:
	python src/eval/run_eval.py
