.PHONY: clear-synthetic run

clear-synthetic:
	rm -rf data/synthetic/processed

run:
	python3 main.py
