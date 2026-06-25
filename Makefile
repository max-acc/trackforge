.PHONY: clear-synthetic run

clear-synthetic:
	rm -rf src/data/synthetic/processed

run:
	python3 main.py
