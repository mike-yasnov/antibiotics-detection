.ANTIBIOTICS: run
run:
	python3 -m app

.ANTIBIOTICS: install
install:
	pip install -r requirements.txt

.ANTIBIOTICS: lint
lint:
	flake8 app/
