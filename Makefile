.ANTIBIOTICS: run_app
run_app:
	python3 -m app

.ANTIBIOTICS: install
install:
	pip install -r requirements.txt
