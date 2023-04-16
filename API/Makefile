APP_PORT := 5000
DOCKER_TAG := latest

.PHONY: run_app
run_app:
	python3 -m uvicorn app:app --host='0.0.0.0' --port=$(APP_PORT)

.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: download_weights
download_weights:
	wget -O weights/genre_classifier.pt https://www.dropbox.com/s/xlax7mfwmzh4rjl/genre_classifier.pt?dl=0
	wget -O weights/rotate_classifier.pt https://www.dropbox.com/s/976wwhi1531hmcg/rotate_classifier.pt?dl=0
