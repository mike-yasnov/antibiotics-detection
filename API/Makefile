APP_PORT := 5000
DOCKER_TAG := latest
DOCKER_IMAGE := mikeyasunov/antibiotics

.PHONY: run_app
run_app:
	python3 -m uvicorn app:app --host='0.0.0.0' --port=$(APP_PORT)

.PHONY: run_app_docker -dp $(APP_PORT):$(APP_PORT) $(DOCKER_IMAGE):$(DOCKER_TAG) 
run_app:
	docker run -dp 

.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: build
build:
	docker build -f Dockerfile . --force-rm=true -t $(DOCKER_IMAGE):$(DOCKER_TAG)
