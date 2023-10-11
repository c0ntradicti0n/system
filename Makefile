-include .env

export COMPOSE_PROFILES=default

format:
	black **/*.py
	isort .
	cd web && npm run format


start:
	docker compose down -v
	BUILDKIT_PROGRESS=plain docker compose build
	docker compose up -d


stop:
	docker compose down -v


sskeys:
	mkdir -p certs/live/polarity.science/
	openssl req -x509 -newkey rsa:4096 -keyout ./certs/live/polarity.science/privkey.pem -out ./certs/live/polarity.science/fullchain.pem -days 365 -nodes -subj "/CN=localhost"

logs:
	docker compose logs -f

train:
	docker build -t train:latest integrator/.
	docker run -v "./models:/models" -it train:latest


viz:
	tensorboard --logdir classifier/models


hotreload-%:
	@while inotifywait -e modify,create,delete -r --include '\.(py|txt)$$' ./$*; do \
		docker compose build $*; \
		docker compose down -v -t 0 $*; \
		docker compose up  --build --force-recreate  -d $*; \
	done


mod:
	rm -rf ./.cache-cr
	rm -rf ./.cache-cr-linker
	rm -rf ./.cache-hf
	rm -rf ./.models-classifier
	rm -rf ./.chroma

	mkdir -p ./.cache-cr
	mkdir -p ./.cache-cr-linker
	mkdir -p ./.cache-hf
	mkdir -p ./.models-classifier
	mkdir -p ./.chroma

	chmod 777 ./.cache-cr
	chmod 777 ./.cache-cr-linker
	chmod 777 ./.cache-hf
	chmod 777 ./.models-classifier
	chmod 777 ./.chroma
