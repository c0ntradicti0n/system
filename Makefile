-include .env

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
