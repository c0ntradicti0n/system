-include .env

build:
	rsync -avz -e "ssh -i ~/.ssh/srl.deploy" ./*.* $(HOST):/home/deploy/control
	ssh -i ~/.ssh/srl.deploy $(HOST) "cd /home/deploy/control && killall screen" || echo "not running"
	ssh -i ~/.ssh/srl.deploy $(HOST) "cd /home/deploy/control && screen -L -d -m . ven/bin/activate ; pip install -r requirements.txt ; python3 logic_generator.py"

logs:
	ssh -i ~/.ssh/srl.deploy $(HOST) "cd /home/deploy/control && tail -f screenlog.0"
pi:
	ssh -i ~/.ssh/srl.deploy  $(HOST)

shift:
	mv ../product/1 ../product/_1
	mkdir ../product/1
	mv ../product/_1 ../product/1
	mv ../product/1/_1 ../product/1/1
	mv ../product/2 ../product/1
	mv ../product/3 ../product/1
	mkdir ../product/2
	mkdir ../product/3

format:
	black **/*.py

test-request:
	curl -X POST -H "Content-Type: application/json" -d '{"value":false, "nonce":"asd"}' http://192.168.1.127:5000/pump/water

undo-trial:
	cd ../trial
	git --git-dir "../.watch" reset --hard

start:

	docker compose down -v
	BUILDKIT_PROGRESS=plain docker compose build
	docker compose up -d

sskeys:
	mkdir -p certs/live/polarity.science/
	openssl req -x509 -newkey rsa:4096 -keyout ./certs/live/polarity.science/privkey.pem -out ./certs/live/polarity.science/fullchain.pem -days 365 -nodes -subj "/CN=localhost"

logs:

	docker compose logs -f
