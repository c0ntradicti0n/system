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
	black *.py

test-request:
	curl -X POST -H "Content-Type: application/json" -d '{"value":false, "nonce":"asd"}' http://192.168.1.127:5000/pump/water

undo-trial:
	cd ../trial
	git --git-dir "../.watch" reset --hard

install: venv-activate
	pip install -r requirements.txt

venv:
	python3 -m venv venv

venv-activate:
	. venv/bin/activate

torch:
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
