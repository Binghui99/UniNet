.PHONY: install smoke task-smoke test clean-smoke

install:
	python3 -m pip install -e .

smoke:
	uninet smoke --output-dir data/smoke
	python3 tmatrix.py data/smoke/benign_browse.pcap -o data/smoke/from-pcap.json --max-tokens 256

task-smoke:
	python3 scripts/generate_task_smoke_data.py
	python3 tasks/task1_anomaly_detection.py --config configs/tasks/task1_anomaly.json --dry-run
	python3 tasks/task2_attack_identification.py --config configs/tasks/task2_attack.json --dry-run
	python3 tasks/task3_iot_device_identification.py --config configs/tasks/task3_iot.json --dry-run
	python3 tasks/task4_website_fingerprinting.py --config configs/tasks/task4_website.json --dry-run

test:
	python3 -m unittest discover -s tests -v

clean-smoke:
	python3 -c 'from pathlib import Path; import shutil; shutil.rmtree(Path("data/smoke"), ignore_errors=True)'
