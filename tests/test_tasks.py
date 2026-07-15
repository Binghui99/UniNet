import io
import unittest
from contextlib import redirect_stdout

from uninet.task_runner import main_for_task


class TaskEntryTest(unittest.TestCase):
    def test_all_four_task_dry_runs(self):
        cases = {
            "anomaly": "data/smoke/tasks/task1_anomaly.json",
            "attack": "data/smoke/tasks/task2_attack.json",
            "iot": "data/smoke/tasks/task3_iot.json",
            "website": "data/smoke/tasks/task4_website.json",
        }
        for task, dataset in cases.items():
            with self.subTest(task=task), redirect_stdout(io.StringIO()):
                self.assertEqual(main_for_task(task, ["--dataset", dataset, "--dry-run"]), 0)

    def test_task_config_supplies_dataset(self):
        with redirect_stdout(io.StringIO()):
            code = main_for_task("iot", ["--config", "configs/tasks/task3_iot.json", "--dry-run"])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
