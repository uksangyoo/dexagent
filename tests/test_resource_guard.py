import json
import sys
import tempfile
import unittest
from pathlib import Path

from dextrain.resource_guard import GIB, Limits, _group_running, exceeded, supervise


class ResourceGuardTests(unittest.TestCase):
    def test_children_cannot_escape_by_parent_exiting_first(self):
        code = "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])"
        with tempfile.TemporaryDirectory() as output:
            result = supervise(
                [sys.executable, "-c", code],
                output,
                limits=Limits(min_disk_free_gib=0, min_host_available_gib=0),
                poll_s=0.05,
                grace_s=0.2,
            )
            pid = json.loads((Path(output) / "process.json").read_text())["pid"]
            self.assertIn("child_exited_with_live_descendants", result["stop_reasons"])
            self.assertFalse(_group_running(pid))

    def test_preflight_refusal_is_recorded_without_launch(self):
        with tempfile.TemporaryDirectory() as output:
            result = supervise(
                [sys.executable, "-c", "raise RuntimeError('must not run')"],
                output,
                limits=Limits(min_host_available_gib=1e9),
            )
            self.assertFalse(result["launched"])
            self.assertFalse((Path(output) / "process.json").exists())
            self.assertIn("host_memory_headroom", result["stop_reasons"][0])
            self.assertEqual(json.loads((Path(output) / "guard_result.json").read_text()), result)

    def test_headroom_triggers_before_exhaustion(self):
        sample = dict(
            disk_free_bytes=19 * GIB,
            host_available_bytes=1024 * GIB,
            process_rss_bytes=1 * GIB,
            gpu_used_bytes=1 * GIB,
            output_bytes=0,
            elapsed_s=1,
        )
        self.assertEqual(exceeded(sample, Limits()), ["disk_headroom"])

    def test_child_is_stopped_when_wall_budget_expires(self):
        with tempfile.TemporaryDirectory() as output:
            result = supervise(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                output,
                limits=Limits(min_disk_free_gib=0, min_host_available_gib=0, max_seconds=0.1),
                poll_s=0.05,
                grace_s=0.2,
            )
        self.assertIn("wall_time_limit", result["stop_reasons"])
        self.assertLess(result["elapsed_s"], 3)
        self.assertNotEqual(result["exit_code"], 0)

    def test_os_file_limit_and_core_dump_limit_are_installed(self):
        code = "import resource; assert resource.getrlimit(resource.RLIMIT_CORE)==(0,0); assert resource.getrlimit(resource.RLIMIT_FSIZE)==(1048576,1048576)"
        with tempfile.TemporaryDirectory() as output:
            result = supervise(
                [sys.executable, "-c", code],
                output,
                limits=Limits(min_disk_free_gib=0, min_host_available_gib=0, max_file_mib=1),
                poll_s=0.05,
            )
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["stop_reasons"])
