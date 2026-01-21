# ============================================================
# run_all_v12_tests.py
# ------------------------------------------------------------
# Runs all verification tests for TH12 fixes
# Fails fast on error
# ============================================================

import subprocess
import sys
import time

TESTS = [
    ("Smoke Test (model forward)", "test_v12_smoke.py"),
    ("Parsing Robustness", "test_v12_parse.py"),
    ("Size-Aware Compare", "test_v12_size_compare.py"),
    ("Compare Stability", "test_v12_compare_stability.py"),
    ("Forward Speed Benchmark", "bench_v12_forward.py"),
]

def run_test(name, script):
    print("\n" + "=" * 60)
    print(f"▶ RUNNING: {name}")
    print("=" * 60)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    dt = time.time() - t0

    if result.returncode != 0:
        print("❌ FAILED")
        print("----- STDOUT -----")
        print(result.stdout)
        print("----- STDERR -----")
        print(result.stderr)
        sys.exit(1)

    print(result.stdout.strip())
    print(f"✅ PASSED ({dt:.2f}s)")

def main():
    print("\n🧠 TH12 FULL SYSTEM TEST SUITE")
    print("=" * 60)

    for name, script in TESTS:
        run_test(name, script)

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED")
    print("TH12 FIXES VERIFIED")
    print("=" * 60)

if __name__ == "__main__":
    main()
