"""
run_all.py -- convenience script to run all five problems in one go, using
the default test image (data/lena.png). Individual scripts can still be run
on their own (see the README for how to point them at a different image).

Usage:
    python run_all.py [--image PATH]
"""
import argparse
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

def run_one(qdir, image_path):
    print("\n" + "=" * 70)
    print("Running %s/main.py" % qdir)
    print("=" * 70)
    script = os.path.join(HERE, qdir, "main.py")
    old_argv = sys.argv
    sys.argv = [script, "--image", image_path]
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        sys.argv = old_argv

def main():
    parser = argparse.ArgumentParser(description="Run q1..q5 with one command")
    parser.add_argument("--image", default=os.path.join(HERE, "data", "lena.png"))
    args = parser.parse_args()
    for q in ["q1", "q2", "q3", "q4", "q5"]:
        run_one(q, args.image)
    print("\nAll done. See q1/output ... q5/output for results.")

if __name__ == "__main__":
    main()
