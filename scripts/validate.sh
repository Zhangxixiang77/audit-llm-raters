#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q
audit-raters validate --replications 30 --output out/validation
audit-raters demo --output out/demo
