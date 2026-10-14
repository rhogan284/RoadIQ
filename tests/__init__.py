# A regular package on purpose: ultralytics 8.4.x ships its own top-level `tests`
# package into site-packages, which shadows a namespace-package tests/ directory and
# breaks `from tests.test_repository import ...`.
