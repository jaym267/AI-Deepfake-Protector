"""Test package.

Present so the test modules can import shared fixture builders from
``.conftest``. Without it pytest imports each test file as a top-level module and
the relative import fails. It also makes pytest add ``backend/`` (rather than
``backend/tests/``) to ``sys.path``, which is what ``from app...`` needs.
"""
