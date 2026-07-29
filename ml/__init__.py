"""Training code for the detection models.

Separate from ``backend/`` on purpose: this package produces model artifacts, the
backend consumes them. Nothing in the request path imports from here.
"""
