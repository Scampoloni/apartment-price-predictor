"""Compatibility entrypoint for the price-estimator Hugging Face Space.

The implementation lives in :mod:`price_estimator.app`.
"""

from price_estimator.app import demo


if __name__ == "__main__":
    demo.launch()
