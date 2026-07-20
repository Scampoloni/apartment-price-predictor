def test_lightweight_imports_do_not_download_models():
    import conversational_agent.core  # noqa: F401
    import price_estimator.src.analysis  # noqa: F401
    import price_estimator.src.predict  # noqa: F401
    import room_classifier.core  # noqa: F401
