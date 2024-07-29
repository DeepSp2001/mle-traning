def test_pkg_inslattation():
    try:
        import housing_price_predictor
    except Exception as e:
        assert (
            False
        ), f"Error : {e}. housing_price_predictor packageis not installed correctly. "
