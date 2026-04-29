def test_config_loads():
    import sys; sys.path.insert(0, ".")
    from src.config.config import ConfigurationManager
    m = ConfigurationManager()
    assert m.get_model_trainer_config().epochs == 50
    assert m.get_cascade_config().stage1_model_path != ""
    print("✅ Config loads correctly")
