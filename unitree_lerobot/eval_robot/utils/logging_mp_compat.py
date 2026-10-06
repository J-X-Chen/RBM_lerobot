def patch_logging_mp(logging_mp):
    """Normalize logging_mp API names across package versions."""
    if getattr(logging_mp, "_unitree_lerobot_compat_patched", False):
        return

    basic_config = getattr(logging_mp, "basic_config", None) or getattr(logging_mp, "basicConfig", None)
    get_logger = getattr(logging_mp, "get_logger", None) or getattr(logging_mp, "getLogger", None)

    if basic_config is not None:

        def safe_basic_config(*args, **kwargs):
            try:
                return basic_config(*args, **kwargs)
            except RuntimeError as exc:
                if "Logging system has already been started" in str(exc):
                    return None
                raise

        logging_mp.basic_config = safe_basic_config
        logging_mp.basicConfig = safe_basic_config

    if get_logger is not None:

        def safe_get_logger(name=None, level=None, *args, **kwargs):
            try:
                logger = get_logger(name, *args, **kwargs)
            except TypeError:
                logger = get_logger(name)
            if level is not None:
                logger.setLevel(level)
            return logger

        logging_mp.get_logger = safe_get_logger
        logging_mp.getLogger = safe_get_logger

    logging_mp._unitree_lerobot_compat_patched = True

    if basic_config is not None:
        logging_mp.basic_config(level=getattr(logging_mp, "INFO", 20))
