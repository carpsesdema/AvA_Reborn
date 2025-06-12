# utils/logger.py
import logging
import sys


def init_logging():
    """Initializes a professional, dual-output logging setup."""

    # --- ROOT LOGGER CONFIGURATION ---
    # This is the base configuration for all loggers.
    # We set it to DEBUG so that our own application modules can be verbose.
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s — %(name)-25s — %(levelname)-8s — %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout  # Default output
    )

    # --- SILENCE NOISY LIBRARIES ---
    # For talkative libraries, we raise their logging level to INFO or WARNING
    # so they don't flood the console with DEBUG messages.
    noisy_libraries = [
        "qasync",
        "urllib3",
        "hpack",
        "asyncio",
        "chromadb.telemetry.posthog",
        "sentence_transformers"
    ]
    for lib_name in noisy_libraries:
        logging.getLogger(lib_name).setLevel(logging.INFO)

    # You can also set a specific level for a very noisy one
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    logging.info("✅ Professional logging initialized. App logs at DEBUG, library logs at INFO/WARNING.")