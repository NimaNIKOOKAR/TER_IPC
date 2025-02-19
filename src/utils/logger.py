# src/utils/logger.py

import logging
import os

def setup_logging_training(log_file: str = "training_log.txt") -> logging.Logger:
    """
    Configure le logger pour écrire à la fois dans la console et dans un fichier.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Supprimer les handlers existants pour éviter les duplications
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Formatter pour les logs
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Handler pour la console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler pour le fichier
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Configuration du logger
def setup_logging_detection(log_file: str = "detection_log.txt") -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Supprimer les handlers existants
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    # Handler pour la console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # Handler pour le fichier
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
