# src/run_segmentation.py

import logging
from src.training.train_segmentation import training_main


NUM_EPOCHS = 10

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.info("Démarrage de l'entraînement...")
    training_main(NUM_EPOCHS)
    logger.info("Entraînement terminé.")