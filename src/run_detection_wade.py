# src/run_detection_wade.py

import os
import argparse
import logging
from src.training.train_detection_wade import detect_diseased_ganglion

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description="Détecter les ganglions malades dans une image CT.")
    parser.add_argument("--image_path", type=str, required=True, help="Chemin de l'image CT (format NIfTI).")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin du modèle entraîné (best_metric_model.pth).")
    parser.add_argument("--output_dir", type=str, required=True, help="Répertoire pour sauvegarder les résultats.")
    args = parser.parse_args()

    # Vérifier si le répertoire de sortie existe, sinon le créer
    os.makedirs(args.output_dir, exist_ok=True)

    # Lancer la détection des ganglions malades
    logger.info("Démarrage de la détection des ganglions malades...")
    detect_diseased_ganglion(args.image_path, args.model_path, args.output_dir)
    logger.info("Détection des ganglions malades terminée.")

