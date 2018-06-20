DAIMLER_CONFIG = {
    "BASE_PATH": "/home/beto0607/Facu/Pedestrians/Datasets/Daimler/DaimlerBenchmark/",#PATH A DaimlerBenchmark
    "TRAINING":{
        "NEG": DAIMLER_CONFIG["BASE_PATH"] + "Data/TrainingData/NonPedestrians/",
        "POS": DAIMLER_CONFIG["BASE_PATH"] + "Data/TrainingData/Pedestrians/48x96/"
    },
    "TEST": DAIMLER_CONFIG["BASE_PATH"] + "Data/TestData/",
    "GROUND_TRUTH": DAIMLER_CONFIG["BASE_PATH"] + "GroundTruth/GroundTruth2D.db"
}

from utils import *
