import argparse
import os
import pickle
# import dill
from rdkit import Chem
from rdkit.Chem import SDMolSupplier
import pandas as pd
import sys

current_dir = os.getcwd() # текущая директория
# Добавляем папку utils в путь
utils_path = os.path.join(current_dir, "utils")
sys.path.append(utils_path)
# уже из рабочей директории обычно импортируем
import descriptors
from my_logger import logger

logger.info("================================  Выпуск модели в продакшен ================================================")

parser = argparse.ArgumentParser(description='Выбор модели и обработка данных для предсказаний.')
parser.add_argument('--model', '-m', type=str, required=True, 
                    help='Выбор модели: logregr, tree_clsf, knn_clsf, forest_clsf')
parser.add_argument('--input', '-i', type=str, required=True, 
                    help='Входной sdf-файл с молекулой для предсказания logBB')

parser.add_argument('--output', '-o', type=str, 
                    help='Файл .csv с предсказаниями')
args = parser.parse_args()


if os.path.isfile(args.input):
    logger.info("file found")
else:
    logger.info(f"Error: file {args.input} not found")
    exit(1) # завершаем программу


# написать скрипт для отбора наилучшей модели.... и тогда выгружать придется единственную
if args.model == "logregr":
    with open("logregr/models/LogRegrStdSc.pkl", 'rb') as model_file:
        best_model = pickle.load(model_file)

elif args.model == "tree_clsf":
    with open("tree_clsf/models/TreeClsfMinMaxSc.pkl", 'rb') as model_file:
        best_model = pickle.load(model_file)

elif args.model == "forest_clsf":
    with open("forest_clsf/models/ForestClsfBalancedMinMaxSc.pkl", 'rb') as model_file:
        best_model = pickle.load(model_file)

elif args.model == "knn_clsf":
    with open("knn_clsf/models/KNNClsfMinMaxSc.pkl", 'rb') as model_file:
        best_model = pickle.load(model_file)

else:
    logger.error(f"Error: неизвестная модель {args.model}")
    exit(1) # завершаем программу

# считываем sdf и превращаем в объект rdkit
try:
    molecula = [mol for mol in SDMolSupplier(args.input) if mol is not None]
except Exception as e:
    logger.error(f"Error: {e}")
    exit(1)

output_folder = "predictions_models"
os.makedirs(output_folder, exist_ok=True)
predictions = best_model.predict(molecula)
output_df = pd.DataFrame(predictions, columns=[f'logBB for the {args.input} from {os.path.basename(args.input).split('.')[0]}'])
if args.output is None:
    args.output = f"{args.model}_{os.path.basename(args.input).split('.')[0]}.csv"
output_df.to_csv(f"{output_folder}/{args.output}", index=False)
logger.info(f"Результаты предсказания сохранены в {args.output}")