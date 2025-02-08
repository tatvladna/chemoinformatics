from sklearn.neighbors import KNeighborsClassifier
from utils import grid_cv
import pandas as pd
from my_logger import logger
import os
import pickle
import numpy as np
RANDOM_STATE = 9012025

# ====================================  ВЫГРУЖАЕМ ДАННЫЕ ================================================
model_name = "KNNClsf"
logger.info(f"====================================  {model_name} =========================================")

folder_path = "../data/learning"
files = os.listdir(folder_path) # список всех файлов в папке

transformers = {}
features_train = {}
features_test = {}
targets = {}


for file_name in files:
    file_path = os.path.join(folder_path, file_name)

    if file_name.endswith(".pkl"):
        with open(file_path, 'rb') as pkl_file:
            transformers[file_name.replace('.pkl', '')] = pickle.load(pkl_file)
    elif file_name.endswith(".csv"):
        if "target" in file_name.lower():
            targets[file_name.replace('.csv', '')] = pd.read_csv(file_path)
        else:
            if "train" in file_name.lower():
                features_train[file_name.replace('.csv', '')] = pd.read_csv(file_path)
            else:
                features_test[file_name.replace('.csv', '')] = pd.read_csv(file_path)

logger.info(f"Загруженные трансформеры: {transformers.keys()}")
logger.info(f"Загруженные тренировочные данные: {features_train.keys()}")
logger.info(f"Загруженные тестовые данные: {features_test.keys()}")
logger.info(f"Загруженные таргеты: {targets.keys()}")


# =========================================  МОДЕЛИРОВАНИЕ ==============================================

output_folder="../knn_clsf"
os.makedirs(output_folder, exist_ok=True)

model = KNeighborsClassifier()

param_grid = {
    'n_neighbors': np.arange(5, 16, 5),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

results = []
train_transformer = None
test_transformer = None
target = None
x_test = None
y_train = None
y_test = None
task = None
sampling = None

samplings = ["MyLocal", "NonSampl", "SMOTE", "UnderSampl", "SMOTETomek"]

for title, x_train in features_train.items():
    for sampling in samplings:
        if title == 'train_balanced_min_max_scaled':
            if sampling != "MyLocal": # не сэмплируем данные, которы уже сбалансированы 
                continue
            task = "BalancedMinMaxSc"
            x_test = features_test["test_balanced_min_max_scaled"]
            y_train = targets["target_train_balanced"]
            y_test = targets["target_test_balanced"]
            train_transformer = transformers['train_balanced_min_max_transformer']
            test_transformer = transformers['test_balanced_min_max_transformer']


        elif title == 'train_balanced_standard_scaled':
            if sampling != "MyLocal": # не сэмплируем данные, которые уже сбалансированы
                continue
            task = "BalancedStdSc"
            x_test = features_test["test_balanced_standard_scaled"]
            y_train = targets["target_train_balanced"]
            y_test = targets["target_test_balanced"]
            train_transformer = transformers['train_balanced_standard_transformer']
            test_transformer = transformers['test_balanced_standard_transformer']
        
        elif title == 'train_min_max_scaled':
            if sampling == "MyLocal": # флаг "MyLocal" нужен только для изначально сбалансированных данных
                continue
            task = "MinMaxSc"
            x_test = features_test["test_min_max_scaled"]
            y_train = targets["target_train"]
            y_test = targets["target_test"]
            train_transformer = transformers['train_min_max_transformer']
            test_transformer = transformers['test_min_max_transformer']
            
        
        elif title == "train_standard_scaled":
            if sampling == "MyLocal": # флаг "MyLocal" нужен только для изначально сбалансированных данных
                continue
            task = "StdSc"
            x_test = features_test["test_standard_scaled"]
            y_train = targets["target_train"]
            y_test = targets["target_test"]
            train_transformer = transformers['train_standard_transformer']
            test_transformer = transformers['test_standard_transformer']


        log_f1, log_balanced_acc, log_auc, fit_time, predict_time, best_params, model_size, memory_usage, balance_train, balance_test  = grid_cv(model=model,
                                                                                                                                sampling = sampling,
                                                                                                                                scoring="balanced_accuracy",
                                                                                                                                param_grid=param_grid,
                                                                                                                                x_train=x_train,
                                                                                                                                y_train=y_train,
                                                                                                                                x_test=x_test,
                                                                                                                                y_test=y_test, 
                                                                                                                                state=RANDOM_STATE,
                                                                                                                                model_name=model_name,
                                                                                                                                task=task,
                                                                                                                                train_transformer=train_transformer,
                                                                                                                                test_transformer=test_transformer,
                                                                                                                                output_folder=output_folder)

        results.append({
            "model": f"{model_name}{sampling}{task}",
            "size_x_test": f"{round(len(x_test) / (len(x_train) + len(x_test)) * 100, 2)}%",
            "balance_activity_train": balance_train,
            "balance_activity_test": balance_test,
            "balanced_accuracy": log_balanced_acc,
            "f1": log_f1,
            "auc": log_auc,
            "mean_time_fit_s_cv": fit_time,
            "mean_time_predict_s_cv": predict_time,
            "size_model_pipeline_mb": model_size,
            "mean_ram_fit_mb": memory_usage,
            "grade": log_f1 >= 0.85 and log_balanced_acc >= 0.85,
            "params": best_params
        })

table_scan = pd.DataFrame(results)
txt_path = os.path.join(output_folder, 'knn_clsf.txt')
csv_path = os.path.join(output_folder, 'knn_clsf.csv')

with open(txt_path, 'w', encoding='utf-8') as file:
    file.write(table_scan.to_string(index=False))

table_scan.to_csv(csv_path, index=False, encoding='utf-8')