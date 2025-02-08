from utils import standard_scaler, grid_cv, min_max_scaler, standard_scaler
from descriptors import (load_molecules, get_descriptors, final_data, error_sdf, standardize_molecules)
import pickle
import pandas as pd
from my_logger import logger
import os

logger.info("Загрузка данных...")
# =========================================  ЗАГРУЗКА ДАННЫХ  =====================================================

# --------------------------- Исправим ошибки  -----------------------------------
error_sdf("../data/train_balanced.sdf", "../data/train_balanced_correct.sdf") # всегда оставляем исходные данные
error_sdf("../data/test_balanced.sdf", "../data/test_balanced_correct.sdf")


logger.info("Загрузка данных...")
# загружаем данные
train = load_molecules("../data/train.sdf")
train_balanced = load_molecules("../data/train_balanced_correct.sdf")

test = load_molecules("../data/test.sdf")
test_balanced = load_molecules("../data/test_balanced_correct.sdf")


# ------------------ получаем таргет  ----------------------------

target_train = [m.GetIntProp('Activity') for m in train]
target_train_balanced = [m.GetIntProp('Activity') for m in train_balanced]

target_test = [m.GetIntProp('Activity') for m in test]
target_test_balanced = [m.GetIntProp('Activity') for m in test_balanced]

# ------------------ превращаем таргеты в Series ------------------------
target_train = pd.Series(target_train)
target_train_balanced = pd.Series(target_train_balanced)
target_test = pd.Series(target_test)
target_test_balanced = pd.Series(target_test_balanced)

# идет сначала стандартизация, затем получение дескрипторов (см. функцию-обертку)
# --------------- получаем базовые дескрипторы ИЗ СТАНДАРТИЗИРОВАННЫХ МОЛЕКУЛ и трансформеры ------------------------
features_train, features_train_transformer = get_descriptors(train)
features_train_balanced, features_train_balanced_transformer = get_descriptors(train_balanced)

features_test, features_test_transformer = get_descriptors(test)
features_test_balanced, features_test_balanced_transformer = get_descriptors(test_balanced)

# ---------------- обновляем молекулы из train и test, обновляем target после удаления дубликатов -----------------------------

train, target_train = final_data(train, features_train, target_train)
train_balanced, target_train_balanced = final_data(train_balanced, features_train_balanced, target_train_balanced)


test, target_test = final_data(test, features_test, target_test)
test_balanced, target_test_balanced = final_data(test_balanced, features_test_balanced, target_test_balanced)


# ======================================== МАСШТАБИРОВАНИЕ =========================================================

# с функцией как-то получше
def scale_data(features_transformer, data, name_columns, scaler_function, scaler_name):
    scaled_features, transformer = scaler_function(features_transformer, data)
    scaled_features.columns = name_columns
    return scaled_features, transformer


features_train_min_max_scaler, features_train_min_max_scaler_transformer = scale_data(features_train_transformer, train, features_train.columns, min_max_scaler, 'min_max')
features_train_standard_scaler, features_train_standard_scaler_transformer = scale_data(features_train_transformer, train, features_train.columns, standard_scaler, 'standard')


features_train_balanced_min_max_scaler, features_train_balanced_min_max_scaler_transformer = scale_data(features_train_balanced_transformer, train_balanced, features_train_balanced.columns, min_max_scaler, 'min_max')
features_train_balanced_standard_scaler, features_train_balanced_standard_scaler_transformer = scale_data(features_train_balanced_transformer, train_balanced, features_train_balanced.columns, standard_scaler, 'standard')


features_test_min_max_scaler, features_test_min_max_scaler_transformer = scale_data(features_test_transformer, test, features_test.columns, min_max_scaler, 'min_max')
features_test_standard_scaler, features_test_standard_scaler_transformer = scale_data(features_test_transformer, test, features_test.columns, standard_scaler, 'standard')


features_test_balanced_min_max_scaler, features_test_balanced_min_max_scaler_transformer = scale_data(features_test_balanced_transformer, test_balanced, features_test_balanced.columns, min_max_scaler, 'min_max')
features_test_balanced_standard_scaler, features_test_balanced_standard_scaler_transformer = scale_data(features_test_balanced_transformer, test_balanced, features_test_balanced.columns, standard_scaler, 'standard')


# ===================================== СОХРАНЕНИЕ ДАННЫХ И ТРАНСФОРМЕРОВ ==============================================

save_dir = "../data/learning"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def save_data(features_scaled, transformer, feature_name, data_name):

    features_scaled.to_csv(f"{save_dir}/{data_name}_{feature_name}_scaled.csv", index=False)

    with open(f"{save_dir}/{data_name}_{feature_name}_transformer.pkl", 'wb') as f:
        pickle.dump(transformer, f)


save_data(features_train_min_max_scaler, features_train_min_max_scaler_transformer, 'min_max', 'train')
save_data(features_train_standard_scaler, features_train_standard_scaler_transformer, 'standard', 'train')

save_data(features_train_balanced_min_max_scaler, features_train_balanced_min_max_scaler_transformer, 'min_max', 'train_balanced')
save_data(features_train_balanced_standard_scaler, features_train_balanced_standard_scaler_transformer, 'standard', 'train_balanced')

save_data(features_test_min_max_scaler, features_test_min_max_scaler_transformer, 'min_max', 'test')
save_data(features_test_standard_scaler, features_test_standard_scaler_transformer, 'standard', 'test')

save_data(features_test_balanced_min_max_scaler, features_test_balanced_min_max_scaler_transformer, 'min_max', 'test_balanced')
save_data(features_test_balanced_standard_scaler, features_test_balanced_standard_scaler_transformer, 'standard', 'test_balanced')


target_train.to_csv("../data/learning/target_train.csv", index=False)
target_train_balanced.to_csv("../data/learning/target_train_balanced.csv", index=False)
target_test.to_csv("../data/learning/target_test.csv", index=False)
target_test_balanced.to_csv("../data/learning/target_test_balanced.csv", index=False)


logger.info("============== Итоговые размеры сгенерированных данных =============================")
logger.info(f"target_train: {len(target_train)}")
logger.info(f"train_min_max_scaler: {len(features_train_min_max_scaler)}")
logger.info(f"train_standard_scaler: {len(features_train_standard_scaler)}")
logger.info(f"-----------------------------------------------------------------------------------")

logger.info(f"target_train_balanced: {len(target_train_balanced)}")
logger.info(f"train_balanced_min_max_scaler: {len(features_train_balanced_min_max_scaler)}")
logger.info(f"train_balanced_standard_scaler: {len(features_train_balanced_standard_scaler)}")
logger.info(f"-----------------------------------------------------------------------------------")

logger.info(f"target_test: {len(target_test)}")
logger.info(f"test_min_max_scaler: {len(features_test_min_max_scaler)}")
logger.info(f"test_standard_scaler: {len(features_test_standard_scaler)}")
logger.info(f"-----------------------------------------------------------------------------------")

logger.info(f"target_test_balanced: {len(target_test_balanced)}")
logger.info(f"test_balanced_min_max_scaler: {len(features_test_balanced_min_max_scaler)}")
logger.info(f"test_balanced_standard_scaler: {len(features_test_balanced_standard_scaler)}")
logger.info(f"-----------------------------------------------------------------------------------")
