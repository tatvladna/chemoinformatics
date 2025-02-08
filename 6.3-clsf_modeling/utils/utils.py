import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
                f1_score, balanced_accuracy_score,
                roc_auc_score, roc_curve, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import seaborn as sns
# from collections import Counter
from my_logger import logger
import psutil
from sklearn.model_selection import KFold
# ============================================  Settings ======================================================
# Настройки pandas`a
def start():
    options = {
        'display': {
            'max_columns': None,
            'max_colwidth': 25,
            'expand_frame_repr': False,  # Don't wrap to multiple pages
            'max_rows': 35,
            'max_seq_items': 50,         # Max length of printed sequence
            'precision': 4,
            'show_dimensions': False
        },
        'mode': {
            'chained_assignment': None   # Controls SettingWithCopyWarning
        }
    }

    for category, option in options.items():
        for op, value in option.items():
            pd.set_option(f'{category}.{op}', value)  # Python 3.6+


# ============================================  StandardScaler ================================================

def standard_scaler(transformer, mols):
    standard_scaler = StandardScaler()
    # scaler.fit(base_descriptors.values)
    # base_descriptors_norm = DataFrame(scaler.transform(base_descriptors.values), 
    #                    index=base_descriptors.index, columns=base_descriptors.columns)
    standard_scaler_descriptors_transformer = Pipeline(
        [('descriptors_generation', transformer), ('normalization', standard_scaler)])

    return DataFrame(standard_scaler_descriptors_transformer.fit_transform(mols)), standard_scaler_descriptors_transformer 

# ============================================  MinMaxScaler ======================================================

def min_max_scaler(transformer, mols):
    min_max_scaler = MinMaxScaler()
    # scaler.fit(base_descriptors.values)
    # base_descriptors_norm = DataFrame(scaler.transform(base_descriptors.values), 
    #                    index=base_descriptors.index, columns=base_descriptors.columns)
    min_max_scaler_descriptors_transformer = Pipeline(
        [('descriptors_generation', transformer), ('normalization', min_max_scaler)])

    return DataFrame(min_max_scaler_descriptors_transformer.fit_transform(mols)), min_max_scaler_descriptors_transformer


# =========================================== GridSearchCV =========================================================

def grid_cv(model, param_grid, scoring,
            x_train, y_train, x_test, y_test, 
            state, model_name=None, task=None,
            sampling=None, test_transformer=None, train_transformer=None, output_folder=None):

    title = model_name + sampling + task
    """
    GridSearchCV: 

    """
    if sampling == "SMOTE":
        logger.info(f"Размеры до SMOTE: {x_train.shape}, {len(y_train)}")
        smote = SMOTE(random_state=42)
        x_train, y_train = smote.fit_resample(x_train, y_train)
        logger.info(f"Размеры после SMOTE: {x_train.shape}, {len(y_train)}")
    elif sampling == "UnderSampl":
        logger.info(f"Размеры до UnderSampling: {x_train.shape}, {len(y_train)}")
        rus = RandomUnderSampler(random_state=42)
        x_train, y_train = rus.fit_resample(x_train, y_train)
        logger.info(f"Размеры после UnderSampling: {x_train.shape}, {len(y_train)}")
    elif sampling == "SMOTETomek":
        logger.info(f"Размеры до SMOTETomek: {x_train.shape}, {len(y_train)}")
        smote_tomek = SMOTETomek(random_state=42)
        x_train, y_train = smote_tomek.fit_resample(x_train, y_train)
        logger.info(f"Размеры после SMOTETomek: {x_train.shape}, {len(y_train)}")
            
    title = model_name + sampling + task

    # баланс классов
    activity_counts_train = y_train.value_counts().to_dict()
    activity_counts_test = y_test.value_counts().to_dict()
    logger.info(f"activity_counts_train: {activity_counts_train}")
    logger.info(f"activity_counts_test: {activity_counts_test}")

    # RepeatedKFold, KFold, StratifiedKFold
    # rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42) # он же просто cv

    grid = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring=scoring,
        cv=kf,
        n_jobs=4
    )

    try:
        # посмотрим сколько оперативной памяти расходует gridcv при обучении суммарно
        process = psutil.Process(os.getpid())
        grid.fit(x_train, y_train.values.ravel())
    except Exception as e:
        logger.exception("Ошибка при обучении модели: %s", e)
        exit(1)

    # количество моделей, обученных GridSearchCV (это количество комбинаций гиперпараметров = количеству моделей)
    num_models_trained = len(grid.cv_results_['params'])
    # суммарный объем ram делим на количество моделей = средний объем ram на обучение одной модели
    memory_usage = (process.memory_info().rss / (1024**2)) / num_models_trained  # переводим в Мб
    
    # ======  ЛУЧШАЯ МОДЕЛЬ ========
    best_model = grid.best_estimator_

    pipeline = Pipeline([
    ('scaler', train_transformer),  # Добавляем трансформер
    ('model', best_model)  # Добавляем модель
    ])

    os.makedirs(f"{output_folder}/models", exist_ok=True)
    pipeline_filename = f"{output_folder}/models/{title}.pkl"

    with open(pipeline_filename, 'wb') as model_file:
        pickle.dump(pipeline, model_file)
    logger.info(f"Лучшая модель сохранена в: {pipeline_filename}")

    model_size = os.path.getsize(pipeline_filename) / (1024**2) # объем памяти, которую занимает модель в виде pipeline в мб
    logger.info(f'|best params|: {grid.best_params_}')
    logger.info(f'|best fit time|: {grid.refit_time_}')

    # ------ predictions ---------
    y_pred = grid.predict(x_test)
    # ----------------------------

    # ==================  Метрики на тестовых данных  ===========================
    y_probs = grid.predict_proba(x_test)[:, 1] # вероятность для положительного класса
    f1 = f1_score(y_test, y_pred, average='weighted')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)

    logger.info(f'F1 Score: {f1}')
    logger.info(f'Balanced Accuracy: {balanced_acc}')
    logger.info(f'AUC: {auc}')

    # ==========================  Confusion Matrix  ===========================
    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'],
                annot_kws={"size": 18})
    plt.xlabel('Predicted Label', fontsize=18)
    plt.ylabel('True Label', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    
    os.makedirs(f'{output_folder}/matrix', exist_ok=True)
    plt.savefig(f'{output_folder}/matrix/{title}.png', dpi=300, bbox_inches='tight')
    plt.close() # обязательно закрываем график

    # =========================  ROC Curve  ===================================
    
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve {title} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')

    os.makedirs(f'{output_folder}/roc', exist_ok=True)
    plt.savefig(f'{output_folder}/roc/{title}.png', dpi=300, bbox_inches='tight')
    plt.close() # обязательно закрываем график

    # sensitivity = tpr # - список значений
    # specificity = 1 - fpr # cписок значений

    # ================================== Индивидуальные графики =====================================
    results = grid.cv_results_
    if model_name == "TreeClsf":
        depths = results['param_max_depth']
        accuracy_scores = results['mean_test_score']

        plt.figure(figsize=(10, 6))
        plt.plot(depths, accuracy_scores, color='g', marker="o", label='Accuracy')
        plt.xlabel('Depth of Tree', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.title(f'{title}: Change of Balanced Accuracy Score with Depth of Tree', fontsize=16)
        plt.grid(True)
        os.makedirs(f'{output_folder}/depth', exist_ok=True)
        plt.savefig(f'{output_folder}/depth/{title}.png', dpi=300, bbox_inches='tight')
        plt.close() # обязательно закрываем график

    return round(f1, 2),\
            round(balanced_acc, 2),\
            round(auc, 2),\
            round(grid.refit_time_, 2),\
            round(grid.cv_results_['mean_score_time'][grid.best_index_], 2),\
            grid.best_params_,\
            round(model_size, 2),\
            round(memory_usage, 2),\
            activity_counts_train,\
            activity_counts_test