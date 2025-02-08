import pandas as pd
from pandas import DataFrame
from rdkit.Chem import Descriptors, SDMolSupplier
from sklearn.preprocessing import  FunctionTransformer
from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter, SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from my_logger import logger
import re


# =========================================  ЗАГРУЗКА ДАННЫХ  =======================================================


def error_sdf(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:  
        flag = False 
        for line in infile:
            line = line.rstrip()
            if line.startswith(">"):  # если строка начинается с ">"
                outfile.write(line + "\n") 
                flag = True # далее идет значение свойства
            elif flag:  # если значение свойства
                outfile.write(line + "\n")
                outfile.write("\n")
                flag = False
            else:
                outfile.write(line + "\n")

def standardize_molecules(mols):
    standardized_mols = []
    for mol in mols:
        if mol is None:
            continue

        try:
            # Сохранение свойств молекулы
            properties = {prop: mol.GetProp(prop) for prop in mol.GetPropNames()}

            # Удаление солей
            remover = SaltRemover.SaltRemover()
            mol = remover.StripMol(mol, dontRemoveEverything=True)

            # Нейтрализация зарядов
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)

            # Приведение к родительскому фрагменту
            parent = rdMolStandardize.FragmentParent(mol)
            mol = parent

            # Приведение к стандартной таутомерной форме
            tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
            mol = tautomer_enumerator.Canonicalize(mol)

            # Удаление стереохимии
            Chem.RemoveStereochemistry(mol)

            # Сброс изотопов
            for atom in mol.GetAtoms():
                atom.SetIsotope(0)

            # Канонизация SMILES
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))

            # Добавление и удаление гидрогенов
            mol = Chem.AddHs(mol)
            mol = Chem.RemoveHs(mol)

            # Исправление металлов
            metal_disconnector = rdMolStandardize.MetalDisconnector()
            mol = metal_disconnector.Disconnect(mol)

            # Восстановление свойств молекулы
            for prop, value in properties.items():
                mol.SetProp(prop, value)
            
            # Добавление стандартизированной молекулы в список
            standardized_mols.append(mol)

        except Exception as e:
            logger.error(f"Ошибка обработки молекулы: {e}")

    return standardized_mols # возвращаем список стандартизированных молекул-объектов


# каждая молекула проходит базовую стандартизацию
def load_molecules(file_path):
    return [
        mol
        for mol in SDMolSupplier(file_path) 
        if mol is not None
        ]

# ===================================   ПОДГОТОВКА БАЗОВЫХ ДЕСКРИПТОРОВ  ===============================================

# создаем словарь из дескриторов структуры
ConstDescriptors = {"heavy_atom_count": Descriptors.HeavyAtomCount,
                    "nhoh_count": Descriptors.NHOHCount,
                    "no_count": Descriptors.NOCount,
                    "num_h_acceptors": Descriptors.NumHAcceptors,
                    "num_h_donors": Descriptors.NumHDonors,
                    "num_heteroatoms": Descriptors.NumHeteroatoms,
                    "num_rotatable_bonds": Descriptors.NumRotatableBonds,
                    "num_valence_electrons": Descriptors.NumValenceElectrons,
                    "num_aromatic_rings": Descriptors.NumAromaticRings,
                    "num_Aliphatic_heterocycles": Descriptors.NumAliphaticHeterocycles,
                    "ring_count": Descriptors.RingCount}

# создаем словарь из физико-химических дескрипторов                            
PhisChemDescriptors = {"full_molecular_weight": Descriptors.MolWt,
                       "log_p": Descriptors.MolLogP,
                       "molecular_refractivity": Descriptors.MolMR,
                       "tspa": Descriptors.TPSA, # топологическая полярная поверхность
                       "balaban_j": Descriptors.BalabanJ,
                       }

# объединяем все дескрипторы в один словарь
descriptors = {}
descriptors.update(ConstDescriptors)
descriptors.update(PhisChemDescriptors)



# функция для генерации дескрипторов из молекул
def mol_dsc_calc(mols):
    return DataFrame({k: f(m) for k, f in descriptors.items()} 
                    for m in mols)

descriptors_names = descriptors.keys()

# функция-обертка, чтобы несколько функций в трансформер поместить
def process_molecules(mols):
    standardized_mols = standardize_molecules(mols) # сначала стандартизируем
    return mol_dsc_calc(standardized_mols)

# оформляем sklearn трансформер для использования в конвеерном моделировании (sklearn Pipeline)
descriptors_transformer = FunctionTransformer(process_molecules, 
                                              validate=False)

def get_descriptors(mols):
    return descriptors_transformer.transform(mols), descriptors_transformer

def final_data(molecules, features, target):
    """
    input: 
    output:
    """
    try: 
        logger.info(f"Количество объектов до обработки: {len(features)}")

        list_duplicates = list(features[features.duplicated()].index)
        logger.info(f"Индексы дубликатов: {list_duplicates}")
        features = features.drop(index=list_duplicates).reset_index(drop=True)
        target = target.drop(index=list_duplicates).reset_index(drop=True)
        logger.info(f"Количество объектов после обработки: {len(features)}")

        for index in reversed(list_duplicates):
            del molecules[index]
            
    except Exception as e:
        logger.error(f"{e}")

    return molecules, target
