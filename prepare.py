import ast

import numpy as np
import pandas as pd
from scipy.stats import shapiro
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def date_to_int(date: str):
    """
    Converts a date in the format of `dd-MM-yy` to an integer.
    Accuracy is compromised for easier code (Not accounting for different number of days in month)
    """
    splitted_date = [int(n) for n in date.split("-")]
    return 375 * splitted_date[2] + 31 * splitted_date[1] + splitted_date[0]


def transform_features(new_data):
    transformed_data = new_data.copy()
    transformed_data["SpecialProperty"] = transformed_data["blood_type"].isin(("O+", "B+"))  # Pascal case replaces snake case :(
    transformed_data = transformed_data.drop("blood_type", axis=1)

    # Extracting possible symptoms and adding them as Boolean values
    possible_symptoms = transformed_data["symptoms"].unique()
    possible_symptoms = set(symptom for combination in possible_symptoms if isinstance(combination, str) for symptom in combination.split(';'))
    for symptom in possible_symptoms:
        transformed_data[symptom] = transformed_data["symptoms"].apply(lambda symptoms_str: int(symptom in symptoms_str) if isinstance(symptoms_str, str) else 0)
    transformed_data = transformed_data.drop("symptoms", axis=1)
    transformed_data["current_location_x"] = transformed_data["current_location"].apply(lambda location: ast.literal_eval(location)[0])
    transformed_data["current_location_y"] = transformed_data["current_location"].apply(lambda location: ast.literal_eval(location)[1])
    transformed_data["is_special_blood"] = transformed_data["SpecialProperty"].apply(lambda is_special: int(is_special))
    transformed_data["is_male"] = transformed_data["sex"].apply(lambda sex: int(sex == "M"))
    # transformed_data["pcr_date"] = transformed_data["pcr_date"].apply(date_to_int)
    transformed_data = transformed_data.drop(["patient_id", "current_location", "sex", "SpecialProperty"], axis=1)
    return transformed_data

def prepare_data(training_data, new_data):
    normalized_data = new_data.copy()
    normalized_data = transform_features(normalized_data)
    training_data = transform_features(training_data)
    print(list(normalized_data.columns))
    labels = list(normalized_data.columns)
    # labels.remove('spread')
    # labels.remove('risk')
    scaler_decision = dict()
    for label in labels:
        stat, p = shapiro(normalized_data[[label]])
        if p < 0.05:
            print(f"<> label {label} is not normally distribured, p-value = {p}, stat = {stat}")
            num_unique = normalized_data[label].nunique()
            if num_unique > 2 and num_unique < 30:
                normalized_data[[label]] = StandardScaler().fit(training_data[[label]]).transform(normalized_data[[label]])
                scaler_decision[label] = 'standard'
            else:
                normalized_data[[label]] = MinMaxScaler((-1, 1)).fit(training_data[[label]]).transform(normalized_data[[label]])
                scaler_decision[label] = 'minmax'
        else:
            print(f"<++> label {label} is normally distribured!, p-value = {p}, stat = {stat}")
            normalized_data[[label]] = StandardScaler().fit(training_data[[label]]).transform(normalized_data[[label]])
            scaler_decision[label] = 'standard'
    normalized_data = transform_to_angle(normalized_data, 'PCR_02', 'PCR_06')
    return normalized_data

def transform_to_angle(dataframe, feature_1, feature_2):
    dataframe.loc[:, f'{feature_1}-{feature_2}-angle'] = dataframe.apply(lambda row: np.arctan2(row[feature_1], row[feature_2]), axis=1)
    return dataframe.copy()