import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from src.transformers import CustomTransformer
from src.vin_utils import get_vin_data
import time

st.title("Узнайте за сколько вы можете продать свой BMW :)")

st.divider()

st.write("""
Приложение поможет вам узнать диапазон цены за которую вы сможете продать свой автомобиль
         основываясь на характеристиках.
""")

VIN_COLS = ['EngineCylinders', 'DisplacementL', 'DisplacementCI', 'DisplacementCC',
       'FuelTypePrimary', 'GVWR', 'EngineHP', 'Doors', 'BodyClass', 'Model',
       'PlantCountry', 'PlantCity', 'VIN',
       'Manufacturer', 'VehicleType']

FEATURES_ORDER = [
    "Vin", "Year", "Mileage", "City", "State",
    "Make", "Model", "NumOfYears", "EngineCylinders", "DisplacementL",
    "DisplacementCI", "DisplacementCC", "FuelTypePrimary", "GVWR",
    "EngineHP", "Doors", "BodyClass", "ModelVIN",
    "PlantCountry", "PlantCity", "Manufacturer", "VehicleType"
]

COLS_TO_EXCLUDE = exclude_cols = [
    "Doors", "is_xDrive", "DisplacementCC",
    "DisplacementL", "DisplacementCI", "EngineHP", "Model_1", "28d", 
    "EngineCylinders_5.0", "28i", 
    "GVWR", "PlantCity", "Manufacturer", "VehicleType"]

@st.cache_resource
def load_model():
    """
    Load the trained model from the models directory.
    """
    model_path = os.path.join("models", "best_random_forest_v1.joblib")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure the model is saved correctly.")
        return None
    model = joblib.load(model_path)
    return model

model = load_model()

if model is None:
    st.stop()



input_tab, prediction_tab = st.tabs(["Характеристики", "Предсказание"])

with input_tab:
    with st.form("Введите характеристики вашей машины:"):
        year = st.slider('Год выпуска машины:', min_value=1981, max_value=2025, value=2024, step=1)
        num_of_years = st.slider('Сколько лет в использовании:', min_value=0, max_value=30, value=5, step=1)
        mileage = st.slider('Пробег:', min_value=0, max_value=300000, value=20000, step=1000)

        # state = st.selectbox('Место продажи:', 
        #                                 ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'WA', 'NJ',
        #                                 'NC', 'MA', 'VA', 'MI', 'CO', 'MD', 'TN', 'AZ', 'IN', 'MN',
        #                                 'WI', 'MO', 'CT', 'OR', 'SC', 'LA', 'AL', 'KY', 'UT',
        #                                 'IA', 'NV', 'KS', 'AR', 'NE', 'MS', 'NM', 'ID', 'Non-US'])

        bmw_model = st.selectbox('Модель:', 
                                        ['1', '2', '3', '4', '5', '6', '7', 'M', 'X', 'Z', 'I'])
        vin = st.text_input('VIN номер:', value='WBXHT3C39H5F68219', help="Номер VIN должен быть 17-значным")
        if len(vin.strip().upper()) != 17:
            st.warning("Неправильно введен Номер VIN!")
        is_submitted = st.form_submit_button("Отправить характеристики")

        input_data = {
                "Year": year,
                "State": "",
                "Mileage": mileage,
                "NumOfYears": num_of_years,
                "Model": str(bmw_model),
                "Vin": vin,
                "City": "",
                "Make": "BMW"
            }
        input_df = pd.DataFrame(input_data, index=[0])
        vin_json = get_vin_data(vin.strip().upper())

        if is_submitted and len(vin.strip().upper()) == 17:
            # Create a DataFrame with the inputs
            
            
            
            vin_progress = st.progress(0, text='Считываем данные с VIN Номера...')
            for percent in range(100):
                time.sleep(0.02)
                vin_progress.progress(percent+1, text="Считываем данные с VIN Номера...")
            vin_progress.empty()
            vin_succes_bar = st.success("Данные успешно считаны! Перейдите в раздел \"Предсказание\" для рассчета цены.")
            time.sleep(2)
            vin_succes_bar.empty()
        
        

with prediction_tab:
    st.text("")
    predict = st.button("Рассчитать цену")
    if predict:
        vin_df = pd.DataFrame(vin_json["Results"][0], index=["Vin Data"])
        vin_df = vin_df[VIN_COLS]
        vin_numeric_cols = ["EngineCylinders", "DisplacementL", "DisplacementCI", "DisplacementCC", "EngineHP", "Doors"]
        for col in vin_numeric_cols:
            vin_df[col] = pd.to_numeric(vin_df[col])
        st.text("")
        combined_df = input_df.set_index("Vin").join(vin_df.set_index("VIN"), how="inner", rsuffix="VIN").reset_index()
        missing_features = [feat for feat in FEATURES_ORDER if feat not in combined_df.columns]
        if missing_features:
            st.error(f"Не хватает следующих характеристик: {missing_features}")
        else:
            combined_df = combined_df[FEATURES_ORDER]
            try:
                price = model.predict(combined_df)[0]
                min_price, max_price = round(price) - 1900, round(price) + 1900
                with st.container(border=True):
                    st.metric("Оптимальная цена продажи:", f"${round(price)}")
                min_price_col, max_price_col = st.columns(2)
                min_price_col.metric("Рекомендуемая минимальная цена продажи:", f"${min_price}", delta=str(round((min_price/price)* 100) - 100)+"%", border=True)
                max_price_col.metric("Рекомендуемая максимальная цена продажи:", f"${max_price}", str(round((max_price/price)*100) - 100) + "%", border=True)
            except:
                st.error("Ошибка, попробуйте еще раз!")
            



# if input["Vin"]:
#     fetch_button = st.sidebar.button("Fetch VIN data")
#     if fetch_button:
#         vin_data = get_vin_data(input_df["Vin"][0])
#         if "Results" in vin_data.keys():
#             vin_df = pd.DataFrame(vin_data["Results"][0], index=["Vin Data"])
#             vin_df = vin_df[VIN_COLS]
#             vin_numeric_cols = ["EngineCylinders", "DisplacementL", "DisplacementCI", "DisplacementCC", "EngineHP", "Doors"]
#             vin_nonNumeric_cols = [col for col in vin_df.columns if col not in vin_numeric_cols]
#             for col in vin_numeric_cols:
#                 vin_df[col] = pd.to_numeric(vin_df[col])
#             st.dataframe(vin_df)
#             combined_df = input_df.set_index("Vin").join(vin_df.set_index("VIN"), how="inner", rsuffix="VIN").reset_index()
#             missing_features = [feat for feat in FEATURES_ORDER if feat not in combined_df.columns]
#             if missing_features:
#                 st.error(f"Missing features: {missing_features}")
#             else:
#                 combined_df = combined_df[FEATURES_ORDER]
#                 st.dataframe(combined_df)
#             #st.dataframe(combined_df)
#             st.sidebar.success("Your Car's information successfully fetched! Now click predict!")
#             predict_button = st.button("Predict")
#             st.write(f"Button clicked: {predict_button}")  # Debugging line

#             if predict_button:
#                 st.write("Predict button was pressed.")  # Debugging line
#                 try:
#                     prediction = model.predict(combined_df)
#                     st.write(f"Prediction: {prediction[0]}")
#                     st.write("Successful Prediction!")
#                 except Exception as e:
#                     st.error(f"Error during prediction: {e}")


# inp_df = pd.read_csv("data/2025-01-08T14-08_export.csv").drop(columns="Unnamed: 0")
# inp_df.Model = str(inp_df.Model)
# st.dataframe(inp_df)
# st.write(inp_df.info())
# pred = model.predict(inp_df)
# st.metric(label="Predicted Price", value=round(pred[0]))