import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import NotFittedError

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vin_nonNumeric_cols=None):
        """
        Initializes the CustomTransformer.

        Parameters:
        - vin_nonNumeric_cols: list of column names that are non-numeric in the VIN data.
        """
        self.vin_nonNumeric_cols = vin_nonNumeric_cols or []
        
        # Define mappings
        self.gdp_map = {
            'Non-US': 0, 
            'Lowest GDP': 1, 
            'Lower GDP': 2, 
            'Mid-Range GDP': 3, 
            'High GDP': 4, 
            'Highest GDP': 5
        }
        
        self.bodyclass_map = {
            "Sedan/Saloon": "Sedan/Saloon",
            "Sport Utility Vehicle (SUV)/Multi-Purpose Vehicle (MPV)": "Sport Utility Vehicle (SUV)/Multi-Purpose Vehicle (MPV)",
            "Coupe": "Coupe",
            "Convertible/Cabriolet": "Convertible/Cabriolet",
            "Wagon": "Other",
            "Roadster": "Other",
            "Hatchback/Liftback/Notchback": "Other"
        }
        
        self.series_map = {
            '1': ['1'],
            '2': ['2'],
            '3': ['3'],
            '4': ['4'],
            '5': ['5'],
            '6': ['6'],
            '7': ['7'],
            'M': ['M32dr', 'M5Sedan', 'M4Coupe'],
            'X': ['X1sDrive28i', 'X1xDrive28i', 'X1xDrive35i', 'X1xDrive', 'X1Sports',
                  'X3AWD', 'X3sDrive28i', 'X3xDrive28i', 'X3xDrive28d', 'X3xDrive35i',
                  'X4xDrive28i', 'X5AWD', 'X5', 'X5sDrive35i', 'X5xDrive35i', 
                  'X5xDrive35d', 'X5xDrive50i', 'X6AWD', 'X6xDrive35i'],
            'Z': ['Z4Roadster', 'Z42dr', 'Z4sDrive28i', 'Z4sDrive35i'],
            'I': ['i3Hatchback']
        }
        
        self.economic_categories = [
            {
                "group_name": "Highest GDP",
                "state_abbreviations": ["CA", "TX", "NY"]
            },
            {
                "group_name": "High GDP",
                "state_abbreviations": ["FL", "IL", "PA", "OH", "GA", "WA", "NJ"]
            },
            {
                "group_name": "Mid-Range GDP",
                "state_abbreviations": ["NC", "MA", "VA", "MI", "CO", "MD", "TN", "AZ", "IN", "MN"]
            },
            {
                "group_name": "Lower GDP",
                "state_abbreviations": ["WI", "MO", "CT", "OR", "SC", "LA", "AL", "KY", "UT"]
            },
            {
                "group_name": "Lowest GDP",
                "state_abbreviations": ["IA", "NV", "KS", "AR", "NE", "MS", "NM", "ID"]
            }
        ]
        
        # Columns to drop after feature engineering
        self.columns_to_drop = [
            "Year", "Doors", "is_xDrive", "DisplacementCC", "DisplacementL", 
            "DisplacementCI", "EngineHP", "Model_1", "28d", "EngineCylinders_5.0"
        ]
        
        # Initialize OneHotEncoder for categorical features
        self.ohe = OneHotEncoder(drop=None, sparse=False, handle_unknown='ignore')
        self.ohe_columns = ['Model', "PlantCountry", "FuelTypePrimary", "EngineCylinders", "BodyClass"]
        self.fitted = False

    def fit(self, X, y=None):
        """
        Fit method. Learns the OneHotEncoder categories from the data.

        Parameters:
        - X: pandas DataFrame
        - y: None

        Returns:
        - self
        """
        # Make a copy to avoid modifying the original data
        X_transformed = X.copy()
        
        # Feature Engineering Steps
        X_transformed['Region'] = X_transformed["State"].apply(self._get_region).map(self.gdp_map)
        X_transformed["Mileage_per_year"] = X_transformed.apply(self._get_mileage_per_year, axis=1)
        X_transformed["Mileage"] = np.log(X_transformed["Mileage"].replace(0, np.nan))
        X_transformed["Mileage_per_year"] = np.log(X_transformed["Mileage_per_year"].replace(0, np.nan))
        
        # Handle possible infinite or NaN values after log transformation
        X_transformed["Mileage"].replace([np.inf, -np.inf], np.nan, inplace=True)
        X_transformed["Mileage_per_year"].replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # It's advisable to handle missing values here if necessary
        
        # Create additional engineered features
        X_transformed['Power_perCylinder'] = X_transformed['EngineHP'] / X_transformed['EngineCylinders']
        X_transformed['Power_perDisplacement'] = X_transformed["EngineHP"] / X_transformed["DisplacementL"]
        X_transformed['CylinderSize'] = X_transformed["DisplacementL"] / X_transformed["EngineCylinders"]
        X_transformed['TotalPowerOutput'] = X_transformed["EngineHP"] * X_transformed["EngineCylinders"]
        X_transformed['TotalPowerCapacity'] = X_transformed["DisplacementL"] * X_transformed["EngineCylinders"]
        
        # Drop the 'Year' column as per the original preprocessing
        if "Year" in X_transformed.columns:
            X_transformed.drop(columns="Year", inplace=True)
        
        # Model by Series and Engine grouping
        model_patterns = {
            'is_xDrive': 'xDrive',
            'is_sDrive': 'sDrive',
            'is_AWD': 'AWD',
            '28i': '28i',
            '35i': '35i',
            '28d': '28d',
            '35d': '35d',
            '50i': '50i'
        }
        
        for new_col, pattern in model_patterns.items():
            X_transformed[new_col] = X_transformed['Model'].str.contains(pattern).astype(int)
        
        # Map 'BodyClass' using bodyclass_map
        X_transformed["BodyClass"] = X_transformed["BodyClass"].map(self.bodyclass_map)
        
        # Group into Series using _get_series method
        X_transformed["Model"] = X_transformed["Model"].apply(self._get_series)
        
        # Fit OneHotEncoder on the specified categorical columns
        self.ohe.fit(X_transformed[self.ohe_columns])
        
        self.fitted = True
        return self

    def transform(self, X):
        """
        Apply the preprocessing steps to the input DataFrame.

        Parameters:
        - X: pandas DataFrame

        Returns:
        - X_features: pandas DataFrame after preprocessing
        """
        if not self.fitted:
            raise NotFittedError("This CustomTransformer instance is not fitted yet. Call 'fit' with appropriate data before using this transformer.")

        # Make a copy to avoid modifying the original data
        X_transformed = X.copy()
        
        # Feature Engineering
        X_transformed['Region'] = X_transformed["State"].apply(self._get_region).map(self.gdp_map)
        X_transformed["Mileage_per_year"] = X_transformed.apply(self._get_mileage_per_year, axis=1)
        
        # Log-transform 'Mileage' and 'Mileage_per_year'
        X_transformed["Mileage"] = np.log(X_transformed["Mileage"].replace(0, np.nan))
        X_transformed["Mileage_per_year"] = np.log(X_transformed["Mileage_per_year"].replace(0, np.nan))
        
        # Handle possible infinite or NaN values after log transformation
        X_transformed["Mileage"].replace([np.inf, -np.inf], np.nan, inplace=True)
        X_transformed["Mileage_per_year"].replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # It's advisable to handle missing values here if necessary
        
        # Create additional engineered features
        X_transformed['Power_perCylinder'] = X_transformed['EngineHP'] / X_transformed['EngineCylinders']
        X_transformed['Power_perDisplacement'] = X_transformed["EngineHP"] / X_transformed["DisplacementL"]
        X_transformed['CylinderSize'] = X_transformed["DisplacementL"] / X_transformed["EngineCylinders"]
        X_transformed['TotalPowerOutput'] = X_transformed["EngineHP"] * X_transformed["EngineCylinders"]
        X_transformed['TotalPowerCapacity'] = X_transformed["DisplacementL"] * X_transformed["EngineCylinders"]
        
        # Drop the 'Year' column as per the original preprocessing
        if "Year" in X_transformed.columns:
            X_transformed.drop(columns="Year", inplace=True)
        
        # Model by Series and Engine grouping
        model_patterns = {
            'is_xDrive': 'xDrive',
            'is_sDrive': 'sDrive',
            'is_AWD': 'AWD',
            '28i': '28i',
            '35i': '35i',
            '28d': '28d',
            '35d': '35d',
            '50i': '50i'
        }
        
        for new_col, pattern in model_patterns.items():
            X_transformed[new_col] = X_transformed['Model'].str.contains(pattern).astype(int)
        
        # Map 'BodyClass' using bodyclass_map
        X_transformed["BodyClass"] = X_transformed["BodyClass"].map(self.bodyclass_map)
        
        # Group into Series using _get_series method
        X_transformed["Model"] = X_transformed["Model"].apply(self._get_series)
        
        # One-Hot Encode categorical features using the fitted OneHotEncoder
        ohe_encoded = self.ohe.transform(X_transformed[self.ohe_columns])
        ohe_feature_names = self.ohe.get_feature_names_out(self.ohe_columns)
        ohe_df = pd.DataFrame(ohe_encoded, columns=ohe_feature_names, index=X_transformed.index)
        
        # Concatenate the one-hot encoded columns to the dataframe
        X_transformed = pd.concat([X_transformed.drop(columns=self.ohe_columns), ohe_df], axis=1)
        
        # Define columns to exclude from features
        exclude_cols = ["Price", "Vin", "Make", "City", "ModelVIN", "State"] + self.vin_nonNumeric_cols
        
        # Select feature columns
        feature_cols = [col for col in X_transformed.columns if col not in exclude_cols]
        
        X_features = X_transformed[feature_cols]
        
        # Drop additional columns as specified
        cols_to_drop = [col for col in self.columns_to_drop if col in X_features.columns]
        if cols_to_drop:
            X_features.drop(columns=cols_to_drop, inplace=True)
        
        # Optionally, handle missing values here (e.g., imputation)
        # For example:
        # X_features.fillna(0, inplace=True)
        
        return X_features

    def _get_series(self, model):
        """
        Maps the model to its corresponding series based on series_map.

        Parameters:
        - model: string representing the model

        Returns:
        - series: string representing the series or None if not found
        """
        for series, models in self.series_map.items():
            if model in models:
                return series
        return None

    def _get_region(self, state):
        """
        Determines the economic category of a state based on economic_categories.

        Parameters:
        - state: string representing the state abbreviation

        Returns:
        - group_name: string representing the economic category
        """
        state = state.strip().upper()
        for region_states in self.economic_categories:
            if state in region_states["state_abbreviations"]:
                return region_states["group_name"]
        return "Non-US"

    def _get_mileage_per_year(self, row):
        """
        Calculates mileage per year.

        Parameters:
        - row: pandas Series representing a row in the DataFrame

        Returns:
        - mileage_per_year: float representing mileage per year
        """
        num_years = row.get("NumOfYears", 0)
        mileage = row.get("Mileage", 0)
        if num_years == 0:
            return mileage
        else:
            return mileage / num_years
