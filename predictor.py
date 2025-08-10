import joblib
import json
import numpy as np
import pandas as pd
from tf_keras.models import load_model
from sklearn.preprocessing import StandardScaler


class KValuePredictor:
    def __init__(self, 
                 model_path=r'C:\\Users\\deban\\Desktop\\Imp Docs\\RaceCar\\k_predictor_model.keras',
                 scaler_x_path=r'C:\\Users\\deban\\Desktop\\Imp Docs\\RaceCar\\feature_scaler.pkl',
                 scaler_y_path=r'C:\\Users\\deban\\Desktop\\Imp Docs\\RaceCar\\target_scaler.pkl'):
        """
        Load trained k value predictor model with all preprocessing components
        """
        try:
            # Load the trained model
            self.model = load_model(model_path)
            print(f"Model loaded from {model_path}")
            
            # Load scalers
            self.scaler_X = joblib.load(scaler_x_path)
            self.scaler_y = joblib.load(scaler_y_path)
            print(f"Scalers loaded successfully")
            
        except Exception as e:
            print(f"Error loading model or scalers: {e}")
            raise

    def preprocess_input(self, input_data):
        """
        Apply the same preprocessing pipeline used during training
        """
        # Convert to DataFrame if needed
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()

        # Base numerical features (12)
        numerical_cols = [
            "tire1", "tire2", "tire3", "tire4", "humidity1", 
            "temp_surr1", "surface_rougness1", 
            "force1", "fric_coeff1", "v1", "t1", "E"
        ]

        # Add missing numerical columns as 0
        for col in numerical_cols:
            if col not in df.columns:
                df[col] = 0

        # Handle tire compound one-hot encoding (5 compounds: C1-C5)
        if 'tire_type_encoded1' in df.columns:
            # Map compound names to numbers if needed
            compound_mapping = {'C1': 0, 'C2': 1, 'C3': 2, 'C4': 3, 'C5': 4}
            
            # Convert string compounds to numbers if necessary
            if df['tire_type_encoded1'].dtype == 'object':
                df['tire_type_encoded1'] = df['tire_type_encoded1'].map(compound_mapping)
            
            # Create one-hot encoded columns for tire compounds
            for compound_idx in range(5):  # C1-C5
                df[f'tire_type_C{compound_idx+1}'] = (df['tire_type_encoded1'] == compound_idx).astype(int)
        else:
            # If tire compound not provided, default to C3 (medium compound)
            for compound_idx in range(5):
                df[f'tire_type_C{compound_idx+1}'] = 1 if compound_idx == 2 else 0  # Default to C3

        # Combine all features in the correct order
        tire_type_cols = ['tire_type_C1', 'tire_type_C2', 'tire_type_C3', 'tire_type_C4', 'tire_type_C5']
        final_cols = numerical_cols + tire_type_cols
        
        X = df[final_cols]
        
        # Verify we have the correct number of features (should be 17)
        expected_features = 17
        assert X.shape[1] == expected_features, f"Expected {expected_features} features, got {X.shape[1]}"
        
        # Scale features (this will suppress the warning by using arrays instead of DataFrames)
        X_scaled = self.scaler_X.transform(X.values)  # Use .values to get numpy array
        
        return X_scaled



    def predict(self, input_data):
        """
        Make prediction on unscaled input data and return the predicted K value
        
        Args:
            input_data: Dictionary or DataFrame with raw (unscaled) feature values
            
        Returns:
            Predicted K value as a float (in original units)
        """
        # Preprocess input (scale features)
        X_scaled = self.preprocess_input(input_data)
        
        # Make prediction on scaled features
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        
        # Transform prediction back to original K value scale
        y_pred_original = self.scaler_y.inverse_transform(y_pred_scaled)
        
        # Return as a single float value
        return float(y_pred_original[0][0])
    
    def predict_batch(self, input_data_list):
        """
        Efficient batch prediction processing multiple samples at once
        """
        import pandas as pd
        
        total = len(input_data_list)
        print(f"Starting efficient batch prediction for {total} samples...")
        
        # Convert all input data to DataFrame at once
        batch_df = pd.DataFrame(input_data_list)
        
        # Preprocess all data at once
        print("Preprocessing all samples...")
        X_scaled = self.preprocess_input(batch_df)
        
        # Make predictions on entire batch
        print("Making predictions...")
        y_pred_scaled = self.model.predict(X_scaled, verbose=1)  # verbose=1 shows progress bar
        
        # Transform back to original scale
        print("Scaling predictions back to original units...")
        y_pred_original = self.scaler_y.inverse_transform(y_pred_scaled)
        
        # Convert to list of individual predictions
        results = [float(pred[0]) for pred in y_pred_original]
        
        print(f"Batch prediction completed! Processed {total} samples.")
        return results


    def get_model_info(self):
        """
        Display basic model information
        """
        print("=" * 50)
        print("K VALUE PREDICTION MODEL")
        print("=" * 50)
        print(f"Input features expected: {self.scaler_X.n_features_in_}")
        print(f"Model architecture: {len(self.model.layers)} layers")
        print("Ready to predict K values from unscaled inputs")
        print("=" * 50)


# Example usage:
if __name__ == "__main__":
    # Initialize predictor
    predictor = KValuePredictor()
    predictor.get_model_info()
    
    # Test with sample unscaled data
    sample_input = {
        "tire1": 100, "tire2": 120, "tire3": 80, "tire4": 90,
        "humidity1": 55, "temp_surr1": 15, "surface_rougness1": 1.5,
        "force1": 30000, "fric_coeff1": 1.0, "v1": 160,
        "t1": 0.1, "tire_type_encoded1": "C2", "E": 4000000  # String compound
    }
    
    try:
        k_value = predictor.predict(sample_input)
        print(f"Predicted K value: {k_value:.2e}")
    except Exception as e:
        print(f"Prediction error: {e}")
