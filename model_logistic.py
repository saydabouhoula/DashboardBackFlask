import pickle
import pandas as pd

class LogisticModel:
    def __init__(self, model_path, thresholds):
        """Initializes the LogisticModel with the provided model_path and thresholds."""
        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)
        self.thresholds = thresholds

    def check_compliance(self, data):
        """Checks if the input data is compliant with the specified thresholds."""
        for component, (low, high) in self.thresholds.items():
            if data[component] < low or data[component] > high:
                return False
        return True

    def predict(self, input_data):
        """Predicts the compliance of the input data."""
        input_df = pd.DataFrame([input_data])
        quality_prediction = self.model.predict(input_df)[0]
        compliance_result = "conforme" if self.check_compliance(input_data) else "non conforme"
        return compliance_result
