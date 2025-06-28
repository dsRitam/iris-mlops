import unittest
import pickle
import numpy as np
import pandas as pd

class TestIrisModel(unittest.TestCase):
    def setUp(self):
        with open('models/week2_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        self.sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])
        self.expected_class = 'setosa'  # updated from 0 to match model output

    def test_prediction_accuracy(self):
        prediction = self.model.predict(self.sample_data)[0]
        self.assertEqual(
            str(prediction), 
            str(self.expected_class), 
            f"Expected class {self.expected_class}, got {prediction}"
        )

    def test_data_validation(self):
        sample_df = pd.DataFrame(self.sample_data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        self.assertEqual(sample_df.shape[1], 4, "Input must have 4 features")
        self.assertFalse(sample_df.isnull().values.any(), "Input contains null values")
        self.assertTrue(
            all(sample_df.dtypes == np.float64) or all(sample_df.dtypes == np.float32),
            "Features must be float64 or float32"
        )

if __name__ == '__main__':
    unittest.main()

