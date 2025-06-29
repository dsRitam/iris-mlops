import unittest
import pickle
import numpy as np
import pandas as pd

class TestIrisModel(unittest.TestCase):
    def setUp(self):
        
        with open('models/week2_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        self.sample_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], 
                                       columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        self.expected_class = 'setosa'  

    def test_prediction_accuracy(self):
        prediction = self.model.predict(self.sample_data)[0]
        self.assertEqual(prediction, self.expected_class, f"Expected class {self.expected_class}, got {prediction}")

    def test_data_validation(self):
        sample_df = self.sample_data
        self.assertEqual(sample_df.shape[1], 4, "Input must have 4 features")
        self.assertFalse(sample_df.isnull().values.any(), "Input contains null values")
        self.assertTrue(all(sample_df.dtypes == np.float64), "Features must be float64")

if __name__ == '__main__':
    unittest.main()

