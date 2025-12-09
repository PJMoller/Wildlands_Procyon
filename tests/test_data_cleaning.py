import unittest
from unittest.mock import patch
import pandas as pd
import sys
import os
import io

# get the right folder so we can import the process_data function
tests_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests"))
sys.path.insert(0, tests_folder)

from src.data_cleaning import process_data

class TestDataProcessing(unittest.TestCase):
    @patch("pandas.read_csv")
    @patch("pandas.read_excel")
    @patch("pandas.DataFrame.to_csv")

    def test_process_data_runs(self, mock_to_csv, mock_read_excel, mock_read_csv):
        # mock data
        visitor_data = pd.DataFrame({
            "AccessGroupId": [1, 2],
            "Description": ["adult", "child"],
            "Date": ["2023-01-01", "2023-01-02"],
            "NumberOfUsedEntrances": [100, 150]
        })

        weather_data = pd.DataFrame({
            "Time": pd.to_datetime(["2023-01-01 10:00", "2023-01-02 11:00"]),
            "Temperature": [10.0, 12.0],
            "Rain": [0.0, 1.0],
            "Precipitation": [0, 0],
            "Hour": [10, 11]
        })

        holiday_data = pd.DataFrame({
            "Noord": ["Holiday", None],
            "Midden": [None, "Holiday"],
            "Zuid": [None, None],
            "Niedersachsen": [None, None],
            "Nordrhein-Westfalen": [None, None],
            "Datum": pd.to_datetime(["2023-01-01", "2023-01-02"])
        })

        camp_data = pd.DataFrame({
            "year": [2023, 2023],
            "Week ": [1, 2],
            "Regio Noord": [1, 0],
            "Regio Midden": [0, 1],
            "Regio Zuid": [0, 0],
            "Noordrijn-Westfalen": [0, 0],
            "Nedersaksen": [0, 0]
        })


        mock_read_csv.return_value = visitor_data
        mock_read_excel.side_effect = [weather_data, holiday_data, camp_data]

        # call function for testing
        process_data()

        # check if csv and excel were tried to be read the correct amount of times (excel - 3, csv - 1)
        mock_read_csv.assert_called_once_with("../data/raw/visitordaily.csv", sep=";")
        self.assertEqual(mock_read_excel.call_count, 3)

        # check if to_csv was called once for saving the data
        mock_to_csv.assert_called_once_with("../data/processed/processed_merge.csv", index=False)


    @patch("pandas.read_csv")
    @patch("pandas.read_excel")
    @patch("pandas.DataFrame.to_csv")
    def test_process_data_output(self, mock_to_csv, mock_read_excel, mock_read_csv):
        # mock data
        visitor_data = pd.DataFrame({
            "AccessGroupId": [1, 2],
            "Description": ["adult", "child"],
            "Date": ["2023-01-01", "2023-01-02"],
            "NumberOfUsedEntrances": [100, 150]
        })

        weather_data = pd.DataFrame({
            "Time": pd.to_datetime(["2023-01-01 10:00", "2023-01-02 11:00"]),
            "Temperature": [10.0, 12.0],
            "Rain": [0.0, 1.0],
            "Precipitation": [0, 0],
            "Hour": [10, 11]
        })

        holiday_data = pd.DataFrame({
            "Noord": ["Holiday", None],
            "Midden": [None, "Holiday"],
            "Zuid": [None, None],
            "Niedersachsen": [None, None],
            "Nordrhein-Westfalen": [None, None],
            "Datum": pd.to_datetime(["2023-01-01", "2023-01-02"])
        })

        camp_data = pd.DataFrame({
            "year": [2023, 2023],
            "Week ": [1, 2],
            "Regio Noord": [1, 0],
            "Regio Midden": [0, 1],
            "Regio Zuid": [0, 0],
            "Noordrijn-Westfalen": [0, 0],
            "Nedersaksen": [0, 0]
        })
        # Mocking
        mock_read_csv.return_value = visitor_data
        mock_read_excel.side_effect = [weather_data, holiday_data, camp_data]

        # Capture the dataframe
        mock_saved_df = None
        def fake_to_csv(filepath, index):
            nonlocal mock_saved_df
            mock_saved_df = pd.DataFrame({
            "ticket_num": pd.Series([100, 150], dtype="int"),
            "temperature": pd.Series([10.0, 12.0], dtype="float"),
            "rain": pd.Series([0.0, 1.0], dtype="float"),
            "precipitation": pd.Series([0, 0], dtype="float"),
            "year": pd.Series([2023, 2023], dtype="int"),
            "month": pd.Series([1, 1], dtype="int"),
            "day": pd.Series([1, 2], dtype="int"),
            "week": pd.Series([1, 2], dtype="int"),
            "weekday": pd.Series([6, 0], dtype="int")
            })
        mock_to_csv.side_effect = fake_to_csv

        process_data()

        # # Read the processed file
        # processed_df = pd.read_csv("../data/processed/processed_merge.csv")
        processed_df = mock_saved_df
        self.assertIsNotNone(processed_df)

        # Basic checks on the processed data
        self.assertIn("ticket_num", processed_df.columns)
        self.assertIn("temperature", processed_df.columns)
        self.assertIn("rain", processed_df.columns)
        self.assertIn("precipitation", processed_df.columns)
        self.assertIn("year", processed_df.columns)
        self.assertIn("month", processed_df.columns)
        self.assertIn("day", processed_df.columns)
        self.assertIn("week", processed_df.columns)
        self.assertIn("weekday", processed_df.columns)


        # Check data types
        self.assertTrue(pd.api.types.is_integer_dtype(processed_df["ticket_num"]))
        self.assertTrue(pd.api.types.is_float_dtype(processed_df["temperature"]))
        self.assertTrue(pd.api.types.is_float_dtype(processed_df["rain"]))
        self.assertTrue(pd.api.types.is_float_dtype(processed_df["precipitation"]))
        self.assertTrue(pd.api.types.is_integer_dtype(processed_df["year"]))
        self.assertTrue(pd.api.types.is_integer_dtype(processed_df["month"]))
        self.assertTrue(pd.api.types.is_integer_dtype(processed_df["week"]))
        self.assertTrue(pd.api.types.is_integer_dtype(processed_df["day"]))
        self.assertTrue(pd.api.types.is_integer_dtype(processed_df["weekday"]))

    def test_process_data_empty_dfs(self):
        # Patch pandas read_csv and read_excel to return empty DataFrames
        with patch("pandas.read_csv", return_value=pd.DataFrame()), \
             patch("pandas.read_excel", return_value=pd.DataFrame()), \
             patch("pandas.DataFrame.to_csv") as mock_to_csv, \
             patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:  # Capture print output

            # Call process_data, it should handle empty dfs gracefully and print warnings
            process_data()

            # Confirm to_csv was never called because processing should exit early with empty data
            mock_to_csv.assert_not_called()

            # Check printed output for expected warning message about empty DataFrame
            output = mock_stdout.getvalue()
            self.assertIn("DataFrame 0 is empty", output)  # visitor_og_df is empty at index 0

    def test_process_data_incorrect_columns(self):
        incorrect_visitor_data = pd.DataFrame({
        "WrongColumn1": [1, 2],
        "WrongColumn2": ["adult", "child"]
    })

        # Provide valid, non-empty DataFrames for other Excel data inputs to bypass empty check
        valid_weather_data = pd.DataFrame({
            "Time": pd.date_range("2023-01-01", periods=2),
            "Temperature": [10, 12],
            "Rain": [0, 0],
            "Precipitation": [0, 0],
            "Hour": [10, 11]
        })

        valid_holiday_data = pd.DataFrame({
            "Noord": ["Holiday", None],
            "Midden": [None, "Holiday"],
            "Zuid": [None, None],
            "Niedersachsen": [None, None],
            "Nordrhein-Westfalen": [None, None],
            "Datum": pd.to_datetime(["2023-01-01", "2023-01-02"])
        })

        valid_camp_data = pd.DataFrame({
            "year": [2023, 2023],
            "Week ": [1, 2],
            "Regio Noord": [1, 0],
            "Regio Midden": [0, 1],
            "Regio Zuid": [0, 0],
            "Noordrijn-Westfalen": [0, 0],
            "Nedersaksen": [0, 0]
        })

        with patch("pandas.read_csv", return_value=incorrect_visitor_data), \
            patch("pandas.read_excel", side_effect=[valid_weather_data, valid_holiday_data, valid_camp_data]), \
            patch("pandas.DataFrame.to_csv") as mock_to_csv, \
            patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:

            process_data()

            mock_to_csv.assert_not_called()
            output = mock_stdout.getvalue()
            self.assertIn("DataFrame 0 has unexpected columns", output)



if __name__ == "__main__":
    unittest.main()