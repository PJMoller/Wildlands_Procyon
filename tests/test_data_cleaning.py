import unittest
from unittest.mock import patch
from flask import ctx
import pandas as pd
import sys
import os
import io

# Add project root for importing
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data_cleaning import process_data

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        # Common mock data for all tests
        self.visitor_data = pd.DataFrame({
            "SubgroupId": [1, 2],
            "Description": ["adult", "child"],
            "Date": ["2023-01-01", "2023-01-02"],
            "NumberOfUsedEntrances": [100, 150],
            #"ticket_num": [1, 2]  # Added to fix KeyError
        })

        self.weather_csv_data = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "temperature": [10, 12],
            "rain": [0, 1],
            "hum": [0, 0],
            "precipitation": [0, 0],
            "snow": [0, 0]
        })

        self.weather_xlsx = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01 10:00", "2023-01-02 11:00"]),
            "temperature": [10.0, 12.0],
            "rain": [0.0, 1.0],
            "precipitation": [0, 0],
            "hour": [10, 11]
        })

        self.holiday_data = pd.DataFrame({
            "NLNoord": ["Holiday", None],
            "NLMidden": [None, "Holiday"],
            "NLZuid": [None, None],
            "Niedersachsen": [None, None],
            "Nordrhein-Westfalen": [None, None],
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "week": [1, 2]
        })

        self.camp_data = pd.DataFrame({
            "year": [2023, 2023],
            "week": [1, 2],
            "promo_NLNoord": [1, 0],
            "promo_NLMidden": [0, 1],
            "promo_NLZuid": [0, 0],
            "promo_Nordrhein-Westfalen": [0, 0],
            "promo_Niedersachsen": [0, 0]
        })

        self.recurring_data = pd.DataFrame({
            "event_name": ["event1", "event2"],
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"])
        })

        self.ticketfam_data = pd.DataFrame({
            "Subgroup": ["adult", "child"],
            "ticket_family": ["family_adult", "family_child"],
            "ticket_num": [1, 2]
        })

    @patch("pandas.read_csv")
    @patch("pandas.read_excel")
    @patch("pandas.DataFrame.to_csv")  # just ignore writes
    def test_process_data_runs(self, mock_to_csv, mock_read_excel, mock_read_csv):
        mock_read_csv.side_effect = [self.visitor_data, self.weather_csv_data]
        mock_read_excel.side_effect = [
            self.weather_xlsx, self.holiday_data, self.camp_data,
            self.recurring_data, self.ticketfam_data
        ]

        mock_to_csv.return_value = None  # do nothing on save

        # run process_data and capture returned DataFrame
        saved_df = process_data()

        # now assert columns
        for col in ["ticket_num", "temperature", "rain_morning", "precip_morning",
                    "year", "month", "day", "week", "weekday"]:
            self.assertIn(col, saved_df.columns)

    @patch("pandas.read_csv")
    @patch("pandas.read_excel")
    @patch("pandas.DataFrame.to_csv")
    def test_process_data_output(self, mock_to_csv, mock_read_excel, mock_read_csv):
        # Setup mocks
        mock_read_csv.side_effect = [self.visitor_data, self.weather_csv_data]
        mock_read_excel.side_effect = [
            self.weather_xlsx, self.holiday_data, self.camp_data,
            self.recurring_data, self.ticketfam_data
        ]
        mock_to_csv.return_value = None

        # Capture returned DataFrame
        df = process_data()

        # Assertions
        self.assertIsNotNone(df)
        for col in ["ticket_num", "temperature", "rain_morning", "precip_morning",
                    "year", "month", "day", "week", "weekday"]:
            self.assertIn(col, df.columns)

    def test_process_data_empty_dfs(self):
        with patch("pandas.read_csv", return_value=pd.DataFrame()), \
             patch("pandas.read_excel", return_value=pd.DataFrame()), \
             patch("pandas.DataFrame.to_csv") as mock_to_csv, \
             patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:

            process_data()
            mock_to_csv.assert_not_called()
            output = mock_stdout.getvalue()
            self.assertIn("DataFrame 0 is empty", output)

            
    @patch("pandas.read_csv")
    @patch("pandas.read_excel")
    @patch("pandas.DataFrame.to_csv")
    def test_process_data_incorrect_columns(self, mock_to_csv, mock_read_excel, mock_read_csv):
        incorrect_visitor_data = pd.DataFrame({
            "WrongColumn1": [1, 2],
            "WrongColumn2": ["adult", "child"]
        })

        valid_weather_data = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "temperature": [10, 12],
            "rain": [0, 0],
            "precipitation": [0, 0],
            "hour": [10, 11]
        })

        # Mock inputs
        mock_read_csv.return_value = incorrect_visitor_data
        mock_read_excel.side_effect = [
            self.weather_xlsx,
            self.holiday_data,
            self.camp_data,
            self.recurring_data,
            self.ticketfam_data
        ]

        # Expect ValueError
        with self.assertRaises(ValueError) as ctx:
            process_data()

        self.assertIn(
            "Visitor data missing required columns",
            str(ctx.exception)
        )

        # Want to make sure nothing is saved
        mock_to_csv.assert_not_called()

if __name__ == "__main__":
    unittest.main()
