import unittest
from unittest.mock import patch
import pandas as pd
import io
import os
import sys

# Add project root for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data_cleaning import process_data

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        # Common mock data for all tests
        self.visitor_data = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-02"],
            "groupID": [1, 2],
            "ticket_name": ["adult", "child"],
            "ticket_num": [10, 20]
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
            "ticket_family": ["family_adult", "family_child"]
        })

    @patch("pandas.read_csv")
    @patch("pandas.read_excel")
    def test_process_data_runs(self, mock_read_excel, mock_read_csv):
        mock_read_csv.side_effect = [self.visitor_data, self.weather_csv_data]
        mock_read_excel.side_effect = [
            self.weather_xlsx, self.holiday_data, self.camp_data,
            self.recurring_data, self.ticketfam_data
        ]

        # Capture DataFrame passed to to_csv
        saved_df_container = {}
        def fake_to_csv(self_df, *args, **kwargs):
            saved_df_container['df'] = self_df

        with patch("pandas.DataFrame.to_csv", new=fake_to_csv):
            process_data()

        saved_df = saved_df_container['df']
        self.assertIsInstance(saved_df, pd.DataFrame)

        # Check expected columns exist
        expected_cols = [
            "ticket_num", "temperature", "rain_morning", "precip_morning",
            "year", "month", "day", "week", "weekday"
        ]
        for col in expected_cols:
            self.assertIn(col, saved_df.columns)

    @patch("pandas.read_csv")
    @patch("pandas.read_excel")
    def test_process_data_output(self, mock_read_excel, mock_read_csv):
        mock_read_csv.side_effect = [self.visitor_data, self.weather_csv_data]
        mock_read_excel.side_effect = [
            self.weather_xlsx, self.holiday_data, self.camp_data,
            self.recurring_data, self.ticketfam_data
        ]

        saved_df_container = {}

        # the *args, **kwargs are needed because process_data may pass index, sep
        # I ignore them and just store the DataFrame for testing
        def fake_to_csv(self_df, *args, **kwargs):
            saved_df_container['df'] = self_df

        # Patching datagame.to_csv inside this test.
        with patch("pandas.DataFrame.to_csv", new=fake_to_csv):
            process_data()

        saved_df = saved_df_container['df']
        self.assertIsInstance(saved_df, pd.DataFrame)
        for col in ["ticket_num", "temperature", "rain_morning", "precip_morning",
                    "year", "month", "day", "week", "weekday"]:
            self.assertIn(col, saved_df.columns)

    def test_process_data_empty_files(self):
        with patch("pandas.read_csv", side_effect=FileNotFoundError("No file")), \
             patch("pandas.read_excel", side_effect=FileNotFoundError("No file")), \
             patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:

            process_data()
            output = mock_stdout.getvalue()
            self.assertIn("Error loading visitor data", output)

    @patch("pandas.read_csv")
    @patch("pandas.read_excel")
    def test_process_data_incorrect_columns(self, mock_read_excel, mock_read_csv):
        incorrect_visitor_data = pd.DataFrame({
            "wrong1": [1, 2],
            "wrong2": ["a", "b"],
            "wrong3": [0, 0],
            "wrong4": [0, 0]
        })
        mock_read_csv.side_effect = [incorrect_visitor_data, self.weather_csv_data]
        mock_read_excel.side_effect = [
            self.weather_xlsx, self.holiday_data, self.camp_data,
            self.recurring_data, self.ticketfam_data
        ]

        # Expect AttributeError when trying to access 'ticket_name'
        with patch("pandas.DataFrame.to_csv") as fake_to_csv:
            with self.assertRaises(AttributeError):
                process_data()
                # Making sure no file was attempted to be written.
            fake_to_csv.assert_not_called()


if __name__ == "__main__":
    unittest.main()
