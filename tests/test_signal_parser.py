# tests/test_signal_parser.py
import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from signal_parser import parse_signal

class TestSignalParser(unittest.TestCase):

    def test_valid_signal(self):
        """Tests a standard, well-formatted signal."""
        line = "09:15;EURUSD;CALL;5"
        expected = {
            "time": "09:15",
            "pair": "EURUSD",
            "direction": "CALL",
            "expiry": 5
        }
        self.assertEqual(parse_signal(line), expected)

    def test_messy_signal_with_spaces_and_separator(self):
        """Tests a signal with extra spaces and a different separator that should be cleaned."""
        line = " 03:40 i EURAUD i PUT i 1 "
        expected = {
            "time": "03:40",
            "pair": "EURAUD",
            "direction": "PUT",
            "expiry": 1
        }
        self.assertEqual(parse_signal(line), expected)

    def test_signal_with_typo_o_for_zero(self):
        """Tests a signal where 'O' is used instead of '0' in the time."""
        line = "O8:O5;GBPUSD;CALL;15"
        expected = {
            "time": "08:05",
            "pair": "GBPUSD",
            "direction": "CALL",
            "expiry": 15
        }
        self.assertEqual(parse_signal(line), expected)

    def test_invalid_signal_format(self):
        """Tests a line that is not a valid signal and should return None."""
        line = "this is not a signal"
        self.assertIsNone(parse_signal(line))

    def test_signal_with_invalid_direction(self):
        """Tests a signal with a wrong direction."""
        line = "10:00;USDJPY;BUY;5"
        self.assertIsNone(parse_signal(line))

    def test_signal_with_missing_part(self):
        """Tests a signal that is missing a component."""
        line = "11:30;EURCAD;CALL"
        self.assertIsNone(parse_signal(line))

if __name__ == '__main__':
    unittest.main()
