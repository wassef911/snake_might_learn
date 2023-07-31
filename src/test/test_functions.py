import unittest
from src.utils import clean_emojis


class TestCleanEmojis(unittest.TestCase):
    def test_remove_emojis(self):
        input_text_with_emojis = "Hello, world! 😊🌎❤️"
        expected_output = "Hello, world! "
        self.assertEqual(clean_emojis(input_text_with_emojis), expected_output)

    def test_no_emojis(self):
        input_text_without_emojis = "This is a test without emojis."
        self.assertEqual(
            clean_emojis(input_text_without_emojis), input_text_without_emojis
        )

    def test_only_emojis(self):
        input_text_only_emojis = "😊🌎❤️"
        expected_output = ""
        self.assertEqual(clean_emojis(input_text_only_emojis), expected_output)


if __name__ == "__main__":
    unittest.main()
