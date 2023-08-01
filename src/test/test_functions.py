import unittest

from src.utils import _clean_emojis, _clean_hyperlinks, _clean_punctuation


class TestCleanEmojis(unittest.TestCase):
    def test_remove_emojis(self):
        input_text_with_emojis = "Hello, world! 😊🌎❤️"
        expected_output = "Hello, world! "
        self.assertEqual(_clean_emojis(input_text_with_emojis), expected_output)

    def test_no_emojis(self):
        input_text_without_emojis = "This is a test without emojis."
        self.assertEqual(
            _clean_emojis(input_text_without_emojis), input_text_without_emojis
        )

    def test_only_emojis(self):
        input_text_only_emojis = "😊🌎❤️"
        expected_output = ""
        self.assertEqual(_clean_emojis(input_text_only_emojis), expected_output)


class TestCleanHyperlinksFunction(unittest.TestCase):
    def test_remove_hyperlinks(self):
        text_with_hyperlinks = (
            "This is a <a href='https://example.com'>link</a> to the website."
        )
        expected_result = "This is a Link. to the website."
        result = _clean_hyperlinks(text_with_hyperlinks)
        self.assertEqual(
            result, expected_result, "Hyperlinks were not removed correctly."
        )

    def test_remove_urls(self):
        text_with_urls = (
            "Visit our website at https://example.com for more information."
        )
        expected_result = "Visit our website at  for more information."
        result = _clean_hyperlinks(text_with_urls)
        self.assertEqual(result, expected_result, "URLs were not removed correctly.")

    def test_remove_special_characters(self):
        text_with_special_characters = (
            "This is &gt; a test string with &#x27;special&#x2F; characters. &#62;"
        )
        expected_result = "This is  a test string with special  characters. "
        result = _clean_hyperlinks(text_with_special_characters)
        self.assertEqual(
            result, expected_result, "Special characters were not removed correctly."
        )

    def test_remove_tags(self):
        text_with_tags = "This is a <p>paragraph</p> with <i>italics</i>."
        expected_result = "This is a  paragraph  with  italics."
        result = _clean_hyperlinks(text_with_tags)
        self.assertEqual(
            result, expected_result, "HTML tags were not removed correctly."
        )

    def test_remove_newlines(self):
        text_with_newlines = "This text has\na newline."
        expected_result = "This text hasa newline."
        result = _clean_hyperlinks(text_with_newlines)
        self.assertEqual(
            result, expected_result, "Newlines were not removed correctly."
        )


class TestCleanPunctuationFunction(unittest.TestCase):
    def test_remove_punctuation(self):
        tweet_with_punctuation = "This is a test tweet!!!"
        expected_result = "test tweet"
        result = _clean_punctuation(tweet_with_punctuation)
        self.assertEqual(
            result, expected_result, "Punctuation was not removed correctly."
        )

    def test_remove_mentions(self):
        tweet_with_mentions = "@user123 Hello, @another_user! How are you?"
        expected_result = "hello"
        result = _clean_punctuation(tweet_with_mentions)
        self.assertEqual(
            result, expected_result, "Mentions were not removed correctly."
        )

    def test_remove_specific_words(self):
        tweet_with_specific_words = "Check out this amazing chatgpt model."
        expected_result = "check amazing model"
        result = _clean_punctuation(tweet_with_specific_words)
        self.assertEqual(
            result, expected_result, "Specific words were not removed correctly."
        )

    def test_remove_urls(self):
        tweet_with_urls = "Check out this link: https://example.com"
        expected_result = "check link"
        result = _clean_punctuation(tweet_with_urls)
        self.assertEqual(result, expected_result, "URLs were not removed correctly.")

    def test_remove_stopwords(self):
        tweet_with_stopwords = "This is a test tweet with some common words."
        expected_result = "test tweet common words"
        result = _clean_punctuation(tweet_with_stopwords)
        self.assertEqual(
            result, expected_result, "Stopwords were not removed correctly."
        )


if __name__ == "__main__":
    unittest.main()
