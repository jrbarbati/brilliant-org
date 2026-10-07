import os
from unittest import TestCase

from simple_lanuage_model import *
from simple_lanuage_model import _generate_unique_words_count, _generate_ngrams, _generate_successors


class TestSimpleLanguageModel(TestCase):
    def test__generate_unique_words_count(self):
        file = 'test_file.txt'
        create_test_file(file)

        result = _generate_unique_words_count(file)
        delete_test_file(file)

        self.assertEqual(58, len(result))

        # Keys are lowercased and stripped of surrounding punctuation
        self.assertIn('this', result)
        self.assertIn('line', result)
        self.assertIn('patterns', result)
        self.assertIn('own', result)
        self.assertNotIn('This', result)
        self.assertNotIn('line.', result)
        self.assertNotIn('patterns,', result)

        # Internal apostrophes are preserved
        self.assertIn("i'm", result)

        # Repeated words are counted across lines
        self.assertEqual(4, result['to'])
        self.assertEqual(3, result['will'])
        self.assertEqual(2, result['model'])
        self.assertEqual(1, result['sentences'])

    def test__generate_ngrams_2_dimensions(self):
        file = 'test_file.txt'
        create_test_file(file)

        result = _generate_ngrams(file)
        delete_test_file(file)

        self._assert_valid_ngrams(result, n=2)

    def test__generate_ngrams_4_dimensions(self):
        file = 'test_file.txt'
        create_test_file(file)

        result = _generate_ngrams(file, dimensions=4)
        delete_test_file(file)

        self._assert_valid_ngrams(result, n=4)

    def test__generate_successors(self):
        file = 'test_file.txt'
        create_test_file(file)

        ngrams = _generate_ngrams(file)
        result = _generate_successors(ngrams)
        delete_test_file(file)

        words = expected_words()
        pairs = list(zip(words, words[1:]))

        # Shape: context tuple -> ordered list of following words (repeats kept)
        self.assertIsInstance(result, dict)

        for context, following in result.items():
            self.assertIsInstance(context, tuple)
            self.assertEqual(1, len(context))
            self.assertIsInstance(following, list)
            self.assertGreater(len(following), 0)
            for successor in following:
                self.assertIsInstance(successor, str)

        # Every word that is followed by something is a key; the final word is not (it appears once, at the end)
        self.assertEqual({(w,) for w in words[:-1]}, set(result))
        self.assertNotIn(('own',), result)

        # Each adjacent pair in the text is recorded exactly as many times as it occurs
        for prev, curr in pairs:
            self.assertEqual(pairs.count((prev, curr)), result[(prev,)].count(curr))

        # Nothing is recorded that never occurred as an adjacent pair
        recorded = {(context[0], curr) for context, following in result.items() for curr in following}
        self.assertEqual(set(pairs), recorded)

        # One transition per adjacent pair, so the list lengths total W - 1
        self.assertEqual(len(words) - 1, sum(len(f) for f in result.values()))

        # A word's successor list is as long as the number of times it was followed by anything
        for context, following in result.items():
            self.assertEqual(words[:-1].count(context[0]), len(following))

        # Successors are stored in text order
        for context, following in result.items():
            self.assertEqual([curr for prev, curr in pairs if prev == context[0]], following)

        # Spot checks against the known text
        self.assertEqual(['build', 'generate', 'follow', 'write'], result[('to',)])
        self.assertEqual(['make', 'count', 'then'], result[('will',)])
        self.assertEqual(['language', 'model', 'text'], result[('the',)])
        self.assertEqual(['a'], result[('is',)])
        self.assertEqual(['analysis', 'model'], result[('language',)])

        # Repeated transitions accumulate: "it should" appears twice, "it will" once
        self.assertEqual(['should', 'will', 'should'], result[('it',)])
        self.assertEqual(Counter({'should': 2, 'will': 1}), Counter(result[('it',)]))

    def test__generate_successors_3_dimensions(self):
        file = 'test_file.txt'
        create_test_file(file)

        ngrams = _generate_ngrams(file, dimensions=3)
        result = _generate_successors(ngrams)
        delete_test_file(file)

        words = expected_words()
        triples = list(zip(words, words[1:], words[2:]))

        # Keys are two-word contexts; every adjacent pair except the final one is followed by something
        self.assertEqual({(a, b) for a, b, _ in triples}, set(result))
        self.assertNotIn(('its', 'own'), result)
        for context in result:
            self.assertEqual(2, len(context))

        # Each triple in the text is recorded exactly as many times as it occurs
        for a, b, c in triples:
            self.assertEqual(triples.count((a, b, c)), result[(a, b)].count(c))

        # One transition per triple, so the list lengths total W - 2
        self.assertEqual(len(words) - 2, sum(len(f) for f in result.values()))

        # Spot checks: a two-word context disambiguates what a single word cannot
        self.assertEqual(['will'], result[('the', 'model')])
        self.assertEqual(['analysis'], result[('the', 'language')])
        self.assertEqual(['that'], result[('language', 'model')])
        self.assertEqual(['own'], result[('on', 'its')])

        # The only repeated context: "it should" is followed by "i'm" (across a line break) and "be", in that order
        self.assertEqual(["i'm", 'be'], result[('it', 'should')])

    def _assert_valid_ngrams(self, result, n: int) -> None:
        """Checks that {result} is the complete, ordered set of n-grams for the test text."""
        words = expected_words()

        # An n-gram is a window of n words, so a text of W words has W - n + 1 windows
        self.assertEqual(len(words) - n + 1, len(result))

        # Every n-gram holds exactly n words
        for ngram in result:
            self.assertEqual(n, ngram.n)
            self.assertEqual(n, len(ngram.words))

        # Each n-gram is the contiguous slice of the text starting at its position
        for i, ngram in enumerate(result):
            self.assertEqual(words[i:i + n], ngram.words)

        # The window slides one word at a time, so neighbors overlap by n - 1 words
        for prev, curr in zip(result, result[1:]):
            self.assertEqual(prev.words[1:], curr.words[:-1])

        # The first n-gram starts the text and the last one ends it
        self.assertEqual(words[:n], result[0].words)
        self.assertEqual(words[-n:], result[-1].words)

        # The windows reconstruct the full text: first n-gram plus the last word of each following one
        reconstructed = result[0].words + [ngram.words[-1] for ngram in result[1:]]
        self.assertEqual(words, reconstructed)


TEST_LINES = [
    'This is a test file line.',
    'These tests will make sure the language analysis works like it should.',
    'I\'m hoping to build a small language model that can be used to generate some text.',
    'The model will count how often each word appears in the text.',
    'It will then look at which words tend to follow one another.',
    '',
    'Using those patterns, it should be able to write a few new sentences on its own.'
]


def expected_words() -> List[str]:
    """Independent tokenization of TEST_LINES: lowercase, surrounding punctuation removed, line breaks ignored."""
    return [word.lower().strip('.,') for line in TEST_LINES for word in line.split()]


def create_test_file(filename: str) -> str:
    with open(filename, 'w') as f:
        f.write('\n'.join(TEST_LINES))

    return filename

def delete_test_file(filename: str) -> bool:
    try:
        os.remove(filename)
        return True
    except:
        return False
