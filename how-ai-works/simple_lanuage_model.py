import argparse
from collections import Counter
import random
from typing import List, Dict, Any, Tuple


class NGram:
    """N-Gram language model unit. {words}"""
    def __init__(self, n: int, words: List[str] | None = None):
        self.n = n
        self.words = words[:n] if words is not None else []

    def set_words(self, words: List[str]) -> None:
        """Sets the {words} list up to length of {self.n}. Any words after the designated length will be lost"""
        self.words = words[:self.n] if len(words) > self.n else words

    def add_word(self, word: str) -> bool:
        """Tries to add {word} to the {words} list. Returns True if successful, False otherwise"""
        if len(word) > self.n:
            return False

        self.words.append(word)
        return True

    def tuple(self) -> Tuple[str]:
        return tuple(self.words[0:self.n-1])

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NGram):
            return self.n == other.n and self.words[0:self.n-1] == other.words[0:self.n-1]

        if isinstance(other, str):
            return ' '.join(self.tuple()) == other

        return False

    def __hash__(self) -> int:
        return hash(self.tuple())

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return ' '.join(self.tuple())


class TextGenerator:
    def __init__(self, text_analysis: Dict[str, Any], seed: int | None = None) -> None:
        self.text_analysis = text_analysis
        self.seed = seed

    def generate_text(self, start: str, response_length: int = 50) -> str:
        if self.seed is not None:
            random.seed(self.seed)

        dimensions = len(start.strip().split())
        text = start.strip().split()

        for i in range(response_length):
            context = tuple(text[i:dimensions+i])

            text.append(random.choice(self.text_analysis['successors'][context]))

        return ' '.join(text)



def language_analysis(filename, dimensions: int | None = None) -> Dict[str, Any]:
    unique_words = _generate_unique_words_count(filename)
    ngrams = _generate_ngrams(filename, dimensions)
    successors = _generate_successors(ngrams)

    return {
        'unique_words': unique_words,
        'ngrams': ngrams,
        'successors': successors,
    }


def _generate_unique_words_count(filename) -> Counter[Any]:
    punctuation = '.;,-“’”:?—‘!()_'
    unique_word_counts = Counter()

    with open(filename, 'r') as f:
        for line in f.readlines():
            for word in line.strip().split():
                lowered = word.lower().strip(punctuation)

                if len(lowered) <= 0:
                    continue

                unique_word_counts[lowered] += 1

    return unique_word_counts


def _generate_ngrams(filename: str, dimensions: int | None = None) -> List[NGram]:
    dimensions = dimensions or 2
    punctuation = '.;,-“’”:?—‘!()_'
    words = []
    ngrams = []

    with open(filename, 'r') as f:
        for line in f.readlines():
            words += [word.lower().strip(punctuation)
                      for word in line.strip().split()
                      if len(word) > 0]

    for i in range(len(words) - (dimensions - 1)):
        ngrams.append(NGram(dimensions, [w for w in words[i:i + dimensions]]))

    return ngrams


def _generate_successors(ngrams: List[NGram]) -> Dict[Tuple[str], Dict[str, int]]:
    successors = dict()

    for ngram in ngrams:
        ngram_t = ngram.tuple()
        if ngram_t not in successors:
            successors[ngram_t] = list()

        successors[ngram_t].append(ngram.words[ngram.n-1])

    return successors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Simple n-gram language model text generator.')
    parser.add_argument('-p', '--prompt', type=str, required=True,
                        help='Words used as prompt to the text generation')
    parser.add_argument('-l', '--response-length', type=int, default=50, help='Length of the response')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Integer used to seed the random number generator')

    return parser.parse_args()


def main():
    try:
        args = parse_args()
        text_analysis = language_analysis('data/cs_theory.txt', dimensions=len(args.prompt.strip().split())+1)
        print(TextGenerator(text_analysis, seed=args.seed).generate_text(args.prompt, response_length=args.response_length))
    except KeyError:
        print('Unable to generate text based on that prompt. Try another.')


if __name__ == '__main__':
    main()
