#!/usr/bin/python3

from __future__ import absolute_import, division, print_function, unicode_literals

import getopt
import math
import re
import sys
from collections import Counter
from typing import Iterator, Literal, TypeAlias

import nltk


class NGramLM:
    """N-gram language model."""

    def __init__(self, n: int) -> None:
        """
        Args:
            n (int): Number of grams to consider. (ie. the N in N-grams)
        """
        self.n = n
        self.occurances: Counter[str] = Counter()

    def train_on_text(self, text: str) -> None:
        """Include `text` in the model."""
        self.occurances.update(self.to_n_gram_generator(text))

    def _get_gram_log_probability(self, gram: str) -> float:
        """Gets the log-base10 of the probability for `gram` to occur."""
        if gram not in self.occurances:
            return 0
        return math.log10(self.occurances[gram] / self.occurances.total())

    def get_log_probability(self, text: str) -> float:
        """Gets the log-base10 of the probability for the grams in `text` to occur."""
        output = 0
        for gram in self.to_n_gram_generator(text):
            output += self._get_gram_log_probability(gram)
        return output

    def add_one_smoothing(self, vocab: set[str]) -> None:
        """Perform +1 occurance to each gram in `vocab` on this model instance."""
        self.occurances.update(vocab)

    def get_percent_unseen(self, text: str) -> float:
        """Gets the percentage of the grams in `text` that's not been seen by this model before."""
        num_unseen = 0
        num_grams = 0
        for gram in self.to_n_gram_generator(text):
            if gram not in self.occurances:
                num_unseen += 1
            num_grams += 1
        return num_unseen / num_grams

    def get_seen_grams(self) -> set[str]:
        """Gets the set of all the grams seen by this model instance."""
        return set(self.occurances.keys())

    def to_n_gram_generator(self, text: str) -> Iterator[str]:
        """Gets a generator that yields the N-gram substrings in a text."""
        for i in range(len(text) - self.n + 1):
            yield text[i : i + self.n]


# Unit tests.
def _unit_test_NGramLM():
    lm = NGramLM(n=4)
    TEXT = "Semua manus"
    GRAMS = ["Semu", "emua", "mua ", "ua m", "a ma", " man", "manu", "anus"]

    assert [*lm.to_n_gram_generator(TEXT)] == GRAMS
    lm.train_on_text("Semua manus")
    assert lm.get_seen_grams() == set(GRAMS)
    assert lm.get_log_probability(GRAMS[0]) == math.log10(1 / len(GRAMS))
    lm.add_one_smoothing(set(GRAMS) | set(["1111", "2222", "3333"]))
    assert lm.get_log_probability(GRAMS[0]) == math.log10(2 / (2 * len(GRAMS) + 3))


# _unit_test_NGramLM()


TextLanguage: TypeAlias = Literal["malaysian", "indonesian", "tamil", "other"]


def build_LM(in_file: str) -> tuple[NGramLM, NGramLM, NGramLM]:
    """Build language models for each label (ie. malaysian, indonesian, tamil),
    returning 3 N-gram models for malaysian, indonesian and tamil in that order.

    Each line in `in_file` contains a label and a string separated by a space.

    Args:
        in_file (str): Input training data file path.

    Returns:
        tuple[NGramLM, NGramLM, NGramLM]: N-gram models for malaysian,
            indonesian and tamil respectively.
    """

    def load_labelled_data(file_path: str) -> Iterator[tuple[TextLanguage, str]]:
        """Loads a labelled input text file line-by-line, yielding tuples
        containing the each input's language and text.
        """
        with open(file_path, "r", encoding="utf8") as file:
            for line in file:
                if line == "\n":
                    break
                language, text = line.rstrip("\n").split(" ", 1)
                yield language, text  # type: ignore

    print("building language models...")

    malaysian_lm = NGramLM(n=4)
    indonesian_lm = NGramLM(n=4)
    tamil_lm = NGramLM(n=4)

    for language, text in load_labelled_data(in_file):
        match language:
            case "malaysian":
                malaysian_lm.train_on_text(text)
            case "indonesian":
                indonesian_lm.train_on_text(text)
            case "tamil":
                tamil_lm.train_on_text(text)

    gram_vocab = (
        malaysian_lm.get_seen_grams() | indonesian_lm.get_seen_grams() | tamil_lm.get_seen_grams()
    )
    malaysian_lm.add_one_smoothing(gram_vocab)
    indonesian_lm.add_one_smoothing(gram_vocab)
    tamil_lm.add_one_smoothing(gram_vocab)

    return (malaysian_lm, indonesian_lm, tamil_lm)


def test_LM(in_file: str, out_file: str, LM: tuple[NGramLM, NGramLM, NGramLM]) -> None:
    """Test the language models on new strings, writing the most probable label
    for each string into a line in `out_file`.

    However, if the percentage of unseen grams in a string is
    `>= OTHER_PERCENT_UNSEEN_THRESHOLD`, that string will be labelled as "other".

    Each line of in_file contains a string without labels.

    Args:
        in_file (str): Input unlabelled test data file path.
        out_file (str): Output file path for predicted labels.
        LM (tuple[NGramLM, NGramLM, NGramLM]): N-gram models for malaysian,
            indonesian and tamil respectively.
    """

    def load_unlabelled_data(file_path: str) -> Iterator[str]:
        """Loads an unlabelled input text file line-by-line, yielding each input text."""
        with open(file_path, "r", encoding="utf8") as file:
            for line in file.readlines():
                if line == "\n":
                    break
                yield line.rstrip("\n")

    print("testing language models...")

    OTHER_PERCENT_UNSEEN_THRESHOLD = 0.6
    """Threshold for percent of the grams in a text being unseen, where equal or
    higher percent will classify the text as "other" language.
    """
    malaysian_lm, indonesian_lm, tamil_lm = LM

    def classify_text(text: str) -> TextLanguage:
        if malaysian_lm.get_percent_unseen(text) >= OTHER_PERCENT_UNSEEN_THRESHOLD:
            return "other"

        most_prob_lang: TextLanguage = "malaysian"
        best_prob = malaysian_lm.get_log_probability(text)

        if (prob := indonesian_lm.get_log_probability(text)) > best_prob:
            most_prob_lang = "indonesian"
            best_prob = prob

        if (prob := tamil_lm.get_log_probability(text)) > best_prob:
            most_prob_lang = "tamil"
            best_prob = prob

        return most_prob_lang

    with open(out_file, "w") as file:
        for text in load_unlabelled_data(in_file):
            language = classify_text(text)
            file.write(language + "\n")


def usage():
    print(
        "usage: "
        + sys.argv[0]
        + " -b input-file-for-building-LM -t input-file-for-testing-LM -o output-file"
    )


input_file_b = input_file_t = output_file = None
try:
    opts, args = getopt.getopt(sys.argv[1:], "b:t:o:")
except getopt.GetoptError:
    usage()
    sys.exit(2)
for o, a in opts:
    if o == "-b":
        input_file_b = a
    elif o == "-t":
        input_file_t = a
    elif o == "-o":
        output_file = a
    else:
        assert False, "unhandled option"
if input_file_b == None or input_file_t == None or output_file == None:
    usage()
    sys.exit(2)

LM = build_LM(input_file_b)
test_LM(input_file_t, output_file, LM)
