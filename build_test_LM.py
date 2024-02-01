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

    def get_seen_grams(self) -> set[str]:
        """Gets the set of all the grams seen by this model instance."""
        return set(self.occurances.keys())

    def to_n_gram_generator(self, text: str) -> Iterator[str]:
        """Gets a generator that yields the N-gram substrings in a text."""
        for i in range(len(text) - self.n + 1):
            yield text[i : i + self.n]


TextLanguage: TypeAlias = Literal["malaysian", "indonesian", "tamil"]


def load_labelled_data(file_path: str) -> Iterator[tuple[TextLanguage, str]]:
    """Loads a labelled input text file line-by-line, yielding tuples
    containing the each input's language and text.
    """
    with open(file_path, "r", encoding="utf8") as file:
        for line in file:
            language, text = line.rstrip("\n").split(" ", 1)
            yield language, text  # type: ignore


def load_unlabelled_data(file_path: str) -> Iterator[str]:
    """Loads an unlabelled input text file line-by-line, yielding each input text."""
    with open(file_path, "r", encoding="utf8") as file:
        for line in file.readlines():
            stripped = line.rstrip("\n")
            if stripped != "":
                yield line.rstrip("\n")


def build_LM(in_file: str) -> tuple[NGramLM, NGramLM, NGramLM]:
    """
    build language models for each label
    each line in in_file contains a label and a string separated by a space
    """
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
    """
    test the language models on new strings
    each line of in_file contains a string
    you should print the most probable label for each string into out_file
    """
    print("testing language models...")

    malaysian_lm, indonesian_lm, tamil_lm = LM

    with open(out_file, "w") as file:
        for text in load_unlabelled_data(in_file):
            most_prob_lang: TextLanguage = "malaysian"
            best_prob = malaysian_lm.get_log_probability(text)

            if (prob := indonesian_lm.get_log_probability(text)) > best_prob:
                most_prob_lang = "indonesian"
                best_prob = prob

            if (prob := tamil_lm.get_log_probability(text)) > best_prob:
                most_prob_lang = "tamil"
                best_prob = prob

            file.write(most_prob_lang + "\n")


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
