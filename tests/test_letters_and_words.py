import pytest

import phantom_tensors.alphabet as alphabet
import phantom_tensors.words as words
from phantom_tensors.alphabet import __all__ as all_letters
from phantom_tensors.words import __all__ as all_words


@pytest.mark.parametrize("name", all_letters)
def test_shipped_letters_are_named_ints(name: str):
    Type = getattr(alphabet, name)
    assert Type.__name__ == name
    assert Type.__supertype__ is int


@pytest.mark.parametrize("name", all_words)
def test_shipped_words_are_named_ints(name: str):
    Type = getattr(words, name)
    assert Type.__name__ == name
    assert Type.__supertype__ is int
