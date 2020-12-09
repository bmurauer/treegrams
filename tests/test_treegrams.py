import unittest

from treegrams.transformers import TreeGramExtractor
from nltk import Tree

def test_treegram():
    transformer = TreeGramExtractor("PQGram", p=2, q=3)

    documents = [
        [
            Tree.fromstring("(ROOT (INNER (INNER (DET Der) (NOUN Hund)) "
                "(VERB jagte) (INNER (DET die) (NOUN Katze)) (. .)))")
        ]
    ]

    expected = [
        [
            "*_ROOT_*_*_INNER",
            "ROOT_INNER_*_*_INNER",
            "INNER_INNER_*_*_DET",
            "INNER_INNER_*_DET_NOUN",
            "INNER_INNER_DET_NOUN_*",
            "INNER_INNER_NOUN_*_*",
            "ROOT_INNER_*_INNER_VERB",
            "ROOT_INNER_INNER_VERB_INNER",
            "INNER_INNER_*_*_DET",
            "INNER_INNER_*_DET_NOUN",
            "INNER_INNER_DET_NOUN_*",
            "INNER_INNER_NOUN_*_*",
            "ROOT_INNER_VERB_INNER_.",
            "ROOT_INNER_INNER_._*",
            "ROOT_INNER_._*_*",
            "*_ROOT_*_INNER_*",
            "*_ROOT_INNER_*_*",
        ]
    ]
    actual = transformer.transform(documents)
    assert actual == expected
