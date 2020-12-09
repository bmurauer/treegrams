import json
import unittest

import nltk
import stanza

from cltc.transformers.stanford import (
    UNKNOWN_KEY,
    WORD_FEATURES_JSON,
    StanfordNlpToFieldTransformer,
    StanfordNlpToNltkTreesTransformer,
    StanfordNlpTransformer,
    StanfordWordFeatureFrequencyTransformer,
)


class TestStanford(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        #  this is a large one
        stanza.download('en')
        cls.parser = StanfordNlpTransformer('en', cpu=True)
        cls.sentences = list(cls.parser.transform(['This is a sentence.']))

    def test_parsing_tree(self):
        text = self.sentences[0]
        self.assertEqual(len(text.sentences), 1)
        self.assertEqual(len(text.sentences[0].words), 5)
        self.assertEqual(len(text.sentences[0].dependencies), 5)
        self.assertEqual(text.sentences[0].words[3].text, 'sentence')
        self.assertEqual(text.sentences[0].dependencies[2][0].text, 'sentence')
        self.assertEqual(text.sentences[0].dependencies[2][1], 'det')
        self.assertEqual(text.sentences[0].dependencies[2][2].text, 'a')

    def test_nltk_flat_lemma(self):
        transformer = StanfordNlpToFieldTransformer('lemma')
        string = list(transformer.transform(self.sentences))[0]
        self.assertEqual(string, 'this be a sentence .')

    def test_nltk_flat_upos(self):
        transformer = StanfordNlpToFieldTransformer('upos')
        string = list(transformer.transform(self.sentences))[0]
        self.assertEqual(string, 'PRON AUX DET NOUN PUNCT')

    def test_nltk_tree_deps_lemma(self):
        t = StanfordNlpToNltkTreesTransformer(['lemma', 'dependency'])
        trees = list(t.transform(self.sentences))[0]
        expected = nltk.Tree.fromstring(
            '(root (root#sentence (nsubj#this) (cop#be) (det#a) '
            '(punct#. )))'
        )
        self.assertEqual(expected, trees[0])

    def test_nltk_tree_deps_upos(self):
        transformer = StanfordNlpToNltkTreesTransformer(['upos', 'dependency'])
        trees = list(transformer.transform(self.sentences))[0]
        expected = nltk.Tree.fromstring(
            '(root (root#NOUN (nsubj#PRON) (cop#AUX) (det#DET) '
            '(punct#PUNCT )))'
        )
        self.assertEqual(expected, trees[0])

    def test_nltk_tree_nodeps_upos(self):
        transformer = StanfordNlpToNltkTreesTransformer(
            'upos',
        )
        trees = list(transformer.transform(self.sentences))[0]
        expected = nltk.Tree.fromstring(
            '(root  (NOUN (PRON ) (AUX ) (DET ) (PUNCT )))')
        self.assertEqual(expected, trees[0])

    def test_feature_tree(self):
        transformer_number = StanfordNlpToNltkTreesTransformer('Number')
        trees = list(transformer_number.transform(self.sentences))[0]
        expected = nltk.Tree.fromstring(
            '(root (Sing (Sing ) (Sing ) (_ ) (_ )))')
        self.assertEqual(expected, trees[0])

        transformer_verbform = StanfordNlpToNltkTreesTransformer('VerbForm')
        trees = list(transformer_verbform.transform(self.sentences))[0]
        # there is one finitive verb form in this sentence.
        expected = nltk.Tree.fromstring('(root (_ (_ ) (Fin ) (_ ) (_ )))')
        self.assertEqual(expected, trees[0])

    def tlat(self):
        transformer = StanfordNlpToFieldTransformer('Number')
        actual = transformer.transform(self.sentences)[0]
        # This is a sentence .
        expected = 'Sing Sing _ Sing _'
        self.assertEqual(expected, actual)

    def test_feature_freq(self):
        transformer = StanfordWordFeatureFrequencyTransformer()
        actual = transformer.fit_transform(self.sentences)[0]

        with open(WORD_FEATURES_JSON) as features_fh:
            expected = {k: 0 for k in json.load(features_fh)}
            expected.update(
                {
                    UNKNOWN_KEY: 0,
                    'Definite__Ind': 1,
                    'Mood__Ind': 1,
                    'Number__Sing': 3,
                    'Person__3': 1,
                    'PronType__Art': 1,
                    'PronType__Dem': 1,
                    'Tense__Pres': 1,
                    'VerbForm__Fin': 1,
                }
            )
            self.assertEqual(expected, actual)
