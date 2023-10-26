import unittest
from deepgo.utils import Ontology


class TestOntologyClass(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.go = Ontology('data/go.obo', with_rels=True)

    def test_ontology_init(self):
        """
        Test if Ontology class attributes are initialized
        """
        self.assertTrue(hasattr(self.go, "ont"))
        self.assertTrue(hasattr(self.go, "ic"))
        self.assertTrue(hasattr(self.go, "ancestors"))
        self.assertTrue(hasattr(self.go, "ic_norm"))
