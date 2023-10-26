import unittest
from deepgo.models import DeepGOModel, DeepGOGATModel
import torch as th

class TestDeepGOModel(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.batch_size = 4
        self.input_length = 32
        self.nb_gos = 32
        self.nb_zero_gos = 32
        self.nb_rels = 5
        self.hidden_dim = 32
        self.embed_dim = 16
        self.device = 'cpu:0'
        self.model = DeepGOModel(
            self.input_length, self.nb_gos, self.nb_zero_gos,
            self.nb_rels, self.device, self.hidden_dim, self.embed_dim)
        
    def test_init(self):
        """
        Check if model parameters are initialized correctly
        """
        self.assertEqual(self.nb_gos, self.model.nb_gos)
        self.assertEqual(self.nb_zero_gos, self.model.nb_zero_gos)
        self.assertEqual(self.nb_rels, self.model.nb_rels)
        self.assertEqual(self.embed_dim, self.model.embed_dim)

    def test_mlp_shapes(self):
        """
        Tests if the MLPBlocks return correct output shapes
        """
        input = th.rand((self.batch_size, self.input_length), dtype=th.float32) 
        output = self.model.net(input)
        out_batch, out_dim = output.shape
        self.assertEqual(self.batch_size, out_batch)
        self.assertEqual(self.embed_dim, out_dim)

    def test_go_embedding_layers(self):
        """
        Tests the GO Embedding dimensions
        """
        ind = th.arange(self.nb_gos + self.nb_zero_gos)
        embeds = self.model.go_embed(ind)
        embed_n, embed_dim = embeds.shape
        self.assertEqual(embed_n, self.nb_gos + self.nb_zero_gos)
        self.assertEqual(embed_dim, self.embed_dim)

        rads = self.model.go_rad(ind)
        embed_n, embed_dim = rads.shape
        self.assertEqual(embed_n, self.nb_gos + self.nb_zero_gos)
        self.assertEqual(embed_dim, 1)


    def test_rel_embedding_layers(self):
        """
        Tests the GO Embedding dimensions
        """
        ind = th.arange(self.nb_rels + 1) # + 1 for hasFunction relation
        embeds = self.model.rel_embed(ind)
        embed_n, embed_dim = embeds.shape
        self.assertEqual(embed_n, self.nb_rels + 1)
        self.assertEqual(embed_dim, self.embed_dim)


        
class TestDeepGOGATModel(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.batch_size = 4
        self.input_length = 32
        self.nb_gos = 32
        self.nb_zero_gos = 32
        self.nb_rels = 5
        self.hidden_dim = 32
        self.embed_dim = 16
        self.device = 'cpu:0'
        self.model = DeepGOGATModel(
            self.input_length, self.nb_gos, self.nb_zero_gos,
            self.nb_rels, self.device, self.hidden_dim, self.embed_dim)
        
    def test_init(self):
        """
        Check if model parameters are initialized correctly
        """
        self.assertEqual(self.nb_gos, self.model.nb_gos)
        self.assertEqual(self.nb_zero_gos, self.model.nb_zero_gos)
        self.assertEqual(self.nb_rels, self.model.nb_rels)
        self.assertEqual(self.embed_dim, self.model.embed_dim)

    def test_mlp_shapes(self):
        """
        Tests if the MLPBlocks return correct output shapes
        """
        input = th.rand((self.batch_size, self.input_length), dtype=th.float32) 
        output = self.model.net1(input)
        out_batch, out_dim = output.shape
        self.assertEqual(self.batch_size, out_batch)
        self.assertEqual(self.hidden_dim, out_dim)

    def test_conv_layers(self):
        """
        Tests GATConv layer shapes
        """
        pass
    

    def test_go_embedding_layers(self):
        """
        Tests the GO Embedding dimensions
        """
        ind = th.arange(self.nb_gos + self.nb_zero_gos)
        embeds = self.model.go_embed(ind)
        embed_n, embed_dim = embeds.shape
        self.assertEqual(embed_n, self.nb_gos + self.nb_zero_gos)
        self.assertEqual(embed_dim, self.embed_dim)

        rads = self.model.go_rad(ind)
        embed_n, embed_dim = rads.shape
        self.assertEqual(embed_n, self.nb_gos + self.nb_zero_gos)
        self.assertEqual(embed_dim, 1)


    def test_rel_embedding_layers(self):
        """
        Tests the GO Embedding dimensions
        """
        ind = th.arange(self.nb_rels + 1) # + 1 for hasFunction relation
        embeds = self.model.rel_embed(ind)
        embed_n, embed_dim = embeds.shape
        self.assertEqual(embed_n, self.nb_rels + 1)
        self.assertEqual(embed_dim, self.embed_dim)


        
