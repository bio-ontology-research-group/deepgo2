from .base import BaseDeepGOModel, Residual, MLPBlock
from torch import nn
import torch as th
from dgl.nn import GATConv

class DeepGOModel(BaseDeepGOModel):
    """
    DeepGO model with ElEmbeddings loss functions.

    Args:
        input_length (int): The number of input features
        nb_gos (int): The number of Gene Ontology (GO) classes to predict
        nb_zero_gos (int): The number of GO classes without training annotations
        nb_rels (int): The number of relations in GO axioms
        device (string): The compute device (cpu:0 or cuda:0)
        hidden_dim (int): The hidden dimension for an MLP
        embed_dim (int): Embedding dimension for GO classes and relations
        margin (float): The margin parameter of ELEmbedding method
    """

    def __init__(self, input_length, nb_gos, nb_zero_gos, nb_rels, device, hidden_dim=2560, embed_dim=2560, margin=0.1):
        super().__init__(input_length, nb_gos, nb_zero_gos, nb_rels, device, hidden_dim, embed_dim, margin)
        # MLP Layers to project the input protein
        net = []
        net.append(MLPBlock(input_length, embed_dim))
        net.append(Residual(MLPBlock(embed_dim, embed_dim)))
        self.net = nn.Sequential(*net)

    def forward(self, features):
        """
        Forward pass of the DeepGO Model.
        
        Args:
            features (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predictions after passing through DeepGOModel layers.
        """
        x = self.net(features)
        go_embed = self.go_embed(self.all_gos)
        hasFunc = self.rel_embed(self.hasFuncIndex)
        hasFuncGO = go_embed + hasFunc
        go_rad = th.abs(self.go_rad(self.all_gos).view(1, -1))
        x = th.matmul(x, hasFuncGO.T) + go_rad
        logits = th.sigmoid(x)
        return logits

    def forward_nf4(self, features):
        """
        Forward pass with Normal Form four.
        
        Args:
            features (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predictions after passing through DeepGOModel layers.
        """
        x = self.net(features)
        go_embed = self.go_embed(self.all_gos)
        hasFunc = self.rel_embed(self.hasFuncIndex)
        hasFuncGO = go_embed + hasFunc
        go_rad = th.abs(self.go_rad(self.all_gos).view(1, -1))
        x = x.unsqueeze(dim=1).expand(-1, self.nb_gos, -1)
        dst = th.linalg.norm(x - hasFuncGO, dim=2)
        logits = th.relu(dst - go_rad - self.margin)
        return logits



class DeepGOGATModel(BaseDeepGOModel):
    """
    DeepGOGAT model with ElEmbeddings loss functions.

    Args:
        input_length (int): The number of input features
        nb_gos (int): The number of Gene Ontology (GO) classes to predict
        nb_zero_gos (int): The number of GO classes without training annotations
        nb_rels (int): The number of relations in GO axioms
        device (string): The compute device (cpu:0 or gpu:0)
        hidden_dim (int): The hidden dimension for an MLP
        embed_dim (int): Embedding dimension for GO classes and relations
        margin (float): The margin parameter of ELEmbedding method
    """
    def __init__(self, input_length, nb_gos, nb_zero_gos, nb_rels, device, hidden_dim=1024, embed_dim=1024, margin=0.1):
        super().__init__(input_length, nb_gos, nb_zero_gos, nb_rels, device, hidden_dim, embed_dim, margin)

        self.net1 = MLPBlock(input_length, hidden_dim)
        self.conv1 = GATConv(hidden_dim, embed_dim, num_heads=1)
        self.net2 = nn.Sequential(
            nn.Linear(embed_dim, nb_gos),
            nn.Sigmoid())

    
    def forward(self, input_nodes, output_nodes, blocks):
        """
        Forward pass of the DeepGOGAT Model.
        
        Args:
            input_nodes (torch.Tensor): Input tensor.
            output_nodes (torch.Tensor): Input tensor.
            blocks (graphs): DGL Graphs
        Returns:
            torch.Tensor: Predictions after passing through DeepGOModel layers.
        """
        g1 = blocks[0]
        features = g1.ndata['feat']['_N']
        x = self.net1(features)
        x = self.conv1(g1, x).squeeze(dim=1)

        go_embed = self.go_embed(self.all_gos)
        hasFunc = self.rel_embed(self.hasFuncIndex)
        hasFuncGO = go_embed + hasFunc
        go_rad = th.abs(self.go_rad(self.all_gos).view(1, -1))

        x = th.matmul(x, hasFuncGO.T) + go_rad
        logits = th.sigmoid(x)
        return logits


    
class MLPModel(nn.Module):
    """
    Baseline MLP model with two fully connected layers with residual connection
    """
    
    def __init__(self, input_length, nb_gos, device, nodes=[1024,]):
        super().__init__()
        self.nb_gos = nb_gos
        net = []
        for hidden_dim in nodes:
            net.append(MLPBlock(input_length, hidden_dim))
            net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
            input_length = hidden_dim
        net.append(nn.Linear(input_length, nb_gos))
        net.append(nn.Sigmoid())
        self.net = nn.Sequential(*net)
        
    def forward(self, features):
        return self.net(features)
