import torch as th
from torch import nn
import math


class Residual(nn.Module):
    """
    A residual layer that adds the output of a function to its input.

    Args:
        fn (nn.Module): The function to be applied to the input.

    """

    def __init__(self, fn):
        """
        Initialize the Residual layer with a given function.

        Args:
            fn (nn.Module): The function to be applied to the input.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x):
        """
        Forward pass of the Residual layer.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: The input tensor added to the result of applying the function `fn` to it.
        """
        return x + self.fn(x)


class MLPBlock(nn.Module):
    """
    A basic Multi-Layer Perceptron (MLP) block with one fully connected layer.

    Args:
        in_features (int): The number of input features.
        output_size (int): The number of output features.
        bias (boolean): Add bias to the linear layer
        layer_norm (boolean): Apply layer normalization
        dropout (float): The dropout value
        activation (nn.Module): The activation function to be applied after each fully connected layer.

    Example:
    ```python
    # Create an MLP block with 2 hidden layers and ReLU activation
    mlp_block = MLPBlock(input_size=64, output_size=10, activation=nn.ReLU())

    # Apply the MLP block to an input tensor
    input_tensor = torch.randn(32, 64)
    output = mlp_block(input_tensor)
    ```
    """
    def __init__(self, in_features, out_features, bias=True, layer_norm=True, dropout=0.1, activation=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.activation(self.linear(x))
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class BaseDeepGOModel(nn.Module):
    """
    A base DeepGO model with ElEmbeddings loss functions

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

    def __init__(self, input_length, nb_gos, nb_zero_gos, nb_rels, device, hidden_dim=2560, embed_dim=2560, margin=0.1):
        super().__init__()
        self.nb_gos = nb_gos
        self.nb_zero_gos = nb_zero_gos
        self.nb_rels = nb_rels
        # ELEmbedding Model Layers
        self.embed_dim = embed_dim
        # Create additional index for hasFunction relation
        self.hasFuncIndex = th.LongTensor([nb_rels]).to(device)
        # Embedding layer for all classes in GO
        self.go_embed = nn.Embedding(nb_gos + nb_zero_gos, embed_dim)
        self.go_norm = nn.BatchNorm1d(embed_dim)
        # Initialize embedding layers
        k = math.sqrt(1 / embed_dim)
        nn.init.uniform_(self.go_embed.weight, -k, k)
        self.go_rad = nn.Embedding(nb_gos + nb_zero_gos, 1)
        nn.init.uniform_(self.go_rad.weight, -k, k)
        self.rel_embed = nn.Embedding(nb_rels + 1, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, -k, k)
        # indices of all GO classes
        self.all_gos = th.arange(self.nb_gos).to(device) 
        self.margin = margin


    def forward(self, features):
        raise NotImplementedError
    
    def el_loss(self, go_normal_forms):
        """
        ELEmbeddings model loss for GO axioms
        Args:
            go_normal_forms (tuple): Tuple with a list of four normal form axioms in GO
        Returns:
            torch.Tensor: Loss function value
        """
        nf1, nf2, nf3, nf4 = go_normal_forms
        loss = self.nf1_loss(nf1)
        if len(nf2):
            loss += self.nf2_loss(nf2)
        if len(nf3):
            loss += self.nf3_loss(nf3)
        if len(nf4):
            loss += self.nf4_loss(nf4)
        return loss

    def class_dist(self, data):
        """
        Computes distance between two n-balls.
        Args:
           data (torch.Tensor): (N, 2)-dim array of indices of classes
        Returns:
           torch.Tensor: (N, 1)-dim array of distances
        """
        c = self.go_norm(self.go_embed(data[:, 0]))
        d = self.go_norm(self.go_embed(data[:, 1]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 1]))
        dist = th.linalg.norm(c - d, dim=1, keepdim=True) + rc - rd
        return dist
        
    def nf1_loss(self, data):
        """
        Computes first normal form (C subclassOf D) loss
        """
        pos_dist = self.class_dist(data)
        loss = th.mean(th.relu(pos_dist - self.margin))
        return loss

    def nf2_loss(self, data):
        """
        Computes second normal form (C and D subclassOf E) loss
        """
        c = self.go_norm(self.go_embed(data[:, 0]))
        d = self.go_norm(self.go_embed(data[:, 1]))
        e = self.go_norm(self.go_embed(data[:, 2]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 1]))
        re = th.abs(self.go_rad(data[:, 2]))
        
        sr = rc + rd
        dst = th.linalg.norm(c - d, dim=1, keepdim=True)
        dst2 = th.linalg.norm(e - c, dim=1, keepdim=True)
        dst3 = th.linalg.norm(e - d, dim=1, keepdim=True)
        loss = th.mean(th.relu(dst - sr - self.margin)
                    + th.relu(dst2 - rc - self.margin)
                    + th.relu(dst3 - rd - self.margin))

        return loss

    def nf3_loss(self, data):
        """
        Computes third normal form (R some C subClassOf D) loss
        """
        n = data.shape[0]
        rE = self.rel_embed(data[:, 0])
        c = self.go_norm(self.go_embed(data[:, 1]))
        d = self.go_norm(self.go_embed(data[:, 2]))
        rc = th.abs(self.go_rad(data[:, 1]))
        rd = th.abs(self.go_rad(data[:, 2]))
        
        rSomeC = c + rE
        euc = th.linalg.norm(rSomeC - d, dim=1, keepdim=True)
        loss = th.mean(th.relu(euc + rc - rd - self.margin))
        return loss


    def nf4_loss(self, data):
        """
        Computes fourth normal form (C subclassOf R some D) loss
        """
        n = data.shape[0]
        c = self.go_norm(self.go_embed(data[:, 0]))
        rE = self.rel_embed(data[:, 1])
        d = self.go_norm(self.go_embed(data[:, 2]))
        
        rc = th.abs(self.go_rad(data[:, 1]))
        rd = th.abs(self.go_rad(data[:, 2]))
        sr = rc + rd
        # c should intersect with d + r
        rSomeD = d + rE
        dst = th.linalg.norm(c - rSomeD, dim=1, keepdim=True)
        loss = th.mean(th.relu(dst - sr - self.margin))
        return loss
