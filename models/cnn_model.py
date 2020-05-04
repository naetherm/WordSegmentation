# Copyright 2019-2020, University of Freiburg.
# Chair of Algorithms and Data Structures.
# Markus Näther <naetherm@informatik.uni-freiburg.de>


from layers.torch_layers import *
from models.abstract_model import AbstractModel


class CNNModel(AbstractModel):

  def __init__(
    self,  
    embedding_dim, 
    vocab_size,
    tags_size, 
    padding_idx, 
    dropout=0.1, 
    max_positions=2048, 
    bias=True, 
    convolutions=((128, 5),)*5):
    super(CNNModel, self).__init__()

    # Specify
    self.pre_embedding_dim = 2*embedding_dim
    self.embedding_dim = embedding_dim
    self.padding_idx = padding_idx
    self.vocab_size = vocab_size
    self.tags_size = tags_size
    self.bias = bias
    self.dropout = dropout
    self.learn_embedding = True

    # Create the model structure
    self.pre_embedding = nn.Embedding(self.vocab_size, self.vocab_size)

    self.embeddings = ConvEmbedding(self.vocab_size, self.embedding_dim, self.padding_idx)

    self.position_embeddings = PositionEmbedding(max_positions, self.embedding_dim, self.padding_idx, learn_embedding=self.learn_embedding)

    # Now the convolutional part
    convolutions = extend_conv_spec(convolutions)
    in_channels = convolutions[0][0]
    self.fc1 = Linear(self.embedding_dim, in_channels, dropout=dropout)
    self.projections = nn.ModuleList()
    self.convolutions = nn.ModuleList()

    self.residuals = []

    layer_in_channels = [in_channels]

    for _, (out_channels, kernel_size, residual) in enumerate(convolutions):

      if residual == 0:
        residual_dim = out_channels
      else:
        residual_dim = layer_in_channels[-residual]

      self.projections.append(
        Linear(residual_dim, out_channels) if residual_dim != out_channels else None
      )

      if kernel_size % 2 == 1:
        padding = kernel_size // 2
      else:
        padding = 0

      self.convolutions.append(
        ConvTBC(in_channels, out_channels*2, kernel_size, dropout=dropout, padding=padding)
      )
      self.residuals.append(residual)
      in_channels = out_channels
      layer_in_channels.append(out_channels)
    self.fc2 = Linear(in_channels, embedding_dim)

    # Last part: Linear projection back to the tagset
    self.hidden2tag = nn.Linear(embedding_dim, tags_size)

  def forward(self, X):
    """
    X -> the input sequence
    """
    # embed tokens and positions
    a = self.embeddings(X)
    b = self.position_embeddings(X)
    x = a + b
    x = F.dropout(x, p=self.dropout, training=True)

    input_embedding = x

    # Project to the convolution size
    x = self.fc1(x)

    # used to mask padding in the input
    encoder_padding_mask = X.eq(self.padding_idx).t() # BxT -> TxB
    if not encoder_padding_mask.any():
      encoder_padding_mask = None

    # BxTxC -> TxBxC
    x = x.transpose(0, 1)

    residuals = [x]

    # loop through all convolutions
    for proj, conv, res_layer in zip(self.projections, self.convolutions, self.residuals):
      if res_layer > 0:
        residual = residuals[-res_layer]
        residual = residual if proj is None else proj(residual)
      else:
        residual = None

      if encoder_padding_mask is not None:
        x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

      x = F.dropout(x, p=self.dropout, training=True)

      if conv.kernel_size[0] % 2 == 1:
        x = conv(x)
      else:
        padding_l = (conv.kernel_size[0] - 1) // 2
        padding_r = conv.kernel_size[0] // 2

        x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
        x = conv(x)

      x = F.glu(x, dim=2)

      if residual is not None:
        x = (x + residual) * math.sqrt(0.5)

    # Back-transformation: TxBxC -> BxTxC
    x = x.transpose(1, 0)

    x = self.fc2(x)

    if encoder_padding_mask is not None:
      encoder_padding_mask = encoder_padding_mask.t() # TxB -> BxT
      x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

    # scale gradients
    #x = GradMultiply.apply(x, 1.0 / 2.0)

    y = (x + input_embedding) * math.sqrt(0.5)

    y = self.hidden2tag(y)

    y = F.log_softmax(y, dim=-1)

    # Done
    return y