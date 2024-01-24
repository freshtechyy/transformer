import torch
import math

class MultiHeadAttention(torch.nn.Module):
  def __init__(self, d_model, num_heads):
    """
    d_model: model embedding dimension
    num_heads: number of attention heads
    """
    super(MultiHeadAttention, self).__init__()
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    self.d_model = d_model
    self.num_heads = num_heads
    # embedding dimension per head
    self.d_head = d_model // num_heads
    
    self.W_q = torch.nn.Linear(d_model, d_model)
    self.W_k = torch.nn.Linear(d_model, d_model)
    self.W_v = torch.nn.Linear(d_model, d_model)
    self.W_o = torch.nn.Linear(d_model, d_model)

  def reshape_heads(self, x):
    batch_size, seq_len, _ = x.size()
    return x.view(batch_size, seq_len, 
                  self.num_heads,
                  self.d_head).transpose(1, 2)

  def scaled_dot_product_attention(self, Q, K, V, mask=None):
    """
    Compute the scaled dot product attention.
    Q: query sequences, shape: (bs, num_heads, src_seq_len/tgt_seq_len, d_head)
    K: key sequences, shape: (bs, num_heads, src_seq_len/tgt_seq_len, d_head)
    V: value sequences, shape: (bs, num_heads, src_seq_len/tgt_seq_len, d_head)
    mask: attention mask
    Return: the scaled dot product attention.
    """
    # Compute alignment of Q and K
    # Three cases of alignment shape:
    #   self-attention in encoder: (bs, num_heads, src_seq_len, src_seq_len)
    #   masked self-attention in decoder: 
    #     (bs, num_heads, tgt_seq_len, tgt_seq_len)
    #   cross-attention in decoder: (bs, num_heads, tgt_seq_len, src_seq_len)
    alignment = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
    
    # Apply mask
    if mask is not None:
      alignment = alignment.masked_fill(mask==0, -1e9)
    
    # Compute attention score
    # attn_scores shape: same as alignment's shape
    attn_scores = torch.softmax(alignment, dim=-1)

    # Output
    # output shape: (bs, num_heads, src_seq_len/tgt_seq_len, d_head)
    output = torch.matmul(attn_scores, V)
    
    return output
  
  def reshape_heads_back(self, x):
    batch_size, _, seq_len, _ = x.size()
    return x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

  def forward(self, Q, K, V, mask=None):
    """
    Q: sequences for query
    K: sequences for key
    V: sequences for value
      Three cases for shapes of Q, K, V
        self-attention in encoder: (bs, src_seq_len, d_model) for Q, K, and V
        masked self-attention in decoder: (bs, tgt_seq_len, d_model) 
          for Q, K, and V
        cross-attention in decoder: (bs, tgt_seq_len, d_model) for Q,
          (bs, src_seq_len, d_model) for K and V
    mask: attention mask
      two cases for shape of mask
        self-attention in encoder and cross-attention in decoder: 
          src_mask shape: (bs, 1, 1, src_seq_len)
        masked self-attention in decoder: 
          tgt_mask shape: (bs, 1, tgt_seq_len, tgt_seq_len)
        
    Return: output of multi-head attention
    """
    # Apply weights and reshape heads from 
    # (bs, src_seq_len/tgt_seq_len, d_model) to 
    # (bs, num_heads, src_seq_len/tgt_seq_len, d_head)
    Q = self.reshape_heads(self.W_q(Q))
    K = self.reshape_heads(self.W_k(K))
    V = self.reshape_heads(self.W_v(V))
    
    # Compute attention score using scaled dot production attention
    # attn_output shape: (bs, num_heads, src_seq_len/tgt_seq_len, d_head)
    attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

    # Compute output
    # output shape: (bs, src_seq_len/tgt_seq_len, d_model)
    output = self.W_o(self.reshape_heads_back(attn_output))

    return output
  
class PositionWiseFeedforward(torch.nn.Module):
  def __init__(self, d_model, d_ff):
    """
    d_model: dimension of model embedding
    d_ff: dimension of feedforward embedding
    """
    super(PositionWiseFeedforward, self).__init__()
    self.fc1 = torch.nn.Linear(d_model, d_ff)
    self.fc2 = torch.nn.Linear(d_ff, d_model)
    self.relu = torch.nn.ReLU()

  def forward(self, x):
    """
    x: input
    """
    return self.fc2(self.relu(self.fc1(x)))
  
class EncoderLayer(torch.nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout):
    """
    d_model: dimension of model embedding
    num_heads: number of attention heads
    d_ff: dimension of feedforward embedding
    dropout: dropout rate
    """
    super(EncoderLayer, self).__init__()
    self.self_attn = MultiHeadAttention(d_model, num_heads)
    self.feedforward = PositionWiseFeedforward(d_model, d_ff)
    self.norm1 = torch.nn.LayerNorm(d_model)
    self.norm2 = torch.nn.LayerNorm(d_model)
    self.dropout = torch.nn.Dropout(dropout)
  
  def forward(self, x, mask):
    """
    x: input sequences, shape: (bs, src_seq_len, d_model)
    mask: attention mask (src_mask), all elements are 1, 
      shape: (bs, 1, 1, src_seq_len)
    """
    # Q, K, V for self_attn in encoder are all input sequences
    # attn_output shape: (bs, src_seq_len, d_model)
    attn_output = self.self_attn(x, x, x, mask)

    # dropout is applied to the output of self-attention to avoid over-fitting
    # residual, and layer normalization
    # x shape: (bs, src_seq_len, d_model)
    x = self.norm1(x + self.dropout(attn_output))

    # ff_output shape: (bs, src_seq_len, d_model)
    ff_output = self.feedforward(x)

    # dropout is applied to the output of feedforward to avoid over-fitting
    # residual, and layer normalization
    # x shape: (bs, src_seq_len, d_model)
    x = self.norm2(x + self.dropout(ff_output))
    return x
  
class DecoderLayer(torch.nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout):
    """
    d_model: dimension of model embedding
    num_heads: number of attention heads
    d_ff: dimension of feedforward embedding
    dropout: dropout rate
    """
    super(DecoderLayer, self).__init__()
    self.self_attn = MultiHeadAttention(d_model, num_heads)
    self.cross_attn = MultiHeadAttention(d_model, num_heads)
    self.feedforward = PositionWiseFeedforward(d_model, d_ff)
    self.norm1 = torch.nn.LayerNorm(d_model)
    self.norm2 = torch.nn.LayerNorm(d_model)
    self.norm3 = torch.nn.LayerNorm(d_model)
    self.dropout = torch.nn.Dropout(dropout)

  def forward(self, x, enc_output, src_mask, tgt_mask):
    """
    x: output sequences, shape: (bs, tgt_seq_len, d_model)
    enc_output: output of last encoder layer, shape: (bs, src_seq_len, d_model)
    src_mask: attention mask is passed to cross-attention
      src_mask shape: (bs, 1, 1, src_seq_len) 
    tgt_mask: attention mask is passed to masked self-attention
      tgt_mask shape: (bs, 1, tgt_seq_len, tgt_seq_len)
    """
    # Q, K, V for self_attn in decoder are all output sequences
    # attn_output shape: (bs, tgt_seq_len, d_model)
    attn_output = self.self_attn(x, x, x, tgt_mask)
    
    # Dropout, residual and layer norm
    # x shape: (bs, tgt_seq_len, d_model)
    x = self.norm1(x + self.dropout(attn_output))
    
    # Q is attn_output after layer norm 1, 
    # K and V are are output of last encoder layer (enc_output)
    # attn_output shape: (bs, tgt_seq_len, d_model)
    attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)

    # Dropout, residual and layer norm
    # x shape: (bs, tgt_seq_len, d_model)
    x = self.norm2(x + self.dropout(attn_output))

    # Feedforward
    # ff_output shape: (bs, tgt_seq_len, d_model)
    ff_output = self.feedforward(x)

    # Dropout, residual and layer norm
    # x shape: (bs, tgt_seq_len, d_model)
    x = self.norm3(x + self.dropout(ff_output))
    return x
  
class PositionalEncoding(torch.nn.Module):
  def __init__(self, d_model, max_seq_len):
    """
    d_model: dimension of model embedding
    max_seq_len: maximum length of positional encoding, 
      >= max(src_seq_len, tgt_seq_len)
    """
    super(PositionalEncoding, self).__init__()

    # pe shape: (max_seq_len, d_model)
    pe = torch.zeros(max_seq_len, d_model)
    # position shape: (max_seq_len, 1)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    # freq_term shape: (d_model/2)
    freq_term = torch.exp(torch.arange(0, d_model, 2).float() * \
                -(math.log(10000.0) / d_model))
    # pe[:, 0::2] shape: (max_seq_len, d_model/2)
    pe[:, 0::2] = torch.sin(position * freq_term)
    # pe[:, 1::2] shape: (max_seq_len, d_model/2)
    pe[:, 1::2] = torch.cos(position * freq_term)

    # register_buffer saves pe in state_dict of model 
    # along with other trainable parameters.
    # Now, pe shape becomes (1, max_seq_len, d_model)
    self.register_buffer('pe', pe.unsqueeze(0))
    
  def forward(self, x):
    """
    Add positional encoding to input sequences or output sequences
    x: input sequences or output sequences
    """
    # Get positional encoding up to the length of input/output sequences
    return x + self.pe[:, :x.size(1)]

class Transformer(torch.nn.Module):
  def __init__(self, src_vocab_size, tgt_vocab_size,
               d_model, num_heads, num_layers, d_ff,
               max_seq_len, dropout):
    """
    src_vocab_size: vocabulary size of input sequences
    tgt_vocab_size: vocabulary size of output sequences
    d_model: dimension of model embedding
    num_heads: number of attention heads
    num_layers: number of encoder/decoder layers
    d_ff: dimension of feedforward embedding
    max_seq_len: maximum length of generated positional encoding
    dropout: dropout rate
    """
    super(Transformer, self).__init__()
    # Input and output embeddings
    self.encoder_embedding = torch.nn.Embedding(src_vocab_size, d_model)
    self.decoder_embedding = torch.nn.Embedding(tgt_vocab_size, d_model)
    
    # Positional encoding
    self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
    
    # Encoder layers
    self.encoder_layers = torch.nn.ModuleList([
        EncoderLayer(d_model, num_heads, d_ff, dropout) \
          for _ in range(num_layers)
      ]
    )

    # Decoder layers
    self.decoder_layers = torch.nn.ModuleList([
        DecoderLayer(d_model, num_heads, d_ff, dropout) \
          for _ in range(num_layers)
      ]
    )
    
    # Linear layer for output
    self.fc = torch.nn.Linear(d_model, tgt_vocab_size)

    # Dropout
    self.dropout = torch.nn.Dropout(dropout)

  def generate_mask(self, src, tgt):
    """
    src: input sequences, shape: (bs, src_seq_len)
    tgt: output sequences, shape: (bs, tgt_seq_len)
    Returns: 
      mask for self-attention in encoder and 
        cross-attention in decoder (all 1s)
      mask for masked self-attention in decoder
    """
    # src_mask shape: (bs, 1, 1, src_seq_len), all elements are 1
    src_mask = torch.ones_like(src).unsqueeze(1).unsqueeze(2)
    # tgt_mask shape: (bs, 1, tgt_seq_len, 1)
    tgt_mask = torch.ones_like(tgt).unsqueeze(1).unsqueeze(3)
    tgt_seq_len = tgt.size(1)
    # nopeak_mask shape: (1, tgt_seq_len, tgt_seq_len)
    nopeak_mask = (1 - torch.triu(torch.ones(1, tgt_seq_len, tgt_seq_len),
                                  diagonal=1)).bool()
    nopeak_mask = nopeak_mask.to(src.device)
    # tgt_mask shape becomes (bs, 1, tgt_seq_len, tgt_seq_len)
    # due to broadcasting
    tgt_mask = tgt_mask & nopeak_mask
    return src_mask, tgt_mask
  
  def forward(self, src, tgt):
    """
    src: input sequences, shape: (bs, src_seq_len)
    tgt: output sequences, shape: (bs, tgt_seq_len)
    """
    # Generate masks for encoder and decoder
    # src_mask shape: (bs, 1, 1, src_seq_len)
    # tgt_mask shape: (bs, 1, tgt_seq_len, tgt_seq_len)
    src_mask, tgt_mask = self.generate_mask(src, tgt)
    
    # Embeddings of input sequences
    # src_emb shape: (bs, src_seq_len, d_model)
    src_emb = self.dropout(
      self.positional_encoding(self.encoder_embedding(src))
    )

    # Embeddings of output sequences
    # tgt_emb shape: (bs, tgt_seq_len, d_model)
    tgt_emb = self.dropout(
      self.positional_encoding(self.decoder_embedding(tgt))
    )

    # Transformer encoder
    # enc_output shape: (bs, src_seq_len, d_model)
    enc_output = src_emb
    for enc_layer in self.encoder_layers:
      enc_output = enc_layer(enc_output, src_mask)

    # Transformer decoder
    # dec_output shape: (bs, tgt_seq_len, d_model)
    dec_output = tgt_emb
    for dec_layer in self.decoder_layers:
      # both src_mask and tgt_mask are passed to decoder layer
      # src_mask is passed to self-attention in decoder layer
      # tgt_mask is passed to cross-attention in decoder layer
      dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

    # Linear layer
    #output shape: (bs, tgt_seq_len, tgt_vocab_size)
    output = self.fc(dec_output)

    return output
  
# Data and model configurations
src_vocab_size = 4000
tgt_vocab_size = 5000
max_seq_len = 100
src_seq_len = 50
tgt_seq_len = 60
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
dropout = 0.1
batch_size = 64
lr = 1e-4
beta1 = 0.9
beta2 = 0.98
eps = 1e-9
epochs = 10

# Create transformer model
transformer = Transformer(src_vocab_size, tgt_vocab_size, 
                          d_model, num_heads, num_layers, 
                          d_ff, max_seq_len, dropout)

# Dummy data used in each training iteration (to be replaced with real data)
# src_data shape: (batch_size, src_seq_length)
src_data = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
# tgt_data shape: (batch_size, tgt_seq_length)
tgt_data = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))

# Loss
criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(transformer.parameters(),
                             lr=lr,
                             betas=(beta1, beta2),
                             eps=eps)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transformer.to(device)
src_data = src_data.to(device)
tgt_data = tgt_data.to(device)

# Training loop
for epoch in range(epochs):
  optimizer.zero_grad()
  output = transformer(src_data, tgt_data[:, :-1])
  loss = criterion(output.contiguous().view(-1, tgt_vocab_size),
                   tgt_data[:, 1:].contiguous().view(-1))
  loss.backward()
  optimizer.step()
  print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
  
