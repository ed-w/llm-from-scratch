from typing import Optional
import torch
import torch.nn as nn

from cs336_basics.utils.nn import GELU, softmax


class RMSNorm(nn.Module):
    """Applies Root mean Square Layer Normalisation over a mini-batch of inputs.

    Args:
        d_model: int
            The number of expected features in the input (hidden dimension).
        eps: float, default is 1e-5
            A value added to the denominator for numerical stability.
    
    Shape:
        - Input: (batch_size, sequence_length, hidden_dimension)
        - Output: same shape as input
    
    """
    def __init__(self,
        d_model: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, a: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.sum(a ** 2, dim=-1, keepdim=True) / self.d_model + self.eps)
        return a / rms * self.weight.view(1, 1, -1)


class PositionWiseFeedForward(nn.Module):
    """This class implements the position-wise feed-forward network used in the Transformer.
    It consists of two linear transformations with no bias and with a GELU activation in between.

    Args:
        d_model: int
            Dimensionality of the feedforward input and output.
        d_ff: int
            Dimensionality of the feedforward network's inner layer.
    """
    def __init__(self,
        d_model: int,
        d_ff: int,
    ):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(GELU(self.w1(x)))


def scaled_dot_product_attention(
    K: torch.FloatTensor,
    Q: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None,
    pdrop: Optional[float] = None,
) -> torch.FloatTensor:
    """Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        K: torch.FloatTensor
            Tensor with attention keys. Shape is
            (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        Q: torch.FloatTensor
            Tensor with attention queries. Shape is
            (batch_size, ..., seq_len, key_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        V: torch.FloatTensor
            Tensor with attention values. Shape is
            (batch_size, ..., seq_len, value_dimension), where
            "..." is optional and represents any number of other
            batch dimensions (e.g., num_heads).
        mask: Optional[torch.BoolTensor]
            An (optional) mask of shape (seq_len, seq_len).
            Attention scores for positions with a mask value of `True` should
            be masked out, i.e., not affect the softmaxed attention probabilities.
        pdrop: Optional[float], default is None.
            If given, drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.

    Returns:
        torch.FloatTensor of shape (batch_size, ..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """
    x = Q @ K.transpose(-2, -1) * (K.shape[-1] ** -0.5)
    if mask is not None:
        x.masked_fill_(mask, -torch.inf)
    x = softmax(x, dim=-1)
    if pdrop and pdrop > 0:
        x = nn.functional.dropout(x, p=pdrop)
    return x @ V


class CausalMultiHeadSelfAttention(nn.Module):
    """This class implements the multi-head self-attention mechanism with causal masking.

    Args:
        d_model: int
            Dimensionality of the feedforward input and output.
        num_heads: int
            Number of heads to use in multi-headed attention.
        attn_pdrop: float
            Drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
    """
    def __init__(self,
        d_model: int,
        num_heads: int,
        attn_pdrop: float | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        
        assert d_model % num_heads == 0
        self.d_keys = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.output_proj = nn.Linear(d_model, d_model, bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape

        # compute Q, K and V projections
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # split in to multiple heads
        queries = queries.view(*shape[:-1], self.num_heads, self.d_keys).transpose(1, -2)
        keys = keys.view(*shape[:-1], self.num_heads, self.d_keys).transpose(1, -2)
        values = values.view(*shape[:-1], self.num_heads, self.d_keys).transpose(1, -2)

        # causal mask
        mask = torch.triu(torch.ones((shape[1], shape[1]), dtype=torch.bool, device=x.device), diagonal=1)
        
        # compute scaled dot-product attention
        attn = scaled_dot_product_attention(
            K=keys,
            Q=queries,
            V=values,
            pdrop=self.attn_pdrop,
            mask=mask
        )
    
        # combine heads
        attn = attn.transpose(1, -2).reshape(shape)
        out = self.output_proj(attn)
        return out


class TransformerBlock(nn.Module):
    """This class implements a single Transformer block.

    Args:
        d_model: int
            The dimensionality of the Transformer block input.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        attn_pdrop: float
            Drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        residual_pdrop: float
            Apply dropout to the output of each sub-layer, before it
            is added to the sub-layer input and normalized (section 5.4).
    """
    def __init__(self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float | None = None,
        residual_pdrop: float | None = None,
    ):
        super().__init__()

        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, attn_pdrop)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(residual_pdrop)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class TransformerLanguageModel(nn.Module):
    """This class implements a Transformer language model.

    Args:
        vocab_size: int
            The number of unique items in the output vocabulary to be predicted.
        context_length: int,
            The maximum number of tokens to process at once.
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_layers: int
            The number of Transformer layers to use.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        attn_pdrop: float
            Drop-out the attention probabilities (the softmax-normalized
            attention scores) with this rate.
        residual_pdrop: float
            Apply dropout to the sum of the token and position embeddings
            as well as the output of each sub-layer, before it is added to the
            sub-layer input and normalized (section 5.4).
    """
    def __init__(self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float | None = None,
        residual_pdrop: float | None = None,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.dropout = nn.Dropout(residual_pdrop)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Given an input tensor x, return the output of the Transformer language model.
        First compute the token and absolute position embeddings, then add toegether and apply dropout.
        Next apply the transformer layers. Finally, apply the final layer norm and linear layer.
        Args:
            in_indices: torch.LongTensor
                Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
                `sequence_length` is at most `context_length`.

        Returns:
            FloatTensor of shape (batch size, sequence_length, vocab_size) with the predicted unnormalized
            next-word distribution for each token.
        """
        x = self.dropout(
            self.token_embeddings(x) + \
            self.position_embeddings(
                torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            )
        )
        
        for layer in self.layers:
            x = layer(x)

        x = self.lm_head(self.ln_final(x))
        return x
