from typing import Optional, Tuple
import torch
from torch.nn inport nn

 class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size = 768,
        intermediate_size = 3072,
        num_hidden_layer = 12,
        num_attention_head = 12,
        num_channels = 3,
        image_size = 224,
        patch_size = 16,
        layer_norm_eps = 1e - 6,
        attention_dropout = 0.0,
        num_image_token : int = None,
        **kwargs
    ):

    super().__init__()

    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.num_hidden_layer = num_hidden_layer
    self.num_attention_head = num_attention_head
    self.num_channels = num_channels
    self.image_size = image_size
    self.patch_size = patch_size
    self.layer_norm_eps = layer_norm_eps
    self.attention_dropout = attention_dropout
    self.num_image_token = num_image_token

    class SiglipVisionEmbedding(nn.Module):
        def __init__(self, config : SiglipVisionConfig):
            super().__init__()
            self.config = config
            self.embed_dim = config.hidden_size
            self.image_size = config.image_size
            self.patch_size = config.patch_size

            self.patch_embedding = nn.conv2d(
                in_channels = config.num_channels,
                out_channels = self.embed_dim,
                kernal_size = self.patch_size,
                stride = self.patch_size,
                padding = 'valid', #Indicates not padding is added
            )

            self.num_patches = (self.image_size // self.patch_size) ** 2
            self.num_position = self.num_patches
            self.position_embedding = nn.Embedding(self.num_position, self.embed_dim)
            self.register_buffer{
                "position ids",
                torch.arrange(self.num_position).expand((1, -1)),
                persistant = False,
            }

            def forward(self, pixel_values : torch.FloatTensor) -> torch.Tensor:
                _, _, height, width = pixel_values.shape# [Batch_Size, Channels, Height, Width]
                # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
                # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
                # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
                patch_embedding = self.patch_embedding(pixel_values)
                # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
                # where Num_Patches = Num_Patches_H * Num_Patches_W
                embeddings = patch.embeds.flatten(2)
                # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
                embeddings = embeddings.transpose(1,2)
                # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
                embeddings = embeddings + self.position_embedding(self.position_ids)
                # [Batch_Size, Num_Patches, Embed_Dim]
                return embeddings

    class SiglipAttention(nn.Module):
        """Multi-headed attention from 'Attention Is All You Need' paper"""

        def __init__(self, config : SiglipVisionConfig):
            self.config = config
            self.embed_dim = config.hidden_size
            self.num_heads = config.num_attention_head
            self.head_dim = self.embed_dim // self.num_heads
            self.scale = self.head_dim ** -0.5 #Equivalent to 1 / sqrt(self.head_dim) 
            self.dropout = self.attention_dropout

            self.k_proj = nn.linear(self.embed_dim, self.embed_dim)
            self.v_proj = nn.linear(self.embed_dim, self.embed_dim)
            self.q_proj = nn.linear(self.embed_dim, self.embed_dim)
            self.out_proj = nn.linear(self.embed_dim, self.embed_dim)

        def forward(
            self.
            hidden_states : torch.Tensor
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
        Batch_Size, seq_len, _ = hidden_states.size()
        # query_states: [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        # key_states: [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        # value_states: [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)
        # query_states: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        query_states = query_states.view(Batch_Size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = query_states.view(Batch_Size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        value_states = query_states.view(Batch_Size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Calculate the attention using the formula Q * K^T / sqrt(d_k). attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)

        if attn_weights.size() != (Batch_Size, num_heads, seq_len, seq_len):
            raise ValueErro(
                f"Attention weights should be of size {{Batch_Size, num_heads, seq_len, seq_len}}, but is "
                f" {attn_weight.size()}"
            )

            # Apply the softmax row-wise. attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
            attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float32).to(query_states.dtype)
            # [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            attn_weight = nn.functional.dropout(attn_weight, p = self.dropout, training = self.training)
            # [Batch_Size, Num_Patches, Embed_Dim]
            attn_output = torch.matmul(attn_weight, value_states)

            if attn_output.size() != (Batch_Size, self.num_heads, seq_len, self.head_dim):
                raise ValueError(
                    f"attn_output should be of size {{Batch_Size, self.num_heads, seq_len, self.head_dim}}, but is"
                    f" {attn_output.size()}"
                )
                # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
                attn_output = attn_output.transpose(1, 2).contiguous()
                # [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
                attn_output = attn_output.reshape(Batch_Size, seq_len, self.embed_dim)
                # [Batch_Size, Num_Patches, Embed_Dim]
                attn_output = self.out_proj(attn_output)

                return attn_output, attn_weights

    class SiglipMLP(nn.Module):
        def __init__(self, config : SiglipVisionConfig):
            super().__init__()
            self.config = config
            self.fc1 = nn.linear(config.hidden_size, config.intermediate_size)
            self.fc2 = nn.linear(config.intermediate_size, config.hidden_size)  

        def forward(self, hidden_states : torch.Tensor) -> torch.Tensor:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
            hidden_states = self.fc1(hidden_states)
            # hidden_states: [Batch_Size, Num_Patches, Intermediate_Size]
            hidden_states == nn.functional.gelu(hidden_states, approximate = "tanh ")
            # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = self.fc2(hidden_states)

            return hidden_states

    class SiglipVisionEncoderLayer(nn.Module):
        def __init__(self, config : SiglipVisionConfig):
            self.embed_dim = config.hidden_size
            self.self_attn = SiglipAttention(config)
            self.layer_norm1 == nn.LayerNorm(self.embed_dim, eps = layer_norm_eps)
            self.mlp = SiglipMLP(config)
            self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps = layer_norm_eps)

        def forward(
            self,
            hidden_states : torch.Tensor
        ) -> torch.Tensor:
        # residual: [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states = hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        # residual: [Batch_Size, Num_Patches, Embed_Dim] 
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states

         return hidden_states

    class SiglipEncoder(nn.Module):
        def __init__(self,config : SiglipVisionConfig):
            super().__init__()
            self.config = config
            self.layers == nn.ModuleList(
                {SiglipEncoderLayer(config) for _ in range(config : num_hidden_layer)}
            )

        def forward(
            self,
            inputs_embeds : torch.Tensor
        ) -> torch.Tensor:
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states

    class SiglipVisionTransformer(nn.Module):
        def __init__(self, config : SiglipVisionConfig):
            super().__init__()
            self.config = config
            embed_dim = config.hidden_size

            self.embeddings = SiglipVisionEmbedding(config)
            self.encoder = SiglipVisionEncoder(config)
            self.post_layernorm = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps)
        
        def forward(self, pixel_values : torch.Tensor) -> torch.Tensor:
            #pixel_values : [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = self.embeddings(pixel_value)

            last_hidden_state = self.encoder(inputs_embeds = hidden_states)

            last_hidden_state = self.post_layernorm(last_hidden_state)

            return last_hidden_state

    class SiglipVisionModel(nn.Module):

        def __init__(self, config : SiglipVisionConfig):
            super().__init__()
            self.config = config
            self.vision_model = SiglipVisionTransformer(config)

        def forward(self, pixel_values) -> tuple:
            #[Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
            return self.vision_model(pixel_value = pixel_values)