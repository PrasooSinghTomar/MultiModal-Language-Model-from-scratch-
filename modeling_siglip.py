from typing import Optional, tuple
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
                _, _, height, width = pixel_value.shape# [Batch_Size, Channels, Height, Width]
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