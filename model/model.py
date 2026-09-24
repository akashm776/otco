import torch.nn as nn
import torch


class OTLIP(nn.Module):
    def __init__(
        self, vision_model, text_model, shared_dim=512, device=None, temp=0.07
    ):
        # Register this class as a PyTorch module so that its parameters are
        # discovered by model.parameters(), moved by model.to(...), and saved
        # in its state_dict.
        super().__init__()

        # Keep the pretrained encoders as child modules. The training setup can
        # then freeze most of their parameters and selectively unfreeze layers.
        self.vision_model = vision_model
        self.text_model = text_model

        # ResNet-50 produces a 2048-dimensional pooled representation, while
        # DistilBERT produces 768-dimensional token representations. These two
        # learned projections map both modalities into the same shared space,
        # where dot products between images and captions become meaningful.
        self.vision_proj = nn.Linear(2048, shared_dim)
        self.text_proj = nn.Linear(768, shared_dim)

        # Temperature controls the sharpness of the similarity logits. Dividing
        # by a small value such as 0.07 magnifies differences in cosine
        # similarity.
        self.temp = temp

    def encode_images(self, images):
        """Encode a batch of preprocessed image tensors into a batch of
        normalized embedding vectors that live in a shared space with
        images.

         Args:
             images (torch.tensor): A batch of image tensors of the
                                    dim (B, 3, 224, 224)
         Returns:
             image_tensor (torch. tensor): A batch tensor of dimensions (B, d)
        """

        # Convert pixels [B, 3, 224, 224] into the vision encoder's features.
        vision_outputs = self.vision_model(images)

        # ResNet's pooler_output is [B, 2048, 1, 1]. Flatten only the
        # non-batch dimensions so each image becomes one [2048] vector.
        image_features = vision_outputs.pooler_output.flatten(1)

        # Map [B, 2048] to [B, shared_dim]. This output is not normalized here
        # because forward() normalizes both modalities together below.
        p_imgs = self.vision_proj(image_features)

        return p_imgs

    def encode_texts(self, text_batch):
        """Encode tokenized captions as vectors in the shared embedding space.

        Args:
            text_batch: Mapping containing ``input_ids`` and ``attention_mask``
                tensors, each with shape ``[B, sequence_length]``.

        Returns:
            A tensor with shape ``[B, shared_dim]``.
        """

        # DistilBERT returns one contextual representation per token:
        # [B, sequence_length] token IDs -> [B, sequence_length, 768].
        text_outputs = self.text_model(**text_batch)

        # DistilBERT has no image-like global pooling output, so create one by
        # averaging its token embeddings. The attention mask prevents padding
        # tokens from contributing to the caption representation.
        token_embeddings = text_outputs.last_hidden_state

        # Expand [B, sequence_length] to [B, sequence_length, 768], allowing
        # the same 0/1 token mask to be applied to every embedding coordinate.
        attention_mask_expanded = (
            text_batch["attention_mask"]
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        # Sum only real-token embeddings and count the real tokens. Clipping the
        # count avoids division by zero for a malformed, fully padded caption.
        sum_embeddings = torch.sum(
            attention_mask_expanded * token_embeddings, 1
        )
        sum_mask = torch.clip(attention_mask_expanded.sum(1), min=1e-9)

        # Produce one mean-pooled [768] representation for every caption.
        text_features = sum_embeddings / sum_mask

        # Map [B, 768] into the same [B, shared_dim] space as the images.
        p_texts = self.text_proj(text_features)

        return p_texts

    def forward(self, images, texts):
        # Encode the two modalities independently before comparing them.
        image_features = self.encode_images(images)
        text_features = self.encode_texts(texts)

        # L2 normalization gives every embedding unit length. Their dot product
        # is consequently cosine similarity rather than an unbounded magnitude-
        # dependent score.
        image_features = image_features / image_features.norm(
            dim=1, keepdim=True
        )
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Compare every caption with every image. With a batch size B, this
        # produces a [B, B] matrix. The diagonal contains matching pairs because
        # the Dataset/DataLoader preserve the image-caption pair ordering; all
        # off-diagonal entries are treated as candidate negatives.
        scale_factor = 1 / self.temp
        logits = (text_features @ image_features.T) * scale_factor

        # Loss functions need both the scaled pairwise logits and the normalized
        # embeddings (the OT-Mix loss constructs synthetic image embeddings).
        return logits, image_features, text_features
