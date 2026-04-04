import torch.nn as nn
import torch

class OTLIP(nn.Module):
    def __init__(self, vision_model, text_model, shared_dim=512, device=None, temp=0.07):
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.vision_proj = nn.Linear(2048, shared_dim)
        self.text_proj = nn.Linear(768, shared_dim)
        self.temp = temp
    
    def encode_images(self, images):
        '''Encode a batch of preprocessed image tensors into a batch of normalized embedding vectors that live in a shared space with images.

            Args:
                vision_model (AutoModel): The vision model that converts batch image tensors into the embedding vectros 
                images (torch.tensor): A batch of image tensors of the dim (B, 3, 224, 224)
                device (torch.device): context-manager that changes the selected device.
                mode (str)

            Returns:
                image_tensor (torch. tensor): A batch tensor of dimensions (B, d)
        '''

        # Encode batch of images using the vision model 
        vision_outputs = self.vision_model(images)

        # Pooling layer to convert into vector to pass to the projection layer  (global average pooling)
        image_features = vision_outputs.pooler_output.flatten(1)

        # Use a linear projection layer to bring the set of vectors into a common dimension space 
        p_imgs = self.vision_proj(image_features)

        return p_imgs

    def encode_texts(self, text_batch):
        '''Turns a batch text of input ids and attention masks into a batch of normalized embedding vectors that live in a shared space with images.
        
            Args:
                text_model: The text model that converts a batch of text input ids and respective attention_masks into a embedding vector
                input_ids (torch.tensor): A vector of input ids of the dim (B, 77)
                attention_mask (torch.tensor) A vector of attention mask of the dim (B,77)
                
            Returns:
                text_vector (torch.tensor): A batch tensor of dimensions (B, d)    
            '''
        
        # Enocode input ids and attention masks into a vector of dimension (B, 77, H)
        text_outputs = self.text_model(**text_batch)

        # Mean pooling output to be fed into the text projection layer   -> Basically average of non padded tokens          
        token_embeddings = text_outputs.last_hidden_state
        attention_mask_expanded = text_batch['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(attention_mask_expanded * token_embeddings, 1)
        sum_mask = torch.clip(attention_mask_expanded.sum(1), min=1e-9)
        text_features = sum_embeddings / sum_mask
        
        # Pass the encoded text vector through a linear projection layer to common dimensions
        p_texts = self.text_proj(text_features)

        return p_texts
    
    def forward(self, images, texts):
        image_features = self.encode_images(images)
        text_features = self.encode_texts(texts)

        # Normalized image and text features 
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Cosine Similarity
        scale_factor = 1/self.temp
        logits = (text_features @ image_features.T) * scale_factor

        return logits, image_features, text_features







