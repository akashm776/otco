import torch
import torch.nn.functional as F
import json
import glob
from pathlib import Path
from datetime import datetime
import os
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer
import random
    

def save_checkpoint(model, optimizer, step, loss, filepath):
    """Save training checkpoint"""
    raw_model = model.module if hasattr(model, 'module') else model
    checkpoint = {
        'step': step,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at step {step}")


def load_model_checkpoint(model, filepath, device):
    """Load model weights only for inference."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    step = checkpoint.get('step', None)
    loss = checkpoint.get('loss', None)

    if step is not None and loss is not None:
        print(f"Loaded model checkpoint from step {step}, loss: {loss:.4f}")
    elif step is not None:
        print(f"Loaded model checkpoint from step {step}")
    else:
        print("Loaded model checkpoint")

    return step, loss


def load_checkpoint(model, optimizer, filepath, device):
    """Load training checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from step {step}, loss: {loss:.4f}")
    return step, loss


def calculate_layer_grad_norm(layer):
    total_norm = 0.0
    for p in layer.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def compute_retrieval_metrics(logits):
    """
    Compute comprehensive retrieval metrics including:
    - Positive vs negative margins (top-1, top-5, median)
    - Rank-based retrieval metrics
    """
    batch_size = logits.size(0)
    
    # Positive similarities (diagonal)
    pos_sims = logits.diag()
    
    # Mask diagonal to get negatives only
    mask = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
    masked_logits = logits.masked_fill(mask, float('-inf'))
    
    # 1) Top-k negative margins
    top1_neg = masked_logits.max(dim=1)[0]
    
    k = min(5, batch_size - 1)
    topk_negs = torch.topk(masked_logits, k=k, dim=1)[0]
    top5_neg_mean = topk_negs.mean(dim=1)
    
    # Median negative
    neg_logits = logits.clone()
    neg_logits[mask] = float('nan')
    all_negs = neg_logits[~mask].reshape(batch_size, -1)
    median_neg = all_negs.median(dim=1)[0]
    
    # 2) Rank diagnostics
    pos_expanded = pos_sims.unsqueeze(1)
    num_greater = (masked_logits >= pos_expanded).sum(dim=1).float()
    ranks = num_greater + 1
    
    recall_at_1 = (ranks == 1).float().mean().item() * 100
    mean_rank = ranks.mean().item()
    
    return {
        'mean_pos': pos_sims.mean().item(),
        'pos_std': pos_sims.std().item(),
        'margin_top1': (pos_sims - top1_neg).mean().item(),
        'margin_top5': (pos_sims - top5_neg_mean).mean().item(),
        'margin_median': (pos_sims - median_neg).mean().item(),
        'mean_top1_neg': top1_neg.mean().item(),
        'mean_top5_neg': top5_neg_mean.mean().item(),
        'mean_median_neg': median_neg.mean().item(),
        'recall_at_1': recall_at_1,
        'mean_rank': mean_rank,
    }


def compute_embedding_stats(image_features, text_features):
    """
    Compute embedding space health metrics.
    """
    return {
        'img_norm_mean': image_features.norm(dim=1).mean().item(),
        'img_norm_min': image_features.norm(dim=1).min().item(),
        'img_norm_max': image_features.norm(dim=1).max().item(),
        'txt_norm_mean': text_features.norm(dim=1).mean().item(),
        'txt_norm_min': text_features.norm(dim=1).min().item(),
        'txt_norm_max': text_features.norm(dim=1).max().item(),
        'img_variance': image_features.std(dim=0).mean().item(),
        'txt_variance': text_features.std(dim=0).mean().item(),
    }


def compute_loss_components(criterion, logits):
    """
    Compute loss separately for positive and negative pairs.
    """
    batch_size = logits.size(0)
    
    logits_biased = logits + criterion.logit_bias
    labels = 2 * torch.eye(batch_size, device=logits.device) - 1
    
    per_element_loss = -F.logsigmoid(labels * logits_biased)
    
    mask = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
    
    pos_losses = per_element_loss[mask]
    neg_losses = per_element_loss[~mask]
    
    return {
        'pos_loss_mean': pos_losses.mean().item(),
        'neg_loss_mean': neg_losses.mean().item(),
        'pos_loss_sum': pos_losses.sum().item(),
        'neg_loss_sum': neg_losses.sum().item(),
    }


def evaluate_retrieval(model, dataloader, device):
    """
    Compute text→image retrieval metrics over entire validation set.
    Uses ALL captions (e.g., 4050 for Flickr8K validation).
    
    Returns:
        dict with 'R@1', 'R@5', 'R@10', 'mean_rank', 'median_rank'
    """
    model.eval()
    
    # Collect ALL caption embeddings
    all_text_features = []
    all_image_ids = []
    
    # Collect UNIQUE image embeddings
    encoded_images = {}
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            image_ids = batch['image_ids'].tolist()
            
            text_batch = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            
            # Encode ALL text (don't deduplicate)
            text_features = model.encode_texts(text_batch)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            all_text_features.append(text_features.cpu())
            all_image_ids.extend(image_ids)
            
            # Encode images (only unique)
            for i, img_id in enumerate(image_ids):
                if img_id not in encoded_images:
                    img_feature = model.encode_images(images[i:i+1])
                    img_feature = img_feature / img_feature.norm(dim=1, keepdim=True)
                    encoded_images[img_id] = img_feature.cpu()
    
    # Concatenate
    text_features = torch.cat(all_text_features, dim=0)
    all_image_ids = torch.tensor(all_image_ids)
    
    unique_image_ids = sorted(encoded_images.keys())
    image_features = torch.cat([encoded_images[img_id] for img_id in unique_image_ids], dim=0)
    
    image_id_to_idx = {img_id: idx for idx, img_id in enumerate(unique_image_ids)}
    caption_label_indices = torch.tensor([image_id_to_idx[img_id.item()] for img_id in all_image_ids])
    
    num_captions = len(text_features)
    num_images = len(image_features)
    
    # CRITICAL ASSERTS
    assert text_features.shape[0] == num_captions
    assert image_features.shape[0] == num_images
    assert len(caption_label_indices) == num_captions
    assert caption_label_indices.min() >= 0 and caption_label_indices.max() < num_images
    
    # Compute similarity: [num_captions, num_images]
    similarity = text_features @ image_features.T
    
    # For each caption, compute retrieval metrics
    ranks = []
    for i in range(num_captions):
        true_image_idx = caption_label_indices[i].item()
        sims = similarity[i]
        sorted_indices = torch.argsort(sims, descending=True)
        rank = (sorted_indices == true_image_idx).nonzero(as_tuple=True)[0].item()
        ranks.append(rank + 1)
    
    ranks = torch.tensor(ranks, dtype=torch.float)
    
    metrics = {
        'R@1': (ranks == 1).float().mean().item() * 100,
        'R@5': (ranks <= 5).float().mean().item() * 100,
        'R@10': (ranks <= 10).float().mean().item() * 100,
        'mean_rank': ranks.mean().item(),
        'median_rank': ranks.median().item(),
        'num_captions': num_captions,
        'num_images': num_images,
        'num_unique_labels': len(torch.unique(caption_label_indices))
    }
    
    model.train()
    return metrics


def evaluate_image_to_text_retrieval(model, dataloader, device):
    """
    Compute image→text retrieval (reverse direction).
    For each unique image, find any of its captions (min-rank protocol).
    """
    model.eval()
    
    # Build caption bank and image bank
    encoded_images = {}
    caption_embeddings = []
    caption_to_image_id = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            image_ids = batch['image_ids'].tolist()
            
            text_batch = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            
            # Encode text
            text_features = model.encode_texts(text_batch)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            caption_embeddings.append(text_features.cpu())
            caption_to_image_id.extend(image_ids)
            
            # Encode unique images
            for i, img_id in enumerate(image_ids):
                if img_id not in encoded_images:
                    img_feature = model.encode_images(images[i:i+1])
                    img_feature = img_feature / img_feature.norm(dim=1, keepdim=True)
                    encoded_images[img_id] = img_feature.cpu()
    
    caption_embeddings = torch.cat(caption_embeddings, dim=0)
    caption_to_image_id = torch.tensor(caption_to_image_id)
    
    unique_image_ids = sorted(encoded_images.keys())
    image_embeddings = torch.cat([encoded_images[img_id] for img_id in unique_image_ids], dim=0)
    
    # Similarity: [num_images, num_captions]
    similarity = image_embeddings @ caption_embeddings.T
    
    # For each image, find rank of ANY of its captions (min-rank)
    ranks = []
    for img_idx, img_id in enumerate(unique_image_ids):
        # Find all caption indices for this image
        caption_indices = (caption_to_image_id == img_id).nonzero(as_tuple=True)[0]
        
        sims = similarity[img_idx]
        sorted_caption_indices = torch.argsort(sims, descending=True)
        
        # Find best rank among all captions for this image
        best_rank = float('inf')
        for caption_idx in caption_indices:
            rank = (sorted_caption_indices == caption_idx).nonzero(as_tuple=True)[0].item() + 1
            best_rank = min(best_rank, rank)
        
        ranks.append(best_rank)
    
    ranks = torch.tensor(ranks, dtype=torch.float)
    
    metrics = {
        'R@1': (ranks == 1).float().mean().item() * 100,
        'R@5': (ranks <= 5).float().mean().item() * 100,
        'R@10': (ranks <= 10).float().mean().item() * 100,
        'mean_rank': ranks.mean().item(),
    }
    
    model.train()
    return metrics


def sanity_check_eval(model, dataloader, device):
    """
    Sanity check: verify evaluation is working by checking one random caption
    """
    
    model.eval()
    
    batch = next(iter(dataloader))
    caption_idx = 0
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    input_ids = batch['input_ids'][caption_idx]
    caption_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    
    correct_image_id = batch['image_ids'][caption_idx].item()
    correct_image_path = batch['image_paths'][caption_idx]
    
    print(f"\n{'='*80}")
    print(f"SANITY CHECK: Single Caption Retrieval")
    print(f"{'='*80}")
    print(f"Caption: '{caption_text}'")
    print(f"Correct image ID: {correct_image_id}")
    print(f"Correct image path: {correct_image_path}")
    
    text_batch = {
        "input_ids": batch['input_ids'][caption_idx:caption_idx+1].to(device),
        "attention_mask": batch['attention_mask'][caption_idx:caption_idx+1].to(device)
    }
    
    with torch.no_grad():
        caption_emb = model.encode_texts(text_batch)
        caption_emb = caption_emb / caption_emb.norm(dim=1, keepdim=True)
    
    # Encode all unique images
    all_image_embs = []
    all_image_ids = []
    all_image_paths = []
    
    encoded_ids = set()
    
    with torch.no_grad():
        for val_batch in dataloader:
            for i, img_id in enumerate(val_batch['image_ids'].tolist()):
                if img_id not in encoded_ids:
                    img = val_batch['images'][i:i+1].to(device)
                    img_emb = model.encode_images(img)
                    img_emb = img_emb / img_emb.norm(dim=1, keepdim=True)
                    
                    all_image_embs.append(img_emb.cpu())
                    all_image_ids.append(img_id)
                    all_image_paths.append(val_batch['image_paths'][i])
                    encoded_ids.add(img_id)
    
    all_image_embs = torch.cat(all_image_embs, dim=0)
    
    sims = (caption_emb.cpu() @ all_image_embs.T).squeeze(0)
    top10_indices = torch.argsort(sims, descending=True)[:10]
    
    print(f"\nTop-10 retrieved images:")
    for rank, idx in enumerate(top10_indices, 1):
        img_id = all_image_ids[idx]
        img_path = all_image_paths[idx]
        sim = sims[idx].item()
        is_correct = "✓ CORRECT" if img_id == correct_image_id else ""
        print(f"  {rank}. Image ID: {img_id:4d} | Sim: {sim:.4f} | {img_path.split('/')[-1]} {is_correct}")
    
    correct_image_indices = [i for i, img_id in enumerate(all_image_ids) if img_id == correct_image_id]
    if correct_image_indices:
        correct_idx = correct_image_indices[0]
        sorted_indices = torch.argsort(sims, descending=True)
        actual_rank = (sorted_indices == correct_idx).nonzero(as_tuple=True)[0].item() + 1
        print(f"\nActual rank of correct image: {actual_rank}")
    else:
        print(f"\n⚠️ WARNING: Correct image ID {correct_image_id} not found in validation set!")
    
    correct_in_top10 = any(all_image_ids[idx] == correct_image_id for idx in top10_indices)
    print(f"Correct image in top-10: {correct_in_top10}")
    print(f"{'='*80}\n")
    
    model.train()
    return correct_in_top10


def detailed_retrieval_analysis(model, dataloader, device, num_samples=100):
    """
    Run detailed analysis on N random captions to understand failure modes.
    """
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Collect all validation data first
    all_batches = []
    for batch in dataloader:
        all_batches.append(batch)
    
    total_captions = sum(len(b['image_ids']) for b in all_batches)
    sample_indices = random.sample(range(total_captions), min(num_samples, total_captions))
    
    # Encode all unique images once
    print("Encoding all unique images...")
    encoded_images = {}
    for batch in all_batches:
        for i, img_id in enumerate(batch['image_ids'].tolist()):
            if img_id not in encoded_images:
                with torch.no_grad():
                    img = batch['images'][i:i+1].to(device)
                    img_emb = model.encode_images(img)
                    img_emb = img_emb / img_emb.norm(dim=1, keepdim=True)
                    encoded_images[img_id] = img_emb.cpu()
    
    unique_image_ids = sorted(encoded_images.keys())
    image_embeddings = torch.cat([encoded_images[img_id] for img_id in unique_image_ids], dim=0)
    image_id_to_idx = {img_id: idx for idx, img_id in enumerate(unique_image_ids)}
    
    # Analyze sampled captions
    results = {
        'ranks': [],
        'caption_lengths': [],
        'top1_hits': 0,
        'top5_hits': 0,
        'top10_hits': 0,
        'failures': []
    }
    
    caption_counter = 0
    for batch in all_batches:
        for i in range(len(batch['image_ids'])):
            if caption_counter not in sample_indices:
                caption_counter += 1
                continue
            
            input_ids = batch['input_ids'][i:i+1].to(device)
            attention_mask = batch['attention_mask'][i:i+1].to(device)
            caption_text = tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True)
            correct_image_id = batch['image_ids'][i].item()
            
            with torch.no_grad():
                text_batch = {"input_ids": input_ids, "attention_mask": attention_mask}
                caption_emb = model.encode_texts(text_batch)
                caption_emb = caption_emb / caption_emb.norm(dim=1, keepdim=True)
            
            sims = (caption_emb.cpu() @ image_embeddings.T).squeeze(0)
            
            correct_idx = image_id_to_idx[correct_image_id]
            sorted_indices = torch.argsort(sims, descending=True)
            rank = (sorted_indices == correct_idx).nonzero(as_tuple=True)[0].item() + 1
            
            caption_length = (attention_mask[0].sum().item() - 2)
            results['ranks'].append(rank)
            results['caption_lengths'].append(caption_length)
            
            if rank == 1:
                results['top1_hits'] += 1
            if rank <= 5:
                results['top5_hits'] += 1
            if rank <= 10:
                results['top10_hits'] += 1
            
            if rank > 10:
                results['failures'].append({
                    'caption': caption_text,
                    'rank': rank,
                    'length': caption_length,
                    'correct_image_id': correct_image_id
                })
            
            caption_counter += 1
    
    # Print analysis
    ranks = torch.tensor(results['ranks'], dtype=torch.float)
    caption_lengths = torch.tensor(results['caption_lengths'], dtype=torch.float)
    
    print(f"\n{'='*80}")
    print(f"DETAILED RETRIEVAL ANALYSIS (N={num_samples} samples)")
    print(f"{'='*80}")
    
    print(f"\nOverall Metrics:")
    print(f"  Recall@1:  {results['top1_hits']/num_samples*100:.2f}%")
    print(f"  Recall@5:  {results['top5_hits']/num_samples*100:.2f}%")
    print(f"  Recall@10: {results['top10_hits']/num_samples*100:.2f}%")
    print(f"  Mean Rank: {ranks.mean():.2f}")
    print(f"  Median Rank: {ranks.median():.2f}")
    
    print(f"\nCaption Length Analysis:")
    print(f"  Mean caption length: {caption_lengths.mean():.1f} tokens")
    print(f"  Std caption length: {caption_lengths.std():.1f} tokens")
    
    short_mask = caption_lengths <= 10
    medium_mask = (caption_lengths > 10) & (caption_lengths <= 15)
    long_mask = caption_lengths > 15
    
    if short_mask.any():
        short_ranks = ranks[short_mask]
        print(f"\n  Short captions (≤10 tokens, N={short_mask.sum()}):")
        print(f"    Mean rank: {short_ranks.mean():.2f}")
        print(f"    Recall@1: {(short_ranks == 1).float().mean()*100:.2f}%")
    
    if medium_mask.any():
        medium_ranks = ranks[medium_mask]
        print(f"\n  Medium captions (11-15 tokens, N={medium_mask.sum()}):")
        print(f"    Mean rank: {medium_ranks.mean():.2f}")
        print(f"    Recall@1: {(medium_ranks == 1).float().mean()*100:.2f}%")
    
    if long_mask.any():
        long_ranks = ranks[long_mask]
        print(f"\n  Long captions (>15 tokens, N={long_mask.sum()}):")
        print(f"    Mean rank: {long_ranks.mean():.2f}")
        print(f"    Recall@1: {(long_ranks == 1).float().mean()*100:.2f}%")
    
    print(f"\nWorst 5 Failures (rank > 10):")
    worst_failures = sorted(results['failures'], key=lambda x: x['rank'], reverse=True)[:5]
    for i, failure in enumerate(worst_failures, 1):
        print(f"\n  {i}. Rank: {failure['rank']}")
        print(f"     Caption: \"{failure['caption']}\"")
        print(f"     Length: {failure['length']} tokens")
    
    print(f"\n{'='*80}\n")
    
    model.train()
    return results


def diagnose_encoder_quality(model, dataloader, device, num_samples=200):
    """
    Check if vision or text encoder is producing better features.
    """
    model.eval()
    
    print(f"\nCollecting embeddings from {num_samples} samples...")
    
    all_text_embs = []
    all_image_ids = []
    encoded_images = {}
    
    with torch.no_grad():
        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            images = batch['images'].to(device)
            text_batch = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            }
            image_ids = batch['image_ids'].tolist()
            
            txt_emb = model.encode_texts(text_batch)
            txt_emb = txt_emb / txt_emb.norm(dim=1, keepdim=True)
            all_text_embs.append(txt_emb.cpu().numpy())
            all_image_ids.extend(image_ids)
            
            for i, img_id in enumerate(image_ids):
                if img_id not in encoded_images:
                    img_emb = model.encode_images(images[i:i+1])
                    img_emb = img_emb / img_emb.norm(dim=1, keepdim=True)
                    encoded_images[img_id] = img_emb.cpu().numpy()
            
            sample_count += len(image_ids)
    
    text_embs = np.vstack(all_text_embs)
    all_image_ids = np.array(all_image_ids)
    
    unique_image_ids = sorted(encoded_images.keys())
    image_embs = np.vstack([encoded_images[img_id] for img_id in unique_image_ids])
    
    print(f"Collected: {len(text_embs)} captions, {len(image_embs)} unique images")
    
    cross_sim = text_embs @ image_embs.T
    image_id_to_idx = {img_id: idx for idx, img_id in enumerate(unique_image_ids)}
    
    correct_sims = []
    incorrect_sims = []
    
    for i, correct_img_id in enumerate(all_image_ids):
        correct_idx = image_id_to_idx[correct_img_id]
        correct_sim = cross_sim[i, correct_idx]
        correct_sims.append(correct_sim)
    mask = np.ones(len(unique_image_ids), dtype=bool)
    mask[correct_idx] = False
    incorrect_sims.extend(cross_sim[i, mask])

    correct_sims = np.array(correct_sims)
    incorrect_sims = np.array(incorrect_sims)

    # Text encoder clustering
    caption_groups = defaultdict(list)
    for i, img_id in enumerate(all_image_ids):
        caption_groups[img_id].append(i)

    text_within_class = []
    text_between_class = []

    for img_id, caption_indices in caption_groups.items():
        if len(caption_indices) < 2:
            continue
        
        for i in range(len(caption_indices)):
            for j in range(i + 1, len(caption_indices)):
                idx_i = caption_indices[i]
                idx_j = caption_indices[j]
                sim = np.dot(text_embs[idx_i], text_embs[idx_j])
                text_within_class.append(sim)
        
        for idx_i in caption_indices:
            other_indices = [idx for other_img_id, indices in caption_groups.items() 
                        if other_img_id != img_id for idx in indices]
            if len(other_indices) > 0:
                sampled = np.random.choice(other_indices, size=min(10, len(other_indices)), replace=False)
                for idx_j in sampled:
                    sim = np.dot(text_embs[idx_i], text_embs[idx_j])
                    text_between_class.append(sim)

    # Vision encoder clustering
    image_sims = image_embs @ image_embs.T
    np.fill_diagonal(image_sims, -1)

    k = min(5, len(unique_image_ids) - 1)
    topk_img_sims = np.sort(image_sims, axis=1)[:, -k:].flatten()

    triu_indices = np.triu_indices(len(unique_image_ids), k=1)
    all_pairs = image_sims[triu_indices]
    num_random_pairs = min(1000, len(all_pairs))
    random_pairs = np.random.choice(all_pairs, size=num_random_pairs, replace=False)

    # Print diagnostics
    print(f"\n{'='*80}")
    print(f"ENCODER QUALITY DIAGNOSIS")
    print(f"{'='*80}")

    print(f"\n📊 CROSS-MODAL ALIGNMENT (Text ↔ Image)")
    print(f"  Correct matches (mean):    {correct_sims.mean():.4f} ± {correct_sims.std():.4f}")
    print(f"  Incorrect matches (mean):  {incorrect_sims.mean():.4f} ± {incorrect_sims.std():.4f}")
    print(f"  Margin (correct - incorrect): {correct_sims.mean() - incorrect_sims.mean():.4f}")

    if len(text_within_class) > 0 and len(text_between_class) > 0:
        text_within_class = np.array(text_within_class)
        text_between_class = np.array(text_between_class)
        
        print(f"\n📝 TEXT ENCODER (Caption clustering)")
        print(f"  Same-image captions (mean):    {text_within_class.mean():.4f} ± {text_within_class.std():.4f}")
        print(f"  Different-image captions (mean): {text_between_class.mean():.4f} ± {text_between_class.std():.4f}")
        print(f"  Separation: {text_within_class.mean() - text_between_class.mean():.4f}")
        
        text_quality = text_within_class.mean() - text_between_class.mean()
    else:
        print(f"\n📝 TEXT ENCODER")
        print(f"  ⚠️  Not enough multi-caption samples to compute clustering")
        text_quality = 0.0

    print(f"\n🖼️  VISION ENCODER (Image clustering)")
    print(f"  Top-{k} similar images (mean):  {topk_img_sims.mean():.4f} ± {topk_img_sims.std():.4f}")
    print(f"  Random image pairs (mean):  {random_pairs.mean():.4f} ± {random_pairs.std():.4f}")
    print(f"  Separation: {topk_img_sims.mean() - random_pairs.mean():.4f}")

    vision_quality = topk_img_sims.mean() - random_pairs.mean()

    print(f"\n{'='*80}")
    print(f"DIAGNOSIS:")
    print(f"{'='*80}")

    margin = correct_sims.mean() - incorrect_sims.mean()

    if margin < 0.1:
        print(f"❌ CRITICAL: Cross-modal margin is very low ({margin:.4f})")
        print(f"   → Model is barely learning alignment")
    elif margin < 0.3:
        print(f"⚠️  WARNING: Cross-modal margin is weak ({margin:.4f})")
        print(f"   → Model needs more training or capacity")
    else:
        print(f"✓ Cross-modal margin is healthy ({margin:.4f})")

    print(f"\nEncoder Comparison:")
    print(f"  Text clustering quality:   {text_quality:.4f}")
    print(f"  Vision clustering quality: {vision_quality:.4f}")

    if text_quality > 2 * vision_quality and text_quality > 0.05:
        print(f"\n⚠️  TEXT encoder is MUCH stronger than VISION encoder")
        print(f"   → RECOMMENDATION: Unfreeze vision encoder last layer")
    elif vision_quality > 2 * text_quality and vision_quality > 0.05:
        print(f"\n⚠️  VISION encoder is MUCH stronger than TEXT encoder")
        print(f"   → RECOMMENDATION: Unfreeze more text layers")
    elif text_quality < 0.02 and vision_quality < 0.02:
        print(f"\n⚠️  BOTH encoders are producing generic features")
        print(f"   → RECOMMENDATION: Unfreeze both encoders' last layers")
    else:
        print(f"\n✓ Encoders appear balanced")

    print(f"{'='*80}\n")

    model.train()

def log_experiment(config, metrics, save_dir="experiments"):
    """Save experiment config and results"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment = {
        'timestamp': timestamp,
        'config': config,
        'final_metrics': metrics
    }

    filepath = f"{save_dir}/exp_{timestamp}.json"
    with open(filepath, 'w') as f:
        json.dump(experiment, f, indent=2)

    print(f"Experiment logged to: {filepath}")


def load_all_experiments():
    """Load all experiment JSONs from experiments/ directory"""
    exp_files = glob.glob("experiments/exp_*.json")
    experiments = []
    
    for f in exp_files:
        try:
            with open(f, 'r') as fp:
                exp = json.load(fp)
                experiments.append(exp)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    return experiments


# Replace the print_experiment_comparison function in utils.py

def print_experiment_comparison():
    """Print comparison table of all experiments"""
    experiments = load_all_experiments()
    
    if not experiments:
        print("\n⚠️  No experiments found in experiments/ directory")
        return
    
    # Filter out experiments without canonical metrics
    valid_experiments = []
    for exp in experiments:
        if 'final_metrics' in exp and 'canonical' in exp.get('final_metrics', {}):
            valid_experiments.append(exp)
    
    if not valid_experiments:
        print("\n⚠️  No experiments with canonical metrics found")
        return
    
    # Sort by canonical avg R@1
    valid_experiments = sorted(
        valid_experiments,
        key=lambda x: x.get('final_metrics', {}).get('canonical', {}).get('average_recall_at_1', 0),
        reverse=True
    )
    
    print(f"\n{'='*120}")
    print(f"EXPERIMENT COMPARISON (sorted by canonical Avg R@1)")
    print(f"{'='*120}")
    print(f"{'Experiment':<25} {'Loss Type':<15} {'Alpha':<8} {'Top-K':<8} {'Tau':<8} {'T2I R@1':<10} {'I2T R@1':<10} {'Avg R@1':<10}")
    print(f"{'-'*120}")
    
    for exp in valid_experiments:
        config = exp.get('config', {})
        metrics = exp.get('final_metrics', {}).get('canonical', {})
        
        exp_name = config.get('experiment_name', 'unknown')
        loss_type = config.get('loss_type', 'unknown')
        alpha = config.get('alpha', 'N/A')
        top_k = config.get('top_k', 'N/A')
        tau = config.get('tau', 'N/A')
        
        t2i_r1 = metrics.get('text_to_image', {}).get('R@1', 0)
        i2t_r1 = metrics.get('image_to_text', {}).get('R@1', 0)
        avg_r1 = metrics.get('average_recall_at_1', 0)
        
        alpha_str = f"{alpha:.2f}" if isinstance(alpha, (int, float)) else str(alpha)
        top_k_str = str(top_k)
        tau_str = f"{tau:.3f}" if isinstance(tau, (int, float)) else str(tau)
        
        print(f"{exp_name:<25} {loss_type:<15} {alpha_str:<8} {top_k_str:<8} {tau_str:<8} "
              f"{t2i_r1:<10.2f} {i2t_r1:<10.2f} {avg_r1:<10.2f}")
    
    print(f"{'='*120}\n")
    
    # Print best per loss type
    print("\n📊 BEST PER LOSS TYPE:")
    print("-" * 80)
    
    by_type = {}
    for exp in valid_experiments:
        loss_type = exp['config'].get('loss_type', 'unknown')
        avg_r1 = exp['final_metrics']['canonical']['average_recall_at_1']
        
        if loss_type not in by_type or avg_r1 > by_type[loss_type]['avg_r1']:
            by_type[loss_type] = {
                'avg_r1': avg_r1,
                'exp_name': exp['config'].get('experiment_name', 'unknown'),
                'config': {k: v for k, v in exp['config'].items() 
                          if k in ['alpha', 'top_k', 'tau', 'queue_size']}
            }
    
    for loss_type, info in sorted(by_type.items(), key=lambda x: x[1]['avg_r1'], reverse=True):
        config_str = ', '.join(f"{k}={v}" for k, v in info['config'].items() if v != 'N/A')
        if config_str:
            print(f"  {loss_type:<20}: {info['avg_r1']:.2f}% ({info['exp_name']}) [{config_str}]")
        else:
            print(f"  {loss_type:<20}: {info['avg_r1']:.2f}% ({info['exp_name']})")
    
    # Print delta from baseline
    baseline_r1 = None
    for exp in valid_experiments:
        if exp['config'].get('loss_type') == 'baseline':
            baseline_r1 = exp['final_metrics']['canonical']['average_recall_at_1']
            break
    
    if baseline_r1 is not None:
        print(f"\n📈 IMPROVEMENT OVER BASELINE ({baseline_r1:.2f}%):")
        print("-" * 80)
        for loss_type, info in sorted(by_type.items(), key=lambda x: x[1]['avg_r1'], reverse=True):
            if loss_type != 'baseline':
                delta = info['avg_r1'] - baseline_r1
                delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
                symbol = "✅" if delta > 0 else "❌"
                print(f"  {symbol} {loss_type:<20}: {delta_str}%")
    
    print(f"\n{'='*80}\n")
