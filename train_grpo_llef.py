import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class GRPOConfig:
    """Configuration for GRPO training"""
    learning_rate: float = 1e-6  # DeepSeekMath GRPO paper value
    group_size: int = 5  # Number of responses per prompt
    ppo_epochs: int = 1  # Number of optimization epochs per batch (default=1 for offline GRPO)
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.01
    kl_coef: float = 0.1
    temperature: float = 1.0  # For sampling
    top_k: int = 50
    top_p: float = 0.9
    max_new_tokens: int = 256
    max_prompt_length: int = 8000  # Max prompt length for generation logging
    clamp_log_ratio: bool = False  # Enable log ratio clamping for numerical stability
    log_ratio_range: float = 20.0  # Clamp log ratio to [-range, +range] if enabled
    use_early_stopping: bool = False  # Enable KL-based early stopping (PPO-style, not in GRPO papers)
    save_steps: int = 20  # Save checkpoint every N steps (0 = disable periodic saving)
    save_dir: str = "./checkpoints"  # Directory to save checkpoints
    log_generation_steps: int = 5  # Generate and log model outputs every N batches (0 = disable)
    max_generation_tokens: int = 256  # Max tokens for logging generations

class GRPOTrainer:
    """Group Relative Policy Optimization Trainer - Adapted for Pre-computed Datasets"""

    def __init__(
        self,
        model_name: str,
        tokenizer: AutoTokenizer,
        config: GRPOConfig,
        train_loader,  # DataLoader with GRPOTextRewardDataset
        val_loader=None,  # Optional validation DataLoader
        device: str = 'cuda'
    ):
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Initialize policy model
        self.policy = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # Initialize reference model (frozen) - offload to CPU to save GPU memory
        print("Loading reference model to CPU to save GPU memory...")
        self.ref_policy = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu"  # Keep on CPU, only move to GPU when needed
        )
        for param in self.ref_policy.parameters():
            param.requires_grad = False

        # NO VALUE HEAD IN GRPO!
        # The value function is replaced by group statistics

        # Only one optimizer needed
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate
        )

    def get_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probabilities for sequences"""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        logits = outputs.logits
        
        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = response_mask[..., 1:].contiguous()
        
        # Calculate log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        selected_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Apply response mask - only count response tokens
        selected_log_probs = selected_log_probs * shift_mask
        
        return selected_log_probs.sum(dim=1)
    
    def compute_group_loss(
        self,
        input_ids: torch.Tensor,        # [group_size, seq_len]
        attention_masks: torch.Tensor,   # [group_size, seq_len]
        response_masks: torch.Tensor,    # [group_size, seq_len]
        rewards: torch.Tensor,           # [group_size] - pre-normalized advantages
        old_log_probs: torch.Tensor,     # [group_size] - log probs from policy before updates
        ref_log_probs: torch.Tensor      # [group_size] - log probs from frozen reference model
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a single group (no optimizer step)"""

        # Rewards from dataset are already normalized as advantages
        # (rewards - mean) / (std + eps) was done in GRPOTextRewardDataset

        # Step 1: Compute current policy log probs (with gradients)
        log_probs = self.get_log_probs(
            self.policy,
            input_ids,
            attention_masks,
            response_masks
        )

        # Step 2: Compute KL penalty (approximation using log prob difference)
        # KL penalty = log(π_current) - log(π_ref) is an approximation to KL(π_current || π_ref)
        # We penalize when current policy diverges from reference
        with torch.no_grad():
            # Use detached log_probs to avoid double gradient flow
            kl_penalty = (log_probs.detach() - ref_log_probs)

        # Subtract KL penalty from rewards (penalize divergence from reference)
        final_rewards = rewards - self.config.kl_coef * kl_penalty

        # Re-normalize advantages after adding KL penalty
        mean_reward = final_rewards.mean()
        std_reward = final_rewards.std()
        if torch.sum(torch.abs(final_rewards - mean_reward)) < 1e-5:
            final_advantages = torch.zeros(final_rewards.shape).to(self.device)
        else:
            final_advantages = (final_rewards - mean_reward) / (std_reward + 1e-8)

        # Compute probability ratio
        log_ratio = log_probs - old_log_probs
        if self.config.clamp_log_ratio:
            # Optional: Clamp log ratio for numerical stability
            log_ratio = torch.clamp(log_ratio,
                                   min=-self.config.log_ratio_range,
                                   max=self.config.log_ratio_range)
        ratio = torch.exp(log_ratio)

        # PPO clipped objective
        surr1 = ratio * final_advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - self.config.clip_ratio,
            1.0 + self.config.clip_ratio
        ) * final_advantages

        # Take minimum (pessimistic bound)
        policy_loss = -torch.min(surr1, surr2).mean()

        # Total loss (no value loss, no backward pass here)
        loss = policy_loss

        # Compute KL for monitoring (approximation: avg log prob difference)
        with torch.no_grad():
            kl = (log_probs.detach() - ref_log_probs).mean()

        # Return loss and metrics (no optimizer step)
        metrics = {
            'kl': kl.item(),
            'mean_reward': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'mean_advantage': final_advantages.mean().item(),
            'advantage_std': final_advantages.std().item()
        }

        return loss, metrics
    
    def save_checkpoint(self, step: int, epoch: int):
        """Save model checkpoint"""
        import os
        os.makedirs(self.config.save_dir, exist_ok=True)

        checkpoint_dir = f"{self.config.save_dir}/step_{step}_epoch_{epoch}"
        self.policy.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        print(f"  Saved checkpoint to {checkpoint_dir}")

    def log_generations(self, step: int):
        """Generate and log model outputs for monitoring training progress"""
        self.policy.eval()

        prompts = []

        # Load and add AAPL prompt from file
        try:
            with open("aapl_prompt.txt", "r") as f:
                aapl_prompt = f.read().strip()
                prompts.append(("Data Prompt (AAPL)", aapl_prompt))
        except Exception as e:
            print(f"Warning: Could not load AAPL prompt: {e}")

        # Add generic prompt
        generic_prompt = "Is it wise to invest in tech stocks like Apple, Microsoft around 2020?"
        prompts.append(("Generic Prompt", generic_prompt))

        print(f"\n{'='*80}")
        print(f"Generation Log at Step {step}")
        print(f"{'='*80}")

        with torch.no_grad():
            for prompt_name, prompt_text in prompts:
                # Tokenize prompt
                inputs = self.tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_prompt_length
                ).to(self.device)

                # Generate
                outputs = self.policy.generate(
                    **inputs,
                    max_new_tokens=self.config.max_generation_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

                # Decode
                generated_text = self.tokenizer.decode(
                    outputs[0][len(inputs['input_ids'][0]):],
                    skip_special_tokens=True
                )

                print(f"\n[{prompt_name}]")
                print(f"Prompt: {prompt_text}")
                print(f"Generated: {generated_text}")
                print(f"{'-'*80}")

        print(f"{'='*80}\n")
        self.policy.train()

    def train(
        self,
        num_epochs: int = 10
    ):
        """Main training loop using dataloaders"""

        global_step = 0  # Track total number of batches processed

        for epoch in range(num_epochs):
            self.policy.train()
            epoch_metrics = {
                'loss': 0,
                'kl': 0,
                'mean_reward': 0,
                'reward_std': 0
            }

            for batch_idx, batch in enumerate(self.train_loader):
                print(f"epoch :{epoch}, batch: {batch_idx}")
                global_step += 1
                # Extract data from batch (already on CPU, move to device)
                input_ids = batch['input_ids'].to(self.device)  # [batch_size, group_size, seq_len]
                attention_masks = batch['attention_mask'].to(self.device)
                response_masks = batch['response_mask'].to(self.device)
                rewards = batch['rewards'].to(self.device)  # [batch_size, group_size]

                # Compute old_log_probs and ref_log_probs ONCE before PPO epochs
                # These remain fixed across all PPO epochs for this batch
                with torch.no_grad():
                    old_log_probs_batch = []
                    ref_log_probs_batch = []

                    for i in range(input_ids.shape[0]):
                        group_input_ids = input_ids[i]
                        group_attention_masks = attention_masks[i]
                        group_response_masks = response_masks[i]

                        # Get old policy log probs (current policy state)
                        old_log_probs = self.get_log_probs(
                            self.policy,
                            group_input_ids,
                            group_attention_masks,
                            group_response_masks
                        )
                        old_log_probs_batch.append(old_log_probs.detach())

                        # Get reference model log probs (move to GPU temporarily)
                        self.ref_policy = self.ref_policy.to(self.device)
                        ref_log_probs = self.get_log_probs(
                            self.ref_policy,
                            group_input_ids,
                            group_attention_masks,
                            group_response_masks
                        )
                        ref_log_probs_batch.append(ref_log_probs.detach())
                        # Move reference model back to CPU
                        self.ref_policy = self.ref_policy.to('cpu')
                        torch.cuda.empty_cache()

                # PPO epochs: multiple passes over the same batch
                for ppo_epoch in range(self.config.ppo_epochs):
                    # Zero gradients at start of each PPO epoch
                    self.optimizer.zero_grad()

                    # Accumulate gradients over all groups in batch
                    batch_loss = 0
                    batch_kl = 0
                    batch_reward = 0
                    batch_reward_std = 0

                    for i in range(input_ids.shape[0]):
                        print(f"group: {i}")
                        # Get single group
                        group_input_ids = input_ids[i]  # [group_size, seq_len]
                        group_attention_masks = attention_masks[i]
                        group_response_masks = response_masks[i]
                        group_rewards = rewards[i]  # [group_size]

                        # Compute loss for this group (no optimizer step)
                        loss, metrics = self.compute_group_loss(
                            input_ids=group_input_ids,
                            attention_masks=group_attention_masks,
                            response_masks=group_response_masks,
                            rewards=group_rewards,
                            old_log_probs=old_log_probs_batch[i],
                            ref_log_probs=ref_log_probs_batch[i]
                        )

                        # Backward pass - accumulate gradients
                        loss.backward()
                        torch.cuda.empty_cache()
                        # Accumulate metrics
                        loss = loss.detach()

                        batch_loss += loss.item()
                        batch_kl += metrics['kl']
                        batch_reward += metrics['mean_reward']
                        batch_reward_std += metrics['reward_std']

                    # Gradient clipping after accumulating all groups
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.config.max_grad_norm
                    )

                    # Single optimizer step for entire batch
                    self.optimizer.step()

                # Average metrics over groups in batch
                num_groups = input_ids.shape[0]
                epoch_metrics['loss'] += batch_loss / num_groups
                epoch_metrics['kl'] += batch_kl / num_groups
                epoch_metrics['mean_reward'] += batch_reward / num_groups
                epoch_metrics['reward_std'] += batch_reward_std / num_groups
                print(f"epoch :{epoch}, batch: {batch_idx},  loss: {batch_loss / num_groups}, kl: {batch_kl / num_groups}")

                # Print progress
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Batch {batch_idx + 1}/{len(self.train_loader)} (step {global_step}): "
                          f"loss={batch_loss/num_groups:.4f}, "
                          f"reward={batch_reward/num_groups:.3f}")

                # Periodic generation logging
                if self.config.log_generation_steps > 0 and (batch_idx + 1) % self.config.log_generation_steps == 0:
                    self.log_generations(global_step)

                # Periodic checkpoint saving
                if self.config.save_steps > 0 and global_step % self.config.save_steps == 0:
                    self.save_checkpoint(global_step, epoch + 1)

            # Average metrics over all batches
            num_batches = len(self.train_loader)
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches

            print(f"\nEpoch {epoch + 1}/{num_epochs} Summary:")
            print(f"  Average Loss: {epoch_metrics['loss']:.4f}")
            print(f"  Average KL: {epoch_metrics['kl']:.4f}")
            print(f"  Average Reward: {epoch_metrics['mean_reward']:.4f}")
            print(f"  Average Reward Std: {epoch_metrics['reward_std']:.4f}")

# Example usage with GRPOTextRewardDataset
def main():
    from torch.utils.data import DataLoader
    from proc_data_llef import GRPOTextRewardDataset

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataset and dataloader
    train_dataset = GRPOTextRewardDataset(
        data_path="llef_first_test_data_list.txt",
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=1000,
        group_size=5
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size= 1,#104,  # Number of groups per batch (104 groups × 5 responses = 520 samples total)
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    # Optional: Create validation dataset/loader
    # val_dataset = GRPOTextRewardDataset(...)
    # val_loader = DataLoader(val_dataset, ...)

    config = GRPOConfig(
        learning_rate=1e-6,  # DeepSeekMath GRPO paper value
        group_size=5,  # Must match dataset group_size
        ppo_epochs=1,  # Set to 1 for offline GRPO with abundant pre-computed data
        clip_ratio=0.2,
        kl_coef=0.04  # DeepSeekMath GRPO paper value
    )

    trainer = GRPOTrainer(
        model_name=model_name,
        tokenizer=tokenizer,
        config=config,
        train_loader=train_loader,
        val_loader=None,
        device='cuda'
    )

    # Train using dataloaders
    trainer.train(num_epochs=5)

    # Save
    trainer.policy.save_pretrained("./grpo_model")
    tokenizer.save_pretrained("./grpo_model")

if __name__ == "__main__":
    main()