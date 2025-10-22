import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pickle
from typing import Dict, List, Tuple
import os, pdb

class TextRewardDataset(Dataset):
    """Dataset for text-reward pairs for RLHF/GRPO training"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        max_prompt_length: int = 8000,
    ):
        """
        Args:
            data_path: Path to txt file containing list of pkl files
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            max_prompt_length: Maximum prompt length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

        # Load data from pkl files
        self.data = []
        with open(data_path, 'r') as f:
            pkl_files = [line.strip() for line in f if line.strip()]

        data_dir = os.path.dirname(data_path) + 'llef_alignment_data/'
        for pkl_file in pkl_files:
            pkl_path = os.path.join(data_dir, pkl_file)
            with open(pkl_path, 'rb') as f:
                D = pickle.load(f)
                self.data.extend(D['saveL'])

        # Print data format
        print(f"Loaded {len(self.data)} items from {len(pkl_files)} pkl files")
        if len(self.data) > 0:
            sample = self.data[0]
            print(f"\nData format (tuple with {len(sample)} elements):")
            print(f"  [0] filename: {type(sample[0]).__name__}")
            print(f"  [1] filtered_count: {type(sample[1]).__name__}")
            print(f"  [2] ticker_idx: {type(sample[2]).__name__}")
            print(f"  [3] batch_idx: {type(sample[3]).__name__}")
            print(f"  [4] ticker: {type(sample[4]).__name__}")
            print(f"  [5] messages: {type(sample[5]).__name__}")
            print(f"  [6] assistant_response: {type(sample[6]).__name__}")
            print(f"  [7] eval_score_int: {type(sample[7]).__name__}")
            print(f"  [8] r: {type(sample[8]).__name__}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # (filename, filtered_count, ticker_idx, batch_idx, ticker, messages, assistant_response, eval_score_int, r) 
        prompt = item[5][1]['content']
        response = item[6]
        reward = item[8].item()
        
        # Tokenize prompt
        
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            padding=False,
            return_tensors=None
        )
        
        # Tokenize full sequence (prompt + response)
        full_text = prompt + response
        full_tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_prompt_length + self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create attention mask for response only
        prompt_length = len(prompt_tokens['input_ids'])
        response_mask = torch.zeros_like(full_tokens['input_ids'])
        response_mask[:, prompt_length:] = 1
        
        return {
            'input_ids': full_tokens['input_ids'].squeeze(),
            'attention_mask': full_tokens['attention_mask'].squeeze(),
            'response_mask': response_mask.squeeze(),
            'prompt_length': prompt_length,
            'rewards': torch.tensor(reward, dtype=torch.float32)
        }

class GRPOTextRewardDataset(Dataset):
    """Dataset for GRPO training with grouped samples (group_size=5)"""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        max_prompt_length: int = 8000,
        group_size: int = 5
    ):
        """
        Args:
            data_path: Path to txt file containing list of pkl files
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            max_prompt_length: Maximum prompt length
            group_size: Number of items per group (default 5)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.group_size = group_size

        # Load data from pkl files and group by (filename, ticker_idx)
        temp_data = []
        with open(data_path, 'r') as f:
            pkl_files = [line.strip() for line in f if line.strip()]

        data_dir = os.path.dirname(data_path) + 'llef_alignment_data/'
        for pkl_file in pkl_files:
            pkl_path = os.path.join(data_dir, pkl_file)
            with open(pkl_path, 'rb') as f:
                D = pickle.load(f)
                temp_data.extend(D['saveL'])

        # Group items by (filename, ticker_idx)
        from collections import defaultdict
        groups_dict = defaultdict(list)

        for item in temp_data:
            # (filename, filtered_count, ticker_idx, batch_idx, ticker, messages, assistant_response, eval_score_int, r)
            filename = item[0]
            ticker_idx = item[2]
            batch_idx = item[3]
            key = (filename, ticker_idx)
            groups_dict[key].append((batch_idx, item))

        # Sort each group by batch_idx and create groups of group_size
        self.data = []
        for key, indexed_items in groups_dict.items():
            # Sort by batch_idx
            indexed_items.sort(key=lambda x: x[0])
            # Extract just the data items (remove batch_idx from tuple)
            group = [item[1] for item in indexed_items]

            # Verify group size
            if len(group) == self.group_size:
                self.data.append(group)
            else:
                print(f"Warning: Group {key} has {len(group)} items, expected {self.group_size}. Skipping.")

        to_del_list = []
        # Normalize rewards within each group
        for group_idx, group in enumerate(self.data):
            rewards = torch.tensor([item[8].item() if torch.is_tensor(item[8]) else item[8] for item in group], dtype=torch.float32)
            mean_reward = rewards.mean()
            std_reward = rewards.std()

            # Normalize: reward_tilde = (reward - mean) / (std + eps)
            if torch.sum(torch.abs(rewards - mean_reward)) < 1e-5: 
                normalized_rewards = torch.zeros(rewards.shape)
                to_del_list.append(group_idx)
            else: 
                normalized_rewards = (rewards - mean_reward) / (std_reward + 1e-8)
            
            # Update each item in the group with normalized reward
            for i in range(len(group)):
                # Replace the original reward (index 8) with normalized reward
                item = list(group[i])
                item[8] = normalized_rewards[i]
                self.data[group_idx][i] = tuple(item)
        
        ## implement deletion of zero rewards groups 
        for index in sorted(to_del_list, reverse=True):
            del self.data[index]

        print(f"\nGRPO Dataset loaded:")
        print(f"  Total groups: {len(self.data)}")
        print(f"  Group size: {self.group_size}")
        print(f"  Total items: {len(self.data) * self.group_size}")
        print(f"  Loaded from {len(pkl_files)} pkl files")

        if len(self.data) > 0:
            for idx in range(10):
                sample_group = self.data[idx]
                print(f"\nSample group (group {idx}):")
                print(f"  ticker_idx: {sample_group[0][2]}, ticker: {sample_group[0][4]}")
                print(f"  batch_idx values: {[item[3] for item in sample_group]}")
                print(f"  normalized rewards: {[item[8].item() if torch.is_tensor(item[8]) else item[8] for item in sample_group]}")        

        ## [TODO] 
        ## implement train/val split using tickers has the split hash 
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        group = self.data[idx]  # List of 5 tuples

        # Process each item in the group
        input_ids_list = []
        attention_mask_list = []
        response_mask_list = []
        prompt_length_list = []
        rewards_list = []

        for item in group:
            # (filename, filtered_count, ticker_idx, batch_idx, ticker, messages, assistant_response, eval_score_int, r)
            prompt = item[5][1]['content']
            response = item[6]
            reward = item[8].item() if torch.is_tensor(item[8]) else item[8]

            # Tokenize prompt
            prompt_tokens = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_prompt_length,
                padding=False,
                return_tensors=None
            )

            # Tokenize full sequence (prompt + response)
            full_text = prompt + response
            full_tokens = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_prompt_length + self.max_length,
                padding='max_length',
                return_tensors='pt'
            )

            # Create attention mask for response only
            prompt_length = len(prompt_tokens['input_ids'])
            response_mask = torch.zeros_like(full_tokens['input_ids'])
            response_mask[:, prompt_length:] = 1

            # Append to lists
            input_ids_list.append(full_tokens['input_ids'].squeeze())
            attention_mask_list.append(full_tokens['attention_mask'].squeeze())
            response_mask_list.append(response_mask.squeeze())
            prompt_length_list.append(prompt_length)
            rewards_list.append(reward)

        # Stack all tensors to create batch dimension of size group_size (5)
        return {
            'input_ids': torch.stack(input_ids_list),  # Shape: (5, max_length)
            'attention_mask': torch.stack(attention_mask_list),  # Shape: (5, max_length)
            'response_mask': torch.stack(response_mask_list),  # Shape: (5, max_length)
            'prompt_length': torch.tensor(prompt_length_list, dtype=torch.long),  # Shape: (5,)
            'rewards': torch.tensor(rewards_list, dtype=torch.float32)  # Shape: (5,)
        }

def create_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer,
    batch_size: int = 8,
    max_length: int = 512
):
    """Create train and validation dataloaders"""
    
    train_dataset = TextRewardDataset(
        train_path, 
        tokenizer, 
        max_length=max_length
    )
    
    val_dataset = TextRewardDataset(
        val_path,
        tokenizer,
        max_length=max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

# Example usage
def prepare_data():
    from transformers import AutoTokenizer
    
    # Initialize tokenizer
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_path="train_data.json",
        val_path="val_data.json",
        tokenizer=tokenizer,
        batch_size=4,
        max_length=512
    )
    
    return tokenizer, train_loader, val_loader

if __name__ == "__main__":
    # Unit test for __init__ function
    from transformers import AutoTokenizer

    print("Testing TextRewardDataset __init__ function...")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test loading data
    dataset = TextRewardDataset(
        data_path="llef_first_test_data_list.txt",
        tokenizer=tokenizer
    )

    print(f"\nDataset length: {len(dataset)}")
    print("\n__init__ test passed!")

    # Unit test for __getitem__ function
    print("\n" + "="*50)
    print("Testing __getitem__ function...")

    # Test single item
    item = dataset[0]
    print(f"\nSingle item test:")
    print(f"  Keys: {list(item.keys())}")
    print(f"  input_ids shape: {item['input_ids'].shape}")
    print(f"  attention_mask shape: {item['attention_mask'].shape}")
    print(f"  response_mask shape: {item['response_mask'].shape}")
    print(f"  prompt_length: {item['prompt_length']}")
    print(f"  reward: {item['rewards'].item()}")

    # Test multiple data points
    print(f"\nTesting multiple data points (indices 0, 10, 100):")
    for idx in [0, 10, 100]:
        if idx < len(dataset):
            item = dataset[idx]
            print(f"  Index {idx}: reward={item['rewards'].item():.4f}, prompt_len={item['prompt_length']}")

    # Test batch with DataLoader
    print(f"\nTesting batch loading with DataLoader:")
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"  Batch rewards shape: {batch['rewards'].shape}")
    print(f"  Batch rewards: {batch['rewards'].tolist()}")

    print("\n__getitem__ test passed!")

    # Unit test for GRPOTextRewardDataset __init__
    print("\n" + "="*50)
    print("Testing GRPOTextRewardDataset __init__ function...")

    grpo_dataset = GRPOTextRewardDataset(
        data_path="llef_first_test_data_list.txt",
        tokenizer=tokenizer
    )

    print(f"\nGRPO Dataset length (number of groups): {len(grpo_dataset)}")
    print("\nGRPO __init__ test passed!")

    # Unit test for GRPOTextRewardDataset __getitem__
    print("\n" + "="*50)
    print("Testing GRPOTextRewardDataset __getitem__ function...")
    
    grpo_item = grpo_dataset[0]
    print(f"\nGRPO item (group) test:")
    print(f"  Keys: {list(grpo_item.keys())}")
    print(f"  input_ids shape: {grpo_item['input_ids'].shape}")
    print(f"  attention_mask shape: {grpo_item['attention_mask'].shape}")
    print(f"  response_mask shape: {grpo_item['response_mask'].shape}")
    print(f"  prompt_length shape: {grpo_item['prompt_length'].shape}")
    print(f"  rewards shape: {grpo_item['rewards'].shape}")
    print(f"  rewards: {grpo_item['rewards'].tolist()}")

    print("\nGRPO __getitem__ test passed!")
    print("\nAll tests passed!")
