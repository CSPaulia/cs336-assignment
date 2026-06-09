import torch

def dataloading(
        data: torch.Tensor,
        batch_size: int,
        context_length: int,
        device: torch.device
    ):
    data_length = data.shape[0]

    start_indices = torch.randint(0, data_length - context_length, (batch_size,))
    input_sequences = torch.stack([data[i:i+context_length] for i in start_indices])
    target_sequences = torch.stack([data[i+1:i+context_length+1] for i in start_indices])
    return (input_sequences.to(device), target_sequences.to(device))