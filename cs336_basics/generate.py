import torch

def generate_text(model, tokenizer, start_text, max_length=100, tau=1.0):
    model.eval()
    input_ids = tokenizer.encode(start_text)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(model.device)

    end_token_id = tokenizer.encode("<|endoftext|>")[0]

    generated_ids = input_ids
    for _ in range(max_length):
        with torch.no_grad():
            logits = model(generated_ids)
            next_token_id = torch.argmax(torch.softmax(logits[:, -1, :] / tau, dim=-1), dim=-1).unsqueeze(0)
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

        if next_token_id.item() == end_token_id:
            break

    generated_text = tokenizer.decode(generated_ids.squeeze().tolist())
    return generated_text