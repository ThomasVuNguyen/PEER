import argparse
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from peer.model import PEERLanguageModel


def load_model(checkpoint_path, device):
    """Load the trained PEER model from checkpoint."""
    # Model hyperparameters (must match training config)
    vocab_size = 50257
    dim = 256
    num_layers = 8
    num_heads = 8
    num_experts = 256 * 256
    top_k = 16

    model = PEERLanguageModel(vocab_size, dim, num_layers, num_heads, num_experts, top_k)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Handle DDP wrapped model (keys start with 'module.')
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.9, device='cuda'):
    """Generate text from a prompt using autoregressive sampling."""
    model.eval()

    # Tokenize input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    generated = input_ids

    for _ in range(max_new_tokens):
        # Truncate to max sequence length (512) if needed
        context = generated[:, -512:]

        # Forward pass
        logits = model(context)

        # Get logits for the last token
        next_token_logits = logits[:, -1, :]

        # Apply temperature
        next_token_logits = next_token_logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')

        # Sample from the filtered distribution
        probs = F.softmax(next_token_logits, dim=-1)

        # Handle edge case where all logits are -inf (due to aggressive filtering)
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            # Fallback: sample from uniform distribution over vocabulary
            next_token = torch.randint(0, probs.size(-1), (1, 1), device=device)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        # Append to generated sequence
        generated = torch.cat([generated, next_token], dim=1)

        # Stop if EOS token is generated
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode and return generated text
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output_text


def main():
    parser = argparse.ArgumentParser(description='Generate text using trained PEER model')
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt for generation')
    parser.add_argument('--checkpoint', type=str, default='final_peer_language_model.pth', help='Path to model checkpoint')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling (0 to disable)')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda/cpu)')

    args = parser.parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    device = torch.device(args.device)

    print(f"Loading model from {args.checkpoint}...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.checkpoint, device)
    print("Model loaded successfully!")

    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device
    )

    print(f"Generated:\n{output}")


if __name__ == "__main__":
    main()
