import argparse
from transformers import LlamaTokenizer LlamaForCausalLM
from markov_chain_inference.model.markov_chain import MarkovChain
from markov_chain_inference.model.state_space import StateSpace
from markov_chain_inference.data.data_loader import get_data_loader
from markov_chain_inference.utils.config import Config

def main(data_path, model_path):
    config = Config()
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    data_loader = get_data_loader(data_path, tokenizer, config.batch_size, config.max_length)

    vocab = tokenizer.get_vocab()
    state_space = StateSpace(vocab)
    markov_chain = MarkovChain(len(vocab))

    for input_ids, _ in data_loader:
        sequences = [seq.tolist() for seq in input_ids]
        markov_chain.update_transitions(sequences)

    markov_chain.normalize()
    print("Transition Matrix:", markov_chain.transition_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Markov Chain-Based Inference")
    parser.add_argument('--data_path', type=str, required=True, help="Path to input data")
    parser.add_argument('--model_path', type=str, required=True, help="Path to LLaMA model")
    args = parser.parse_args()
    main(args.data_path, args.model_path)
