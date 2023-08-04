
# GPT Embeddings Script

This Python script is a command-line program that manipulates sentence embeddings and generates new sentences based on user input.

## Dependencies

The script imports various libraries to perform its operations:

- `os`: Used for interacting with the OS, mainly to access environment variables.
- `contextlib`: Provides the `nullcontext` which is used for handling precision in torch calculations.
- `torch`: PyTorch library used for tensor computations and neural network operations.
- `openai`: Used to interact with the OpenAI API.
- `tiktoken`: A tool from OpenAI for encoding and decoding tokens.
- `GPT` and `GPTConfig` from `model`: Classes used to create a GPT model and its configuration.
- `hf_hub_download` from `huggingface_hub`: Downloads a model checkpoint from HuggingFace's model hub.
- `dotenv`: Loads environment variables from a .env file.

## Setup and Configuration

1. Install the required dependencies with pip:

```shell
pip install -r requirements.txt
```

2. Set the `OPENAI_API_KEY` environment variable with your OpenAI API key in a `.env` file:

```env
OPENAI_API_KEY=your-api-key
```

Replace "your-api-key" with your actual OpenAI API key.

3. Run the script with Python:

```shell
python main.py --device=cpu
```

## Execution

In the script execution phase, you will be prompted to enter the starting sentence, the sequence to subtract, and the sequence to add. The script will output a sentence based on these inputs. You can continue providing inputs or quit by typing 'y' when asked if you want to quit.

Please note that the `--device=cpu` argument forces the script to run on the CPU. If you have a CUDA-compatible GPU and want to use it, you can replace `cpu` with `cuda`.

# wikivec2text

Simple embedding -> text model trained on a tiny subset of Wikipedia sentences ([~7m](https://www.kaggle.com/datasets/mikeortman/wikipedia-sentences)) embedded via text-embedding-ada-002. 

Used to [demonstrate the arithmetic properties of sentence embeddings](https://twitter.com/MF_FOOM/status/1687219083761385475), e.g:

<img width="350" src="https://github.com/MF-FOOM/wikivec2text/assets/141304309/5b1be1fc-f447-402a-939a-a275bac8fd4d">

> **Warning**
> The [checkpoint on HF](https://huggingface.co/MF-FOOM/wikivec2text) is a `gpt2-small` finetuned heavily on a small subset of well-formatted Wikipedia sentences like `Ten years later, Merrill Lynch merged with E. A. Pierce.`
>
> Since these sentences are so structured and formal, it's very easy to go OOD if you're not careful. Stick to informational sentences that could plausibly appear in an encyclopedia, always start sentences with a capital letter, end with a period, etc.

## Acknowledgements

Built on [nanoGPT](https://github.com/karpathy/nanoGPT).
