# Mini GPT --- Character-Level Language Model

A small GPT-style decoder-only Transformer implemented with PyTorch.
This project demonstrates the core ideas behind autoregressive language
models: token embeddings, positional embeddings, causal self-attention,
feed-forward layers, next-token prediction, and text generation.

## Project overview

The model is trained on a tiny, repeated text corpus and learns to
predict the next character in a sequence. It then generates new
characters one at a time using its previous output as context.

> This is an educational implementation, not a production LLM. Its small
> dataset, character-level tokenizer, and compact architecture limit the
> quality of generated text.

## Technologies

-   Python
-   PyTorch
-   Character-level tokenization
-   Decoder-only Transformer architecture

## Model configuration

  Setting                             Value
  ------------------------------- ---------
  Vocabulary size                        22
  Embedding dimension                    64
  Attention heads                         4
  Transformer blocks                      2
  Context length (`block_size`)          32
  Batch size                             32
  Training steps                        500
  Learning rate                       0.001
  Parameters                        104,982

## Training results

Observed run:

    Step   Training loss
  ------ ---------------
       0          3.1206
     100          0.8102
     200          0.1461
     300          0.1059
     400          0.0969

The loss decreased substantially, showing that the model learned
patterns in the training corpus. Generated output included recognizable
training phrases, with some spelling errors. This is expected for such a
small model and dataset.

## Architecture

1.  **Character tokenizer** maps each character to an integer ID.
2.  **Token embedding** maps token IDs to learned vectors.
3.  **Positional embedding** adds sequence-position information.
4.  **GPT blocks** apply causal multi-head self-attention, residual
    connections, layer normalization, and a feed-forward network.
5.  **Output layer** produces vocabulary logits for each sequence
    position.
6.  **Cross-entropy loss** compares logits with the next-character
    targets.
7.  **AdamW optimizer** updates model parameters.
8.  **Autoregressive generation** samples one next character at a time.

## Run locally

From the project directory:

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch
python mini_gpt.py
```

The project was run on CPU with PyTorch `2.1.2+cpu`; a GPU is not
required.

## Project structure

``` text
C:\AISkills
├── mini_gpt.py
├── README.md
└── venv/
```

## Key learning outcomes

-   Understand character-level tokenization and token IDs.
-   Convert token IDs into embeddings.
-   Add positional information to embeddings.
-   Apply a causal mask so positions cannot attend to future tokens.
-   Train a decoder-only Transformer with next-token targets.
-   Generate text autoregressively.

## Limitations and possible improvements

-   Use a larger and more diverse training corpus.
-   Introduce a subword tokenizer such as BPE.
-   Add train/validation splits and evaluate on held-out text.
-   Save and reload model checkpoints.
-   Experiment with context length, model size, learning rate, and
    training duration.
-   Add temperature and top-k/top-p sampling controls.

## Author

Harish Pilla

## Disclaimer

This project is intended for learning and experimentation. It is not
suitable for production use or factual question answering.
