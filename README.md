# jax GPT-2 Exercises

this repository has two jupyter notebooks, meant to guide you through building a GPT-2 style language model using JAX. its meant to be an educational resource for learning about both some basics of jax as well as transformer based language models

click 'Open in Colab' to run this notebook. you'll be working on a personal copy and any changes you make won't affect the original notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arb8020/jax-gpt2-exercises/blob/main/jax_gpt2.ipynb)

## overview

the notebook covers the following topics:

1. intro to GPT/language modeling
2. jax basics like jit/vmap/grad/pytrees
3. model training with sgd/adamw
4. token/positional embeddings
5. attention
6. feed forward models
7. layer norm
8. residuals
9. byte pair encoding
10. sampling strategies like top-k
11. loading and using GPT-2 weights/tokenizer

## prerequisites

to get the most out of this notebook, you should have:
- basic python familiarity
- interest in learning about language models and JAX

## contributing

if you have any ideas on how to make this notebook better (bug fixes, improved explanations, best practices, etc) please feel free open an issue, or message me on twitter/X, i want to make this notebook as useful as possible!

in the future, i hope to add/make new notebooks for the following topics:

- inference performance improvements with kv cache
- distributed training with jax
- rotary positional encoding
- llama forward pass
- sparse autoencoders
- ...

## acknowledgements

- @_xjdr on twitter for opening my eyes to jax
- https://jax.readthedocs.io/en/latest/
- arxiv
