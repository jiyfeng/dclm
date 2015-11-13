# Document Context Language Models (DCLMs) #

A set of document-level language models with contextual information. Please refer to our [ICLR 2016 submission](http://arxiv.org/abs/1511.03962) for more technical details.

## Getting started ##

You need the [Boost C++ libraries]() (>=1.56) to save/load word vocabulary and trained models. 

## Building ##

1. First you need to fetch the [cnn library](https://github.com/clab/cnn) into the main directory, then follow the instruction to get additional libraries and compile cnn.

2. To compile all DCLMs, run

    make

It will produce three executable files: *main-dclm*, *dam*, and *baseline*.

- *baseline* refers to the RNNLM model with contextual information
- *dam* refers to the attentional DCLM
- *main-dclm* includes other four models: context-to-output DCLM, context-to-context DCLM, RNNLM without sentence boundary, and Hierarchical RNNLM.


