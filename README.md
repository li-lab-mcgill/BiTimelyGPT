# BiTimelyGPT

BiTimelyGPT is a bidirectional generative pre-training Transformer designed for representation learning using healthcare time-series data, including continuously monitored biosignals and irregularly sampled time series from longitudinal clinical records.

# Overflow

This figure indicates an overview of BiTimelyGPT architecture.

***Panel a***. BiTimelyGPT architecture is stacked with $L$ bidirectional Retention layers

***Panel b***. Bidirectional Alternating AutoRegressive Modeling (BAAR) framework alternately models left-to-right and right-to-left information across layers, thereby pre-training deep bidirectional contextualized representations.

It pre-trains by a Next-Previous-Token Prediction strategy on the final two layers and fine-tunes the [SOS] token in the final layer for discriminative tasks. 

***Panel c***. Bidirectional Retention Layer alternates between forward and backward Retention across layers.

<img src=https://github.com/li-lab-mcgill/BiTimelyGPT/blob/main/figures/BiTImelyGPT_arch.png width="800">

This figure depicts the process of BiTimelyGPT's Next-Previous-Token Prediction pre-training. 

The input of time-series sequence with $T$ timesteps and $V$ features, $x \in R^{T \times V}$ is tokenized via a convolution-subsampling module. 
The convolution-subsampling tokenizer comprises of two 1-D convolution layers with a kernel size of 3 and stride of 2. 
The resulting sequence of token has dimension $N \times V$ , reducing the sequence length by 1/4, i.e, $N = T /4$.
These tokens are projected onto the input embedding $X \in R^{N \times d}$ by an input projection layer. 
By adding [SOS] and [EOS] tokens, the sequence dimension becomes $(N + 2) \times d$.
Given the L bidirectional generative layers, BiTimelyGPT alternates between forward and backward Retention layers to train bidirectional contextualized representations.
Moreover, the Retention mechanism provides an efficient chunk-wise forward pass that segments an input sequence into multiple chunks. Given a chunk size of
$C$, the $N \times d$ input embedding is reshaped into a $C \times N/C \times d$ tensor. The output projection
layer takes the output embedding with the shape $N \times d$ in the last two layers, which are used to predict the original sequence of tokens with the shape $N \times V$ for Next-Previous-Token
Prediction task.

<img src=https://github.com/li-lab-mcgill/BiTimelyGPT/blob/main/figures/BiTimelyGPT_flow.png>


# Relevant Publications

This published code is referenced from: 

***Ziyang Song***, Qincheng Lu, Mike He Zhu, David Buckeridge, and Yue Li. (2024).
Bidirectional generative pre-training for improving healthcare time-series representation learning.
Machine Learning for HealthCare (MLHC), Proceedings of Machine Learning Research (JMLR Proceedings track).
