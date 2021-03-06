r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
     # Dan: this got me to ~50%
    # hypers['batch_size']=512
    # hypers['seq_len']=32
    # hypers['h_dim']=128
    # hypers['n_layers']=3
    # hypers['dropout']=0.4
    # hypers['learn_rate']=0.01
    # hypers['lr_sched_factor']=0.1
    # hypers['lr_sched_patience']=10
    
    # After that, lowered lr to 0.005
    # will increase seq len to 64, h_dim to 256, n_layers to 4

    hypers['batch_size']=512
    hypers['seq_len']=32
    hypers['h_dim']=128
    hypers['n_layers']=3
    hypers['dropout']=0.4
    hypers['learn_rate']=0.01
    hypers['lr_sched_factor']=0.1
    hypers['lr_sched_patience']=10
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None

def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    # hypers['batch_size'] = 512
    # hypers['h_dim'] = 128
    # hypers['z_dim'] = 32
    # hypers['x_sigma2'] = 0.1
    # hypers['learn_rate'] = 0.0001
    # hypers['betas'] = (0.9, 0.999)

    hypers['batch_size'] = 64
    hypers['h_dim'] = 2048
    hypers['z_dim'] = 40
    hypers['x_sigma2'] = 0.003
    hypers['learn_rate'] = 0.0002
    hypers['betas'] = (0.7, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None

def part3_gan_hyperparams():
    hypers = dict(
        batch_size=64,
        z_dim=50,
        data_label=1,
        label_noise=0.2,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0002,
            betas=(0.3, 0.7),
            # momentum=0.9,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0002,
            betas=(0.5, 0.99),
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
