# Matrix Multiplication Decomposition from AlphaEvolve

This repository contains code to generate a symbolic representation of the matrix multiplication decompositions from Google DeepMind's AlphaEvolve paper.

## Background

The AlphaEvolve paper by Google DeepMind presents a groundbreaking approach where an AI system can automatically discover efficient algorithms for a wide range of computational tasks. One notable achievement is finding novel matrix multiplication algorithms that improve on previous best-known results.

In traditional matrix multiplication, multiplying two matrices of sizes n×m and m×p requires n×m×p multiplications. However, more efficient algorithms can reduce this computational complexity, which is especially important for large matrices.

However, this decomposition is presented as a set of three numeric tensors.
If you aren't familiar with tensor decomposition, it isn't obvious how this
translates into a more efficient matrix multiplication algorithm.

## Example

An example of what the `<4,4,4>` decomposition looks like is in
[444_decomposition.pdf](./444_decomposition.pdf). Note that SymPy uses 0-based indexing for the matrix
entries (so, e.g., `A[0,0]` is the first element, `A[3,3]` is the last).

## Available Decompositions

This repository includes:

- `sympy_decomposition.py`: A script that converts raw tensor decomposition data into symbolic mathematical expressions and LaTeX representations
- `444_decomposition.py`: An example of the <4,4,4> matrix multiplication decomposition discovered by AlphaEvolve

The `<4,4,4>` decomposition allows multiplying two 4×4 matrices using just 48 multiplications instead of the naive 64 multiplications.

## How It Works

The core functionality is in `sympy_decomposition.py`, which processes tensor decomposition data (given as matrices U, V, W) and converts them into:

1. Symbolic expressions using SymPy
2. LaTeX representations for easy visualization
3. Verification that the decomposition correctly computes matrix products

The script can also compile the LaTeX output into a PDF document displaying the full algorithm.

## Usage

You can run the script with:

```bash
python sympy_decomposition.py --decomp-file 444_decomposition.py
```


This will print the decomposition to the terminal.

You can also run


```
python sympy_decomposition.py --decomp-file 444_decomposition.py --compile-latex
```

to compile this into a LaTeX PDF.

To verify the decomposition run

```
python sympy_decomposition.py --decomp-file 444_decomposition.py --show-counts
```

This will show the total number of multiplications in the decomposition (for
`<4,4,4>`, it should output 48).

And run

```
python sympy_decomposition.py --decomp-file 444_decomposition.py --verify
```

This will symbolically expand the decomposition to verify that it equals the
naive `AB` multiplication.

### Command Line Options

```
usage: sympy_decomposition.py [-h] [--decomp-file DECOMP_FILE] [--decomp-var DECOMP_VAR] [--n N] [--m M] [--p P] [--rank RANK]
                              [--example {strassen,minimal_float,standard_121}] [--output-file OUTPUT_FILE] [--verify]
                              [--compile-latex] [--latex-compiler LATEX_COMPILER] [--show-counts]
```

| Option | Description |
|--------|-------------|
| `--decomp-file` | Path to Python file with decomposition data |
| `--decomp-var` | Variable name for the (U,V,W) tuple in the file |
| `--n` | Dimension n (rows of A) |
| `--m` | Dimension m (cols of A / rows of B) |
| `--p` | Dimension p (cols of B) |
| `--rank` | Rank of the decomposition |
| `--example` | Run a built-in example from my_decompositions.py (requires the file). Choices: strassen, minimal_float, standard_121 |
| `--output-file`, `-o` | Base name for output .tex file (e.g., 'my_algo'). Default: 'out' |
| `--verify` | Perform symbolic verification of the decomposition |
| `--compile-latex` | Attempt to compile the .tex file to .pdf |
| `--latex-compiler` | LaTeX compiler command. Default: pdflatex |
| `--show-counts` | Show detailed SymPy multiplication counts |

### Examples

**Process a decomposition file and verify it**:
```bash
python sympy_decomposition.py --decomp-file 444_decomposition.py --verify
```

**Generate and compile a LaTeX document**:
```bash
python sympy_decomposition.py --decomp-file 444_decomposition.py --compile-latex
```

**Use a built-in example**:
```bash
python sympy_decomposition.py --example strassen --verify
```

**Specify dimensions manually**:
```bash
python sympy_decomposition.py --decomp-file my_decomposition.py --n 3 --m 3 --p 3 --rank 23
```

## Additional Resources

- [AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) - Blog post from Google DeepMind
- [AlphaEvolve: Training Large Language Models to Evolve Better Algorithms](https://arxiv.org/abs/2402.13035) - The research paper
- [Interactive notebook with mathematical results](https://colab.research.google.com/github/google-deepmind/alphaevolve_results/blob/master/mathematical_results.ipynb) - Contains the decompositions in the format used by this code

The AlphaEvolve paper demonstrated that large language models can be trained to evolve better algorithms through a process similar to genetic programming, leading to the discovery of these efficient matrix multiplication algorithms.
