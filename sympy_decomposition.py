import sympy
import numpy as np
import argparse
import importlib.util
import os

def _get_sym_coeff(val):
    """Converts a number to a SymPy Rational or Integer."""
    if isinstance(val, (float, np.floating)):
        # Convert float to SymPy Rational for exact representation
        return sympy.Rational(float(val))
    # Convert other types (like int, np.int) to SymPy Integer or appropriate type
    return sympy.S(val)

def tensor_decomposition_to_latex_algorithm(
    decomposition: tuple[np.ndarray, np.ndarray, np.ndarray],
    n: int, m: int, p: int, rank: int
) -> list[str]:
    """
    Converts a tensor decomposition for matrix multiplication into a list of
    LaTeX strings representing the algorithm.

    Args:
      decomposition: Tuple of 3 factor matrices (U, V, W) with float/int entries.
        U (factor_matrix_1) shape (n*m, rank).
        V (factor_matrix_2) shape (m*p, rank).
        W (factor_matrix_3) shape (p*n, rank).
      n: Rows in matrix A.
      m: Columns in A / Rows in B.
      p: Columns in matrix B.
      rank: Rank of the decomposition.

    Returns:
      A list of strings, where each string is a line of LaTeX code
      (either a comment or an equation) for the algorithm.
    """
    factor_matrix_U, factor_matrix_V, factor_matrix_W = decomposition

    # Basic validation
    expected_U_shape = (n * m, rank)
    if factor_matrix_U.shape != expected_U_shape:
        raise ValueError(f'Expected U shape {expected_U_shape}, got {factor_matrix_U.shape}')
    expected_V_shape = (m * p, rank)
    if factor_matrix_V.shape != expected_V_shape:
        raise ValueError(f'Expected V shape {expected_V_shape}, got {factor_matrix_V.shape}')
    expected_W_shape = (p * n, rank)
    if factor_matrix_W.shape != expected_W_shape:
        raise ValueError(f'Expected W shape {expected_W_shape}, got {factor_matrix_W.shape}')

    A_sym = sympy.MatrixSymbol('A', n, m)
    B_sym = sympy.MatrixSymbol('B', m, p)
    C_sym = sympy.MatrixSymbol('C', n, p)

    if rank == 0:
        M_symbols = tuple()
    elif rank == 1:
        M_symbols = (sympy.symbols('M_0'),)
    else:
        M_symbols = sympy.symbols(f'M_0:{rank}')

    latex_steps = ["% Intermediate products M_r:"]

    for r_idx in range(rank):
        current_M_sym = M_symbols[r_idx]

        # L_r = sum_{i,j} U[i*m+j, r_idx] * A_ij
        L_r_expr = sympy.S.Zero
        for i_row_A in range(n):
            for j_col_A in range(m):
                coeff_val = factor_matrix_U[i_row_A * m + j_col_A, r_idx]
                sym_coeff = _get_sym_coeff(coeff_val)
                if sym_coeff != 0:
                    L_r_expr += sym_coeff * A_sym[i_row_A, j_col_A]
        L_r_expr = sympy.expand(L_r_expr) # Expand to show sums clearly

        # S_r = sum_{j,k} V[j*p+k, r_idx] * B_jk
        S_r_expr = sympy.S.Zero
        for j_row_B in range(m):
            for k_col_B in range(p):
                coeff_val = factor_matrix_V[j_row_B * p + k_col_B, r_idx]
                sym_coeff = _get_sym_coeff(coeff_val)
                if sym_coeff != 0:
                    S_r_expr += sym_coeff * B_sym[j_row_B, k_col_B]
        S_r_expr = sympy.expand(S_r_expr) # Expand for clarity

        if L_r_expr == sympy.S.Zero or S_r_expr == sympy.S.Zero:
            latex_steps.append(f"{sympy.latex(current_M_sym)} &= 0")
            if L_r_expr == sympy.S.Zero and S_r_expr == sympy.S.Zero:
                 latex_steps.append(f"% \\quad (L_{r_idx} \\text{{ and }} S_{r_idx} \\text{{ terms were zero}})")
            elif L_r_expr == sympy.S.Zero:
                latex_steps.append(f"% \\quad (L_{r_idx} \\text{{ term was zero}})")
            else: # S_r_expr == sympy.S.Zero
                latex_steps.append(f"% \\quad (S_{r_idx} \\text{{ term was zero}})")
        else:
            # Using expand for L_r and S_r, but not for the product itself unless it's simple
            # sympy.latex can produce very long lines if not careful with complex products
            term_L = f"({sympy.latex(L_r_expr)})"
            term_S = f"({sympy.latex(S_r_expr)})"
            latex_steps.append(f"{sympy.latex(current_M_sym)} &= {term_L} {term_S}")
            

    latex_steps.append("% Compute elements of C = A B:")
    for i_row_C in range(n):
        for k_col_C in range(p):
            C_ik_expr = sympy.S.Zero
            has_any_term = False
            for r_idx in range(rank):
                coeff_val = factor_matrix_W[k_col_C * n + i_row_C, r_idx]
                sym_coeff = _get_sym_coeff(coeff_val)
                if sym_coeff != 0:
                    has_any_term = True
                    C_ik_expr += sym_coeff * M_symbols[r_idx]
            
            C_ik_expr_simplified = sympy.expand(C_ik_expr) # Expand for sum of M terms

            if C_ik_expr_simplified != sympy.S.Zero or has_any_term : # Show C_ik = 0 if all W coeffs are zero but it's part of C
                 latex_steps.append(f"{sympy.latex(C_sym[i_row_C, k_col_C])} &= {sympy.latex(C_ik_expr_simplified)}")
            # else: # If all W coefficients for this C_ik are zero, it's implicitly zero.
            #    pass # Or explicitly add latex_steps.append(f"{sympy.latex(C_sym[i_row_C, k_col_C])} &= 0") if desired

    return latex_steps

def print_latex_document(latex_steps: list[str], title: str = "Matrix Multiplication Algorithm"):
    """Prints a full LaTeX document structure around the algorithm steps."""
    print(r"\documentclass{article}")
    print(r"\usepackage{amsmath}")
    print(r"\usepackage{amsfonts}") % For \mathbb symbols if any (not used here but good practice)
    print(r"\title{" + title + "}")
    print(r"\author{Tensor Decomposition Converter}")
    print(r"\date{\today}")
    print(r"\begin{document}")
    print(r"\maketitle")
    print(r"\begin{align*}")
    for step_latex in latex_steps:
        if step_latex.startswith("%"):
            # LaTeX comments, or could use \text{} if it needs to be in math mode and visible
            print(f"  {step_latex}")
        else:
            print(f"  {step_latex} \\\\")
    print(r"\end{align*}")
    print(r"\end{document}")

def main():
    parser = argparse.ArgumentParser(description="Convert tensor decomposition of matrix multiplication to LaTeX algorithm.")
    parser.add_argument("--decomp_file", type=str, help="Path to Python file containing the decomposition tuple (U, V, W).")
    parser.add_argument("--decomp_var", type=str, help="Name of the variable holding the decomposition tuple in the file.")
    parser.add_argument("--metadata_var", type=str, help="Name of the variable holding a dict with n, m, p, rank in the file (optional).")
    
    # Manual specification if metadata_var is not provided or for override
    parser.add_argument("--n", type=int, help="Dimension n (rows of A).")
    parser.add_argument("--m", type=int, help="Dimension m (cols of A / rows of B).")
    parser.add_argument("--p", type=int, help="Dimension p (cols of B).")
    parser.add_argument("--rank", type=int, help="Rank of the decomposition (optional, can be inferred if not provided via metadata).")

    parser.add_argument("--example", type=str, choices=["strassen_float", "minimal_float", "standard_121"], 
                        help="Run a built-in example from my_decompositions.py (requires the file).")


    args = parser.parse_args()

    decomposition = None
    n_val, m_val, p_val, rank_val = None, None, None, None
    title = "Matrix Multiplication Algorithm"

    if args.example:
        if not os.path.exists("my_decompositions.py"):
            print("Error: my_decompositions.py not found. This file is required for examples.")
            return
        
        spec = importlib.util.spec_from_file_location("my_decompositions_module", "my_decompositions.py")
        decompositions_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(decompositions_module)

        if args.example == "strassen_float":
            decomposition = decompositions_module.strassen_prompt_decomposition_float
            metadata = decompositions_module.strassen_metadata
            title = "Strassen's Algorithm (2x2x2, Float Coeffs)"
        elif args.example == "minimal_float":
            decomposition = decompositions_module.minimal_float_decomposition
            metadata = decompositions_module.minimal_float_metadata
            title = "Minimal 1x1x1 Algorithm (Float Coeffs)"
        elif args.example == "standard_121":
            decomposition = decompositions_module.standard_121_decomposition
            metadata = decompositions_module.standard_121_metadata
            title = "Standard 1x2x1 Algorithm"
        
        n_val, m_val, p_val, rank_val = metadata['n'], metadata['m'], metadata['p'], metadata['rank']

    elif args.decomp_file and args.decomp_var:
        if not os.path.exists(args.decomp_file):
            print(f"Error: File not found: {args.decomp_file}")
            return
        
        module_name = os.path.splitext(os.path.basename(args.decomp_file))[0]
        spec = importlib.util.spec_from_file_location(module_name, args.decomp_file)
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)

        if not hasattr(custom_module, args.decomp_var):
            print(f"Error: Variable '{args.decomp_var}' not found in {args.decomp_file}.")
            return
        decomposition = getattr(custom_module, args.decomp_var)
        title = f"Algorithm from {args.decomp_file} ({args.decomp_var})"

        if args.metadata_var and hasattr(custom_module, args.metadata_var):
            metadata = getattr(custom_module, args.metadata_var)
            n_val, m_val, p_val = metadata.get('n'), metadata.get('m'), metadata.get('p')
            rank_val = metadata.get('rank')
        
        # Override or provide missing n, m, p, rank from command line
        if args.n is not None: n_val = args.n
        if args.m is not None: m_val = args.m
        if args.p is not None: p_val = args.p
        if args.rank is not None: rank_val = args.rank
        
        if not all([isinstance(val, int) for val in [n_val, m_val, p_val]]):
            print("Error: Dimensions n, m, p must be provided either via metadata or command line arguments.")
            return
        if rank_val is None: # Try to infer rank if not provided
             if decomposition and len(decomposition) == 3 and decomposition[0] is not None:
                 rank_val = decomposition[0].shape[1]
             else:
                 print("Error: Rank must be provided or inferable from decomposition.")
                 return
        if not isinstance(rank_val, int):
            print("Error: Rank must be an integer.")
            return

    else:
        parser.print_help()
        print("\nNo decomposition source specified. Use --example or --decomp_file and --decomp_var.")
        return

    if decomposition is None:
        print("Error: Decomposition could not be loaded.")
        return

    try:
        latex_steps = tensor_decomposition_to_latex_algorithm(decomposition, n_val, m_val, p_val, rank_val)
        print_latex_document(latex_steps, title)
    except ValueError as e:
        print(f"Error during algorithm generation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    # Example of how to use the original verification function (not part of this script's main task)
    # from the prompt, if you had it in a separate file or defined here.
    # def verify_tensor_decomposition(...): ...
    # verify_tensor_decomposition(decomp_s, n_s, m_s, p_s, rank_s)

    main()