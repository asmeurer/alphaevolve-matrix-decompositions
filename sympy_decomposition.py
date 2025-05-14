import sympy
import numpy as np
import argparse
import importlib.util
import os
import subprocess
import sys

def tensor_decomposition_to_algorithm_data(
    decomposition: tuple[np.ndarray, np.ndarray, np.ndarray],
    n: int, m: int, p: int, rank: int
) -> dict:
    """
    Converts a tensor decomposition for matrix multiplication into symbolic
    expressions and LaTeX strings. Uses ImmutableMatrix.
    """
    factor_matrix_U, factor_matrix_V, factor_matrix_W = decomposition

    A_sym = sympy.MatrixSymbol('A', n, m)
    B_sym = sympy.MatrixSymbol('B', m, p)
    C_sym = sympy.MatrixSymbol('C', n, p) # Used for LaTeX representation of C_ik

    if rank == 0:
        M_symbols = tuple()
    elif rank == 1:
        M_symbols = (sympy.symbols('M_0'),)
    else:
        M_symbols = sympy.symbols(f'M_0:{rank}')

    latex_steps = ["% Intermediate products M_r:"]
    L_exprs_list = []
    S_exprs_list = []
    M_def_exprs_list = [] # Stores L_r * S_r symbolic expressions

    for r_idx in range(rank):
        current_M_sym = M_symbols[r_idx]

        L_r_expr = sympy.S.Zero
        for i_row_A in range(n):
            for j_col_A in range(m):
                coeff_val = factor_matrix_U[i_row_A * m + j_col_A, r_idx]
                sym_coeff = sympy.nsimplify(coeff_val)
                if sym_coeff != 0:
                    L_r_expr += sym_coeff * A_sym[i_row_A, j_col_A]
        # L_r_expr = sympy.expand(L_r_expr)
        L_exprs_list.append(L_r_expr)

        S_r_expr = sympy.S.Zero
        for j_row_B in range(m):
            for k_col_B in range(p):
                coeff_val = factor_matrix_V[j_row_B * p + k_col_B, r_idx]
                sym_coeff = sympy.nsimplify(coeff_val)
                if sym_coeff != 0:
                    S_r_expr += sym_coeff * B_sym[j_row_B, k_col_B]
        # S_r_expr = sympy.expand(S_r_expr)
        S_exprs_list.append(S_r_expr)

        M_r_def_expr = L_r_expr * S_r_expr
        M_def_exprs_list.append(M_r_def_expr)

        if L_r_expr == sympy.S.Zero or S_r_expr == sympy.S.Zero:
            latex_steps.append(f"{sympy.latex(current_M_sym)} &= 0")
            if L_r_expr == sympy.S.Zero and S_r_expr == sympy.S.Zero:
                 latex_steps.append(f"% \\quad (L_{{{r_idx}}} \\text{{ and }} S_{{{r_idx}}} \\text{{ terms were zero}})")
            elif L_r_expr == sympy.S.Zero:
                latex_steps.append(f"% \\quad (L_{{{r_idx}}} \\text{{ term was zero}})")
            else:
                latex_steps.append(f"% \\quad (S_{{{r_idx}}} \\text{{ term was zero}})")
        else:
            latex_steps.append(f"{sympy.latex(current_M_sym)} &= {sympy.latex(M_r_def_expr)}")

    latex_steps.append("% Compute elements of C = A B:")
    C_elements_expr_list = [] # Will store symbolic expressions for each C_ik
    for i_row_C in range(n):
        for k_col_C in range(p):
            C_ik_expr_from_M = sympy.S.Zero
            has_any_term = False
            for r_idx in range(rank):
                coeff_val = factor_matrix_W[k_col_C * n + i_row_C, r_idx]
                sym_coeff = sympy.nsimplify(coeff_val)
                if sym_coeff != 0:
                    has_any_term = True
                    if rank > 0 : # Ensure M_symbols[r_idx] is valid
                        C_ik_expr_from_M += sym_coeff * M_symbols[r_idx]

            C_ik_expr_from_M_expanded = sympy.expand(C_ik_expr_from_M)
            C_elements_expr_list.append(C_ik_expr_from_M_expanded)

            # For LaTeX, use C_sym with indices
            if C_ik_expr_from_M_expanded != sympy.S.Zero or has_any_term:
                 latex_steps.append(f"{sympy.latex(C_sym[i_row_C, k_col_C])} &= {sympy.latex(C_ik_expr_from_M_expanded)}")

    # Create an ImmutableMatrix of the symbolic expressions for C_ik
    C_matrix_expr_from_M = sympy.ImmutableMatrix(n, p, C_elements_expr_list)

    return {
        'latex_steps': latex_steps,
        'L_exprs': L_exprs_list,
        'S_exprs': S_exprs_list,
        'M_def_exprs': M_def_exprs_list,
        'C_matrix_expr': C_matrix_expr_from_M,
        'A_sym': A_sym, 'B_sym': B_sym, 'C_sym': C_sym, 'M_symbols': M_symbols,
        'total_algo_multiplications': rank
    }

def print_latex_document(latex_steps: list[str], title: str = "Matrix Multiplication Algorithm", multiplications: int = None):
    doc = [
        r"\documentclass{article}",
        r"\usepackage{amsmath}",
        r"\usepackage{amsfonts}",
        r"\usepackage{geometry}",
        r"\usepackage{breqn}",  # Package for automatic line breaking of equations
        r"\geometry{a4paper, margin=1in}",
        r"\setlength{\mathindent}{0pt}",  # Reduce indentation in math environments
        r"\setlength{\textwidth}{6.5in}",  # Adjust text width
        r"\setlength{\columnwidth}{6in}",  # Adjust column width for breqn

        # Allow line breaking within math expressions
        r"\allowdisplaybreaks[4]",

        # Enable long equation breaking at operators
        r"\interdisplaylinepenalty=2500",

        r"\title{" + title + (f" ({multiplications} multiplications)" if multiplications is not None else "") +r"}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle"
    ]

    # We'll use dmath* from breqn instead of align*
    # This automatically breaks long equations
    doc.append(r"% Equation steps:")

    for step_latex in latex_steps:
        if step_latex.startswith("%"): # LaTeX comments
            doc.append(f"{step_latex}")
            continue

        # Extract left and right parts of the equation (separated by &=)
        if "&=" in step_latex:
            left_side, right_side = step_latex.split("&=", 1)
            doc.append(r"\begin{dmath*}")
            doc.append(f"{left_side.strip()} = {right_side.strip()}")
            doc.append(r"\end{dmath*}")
        else:
            # For lines without &=, just wrap in dmath*
            doc.append(r"\begin{dmath*}")
            doc.append(step_latex)
            doc.append(r"\end{dmath*}")

    doc.append(r"\end{document}")
    return "\n".join(doc)


def verify_symbolically(n: int, m: int, p: int, algo_data: dict):
    """
    Symbolically verifies if the decomposition correctly reconstructs A*B.
    Uses as_explicit() for matrix products.
    """
    print("\n--- Symbolic Verification ---")
    A_sym = algo_data['A_sym']
    B_sym = algo_data['B_sym']
    M_symbols = algo_data['M_symbols']
    L_exprs = algo_data['L_exprs']
    S_exprs = algo_data['S_exprs']
    C_matrix_expr_from_M = algo_data['C_matrix_expr'] # This is an ImmutableMatrix of C_ik = sum W_coeff * M_k

    rank_val = algo_data['total_algo_multiplications'] # Use the rank from algo_data

    if rank_val == 0:
        M_substitutions = {}
    else:
        # M_def_exprs already contains L_r * S_r
        M_substitutions = {M_symbols[r]: algo_data['M_def_exprs'][r] for r in range(rank_val)}

    # Substitute M_k = L_k * S_k into the C_matrix expressions
    C_algo_substituted = C_matrix_expr_from_M.subs(M_substitutions)

    # Now fully expand this to compare with standard matrix product
    # C_algo_substituted should be an ImmutableMatrix of expressions.
    # We need to expand each element.
    C_algo_fully_expanded_list = [sympy.expand(elem) for elem in C_algo_substituted]
    C_algo_fully_expanded = sympy.ImmutableMatrix(n, p, C_algo_fully_expanded_list)

    # Standard matrix product, converted to an explicit matrix of expressions
    C_standard_matrix_prod = A_sym * B_sym
    C_standard_explicit = C_standard_matrix_prod.as_explicit()
    C_standard_fully_expanded_list = [sympy.expand(elem) for elem in C_standard_explicit]
    C_standard_fully_expanded = sympy.ImmutableMatrix(n, p, C_standard_fully_expanded_list)


    # Ensure comparison is between fully expanded forms
    diff_matrix = sympy.simplify(C_algo_fully_expanded - C_standard_fully_expanded)

    is_correct = (diff_matrix == sympy.zeros(n, p))
    if is_correct:
        print("Symbolic verification successful: Decomposition correctly reconstructs A * B.")
    else:
        print("Symbolic verification FAILED.")
        print("Algorithm's C (fully expanded):")
        sympy.pprint(C_algo_fully_expanded)
        print("Standard A*B (fully expanded):")
        sympy.pprint(C_standard_fully_expanded)
        print("Difference (Algorithm C - Standard A*B) simplified:")
        sympy.pprint(diff_matrix)
    return is_correct

def count_multiplications_in_steps(algo_data: dict):
    """Counts sympy.Mul instances using ImmutableMatrix's .count() method."""
    # ... (same as before, but ensure C_matrix_expr is ImmutableMatrix)
    print("\n--- Multiplication Counts (SymPy Internal) ---")
    print(f"Algorithm 'rank' (primary multiplications M_r = L_r * S_r): {algo_data['total_algo_multiplications']}")

    l_muls = sum(expr.count(sympy.Mul) for expr in algo_data['L_exprs'])
    s_muls = sum(expr.count(sympy.Mul) for expr in algo_data['S_exprs'])

    # M_def_exprs are L_r * S_r.
    # The .count(Mul) on L_r*S_r will count 1 if L_r and S_r are not 0 or 1,
    # plus any Muls within L_r or S_r if they weren't fully expanded or had products.
    # Since L_r and S_r are expanded sums, this count should mostly reflect the outer product.
    m_prod_muls_in_defs = sum(expr.count(sympy.Mul) for expr in algo_data['M_def_exprs'])

    # C_matrix_expr is an ImmutableMatrix. Sum counts over its elements.
    c_assembly_muls = sum(elem.count(sympy.Mul) for elem in algo_data['C_matrix_expr'])

    # These counts are more about coefficient multiplications if not +/-1
    # print(f"  Scalar muls in L_r sums: {l_muls}")
    # print(f"  Scalar muls in S_r sums: {s_muls}")
    # print(f"  Scalar muls in C_ik sums (W_coeff * M_r): {c_assembly_muls}")
    # print(f"  Total SymPy.Mul in M_r definitions (L_r * S_r): {m_prod_muls_in_defs}")
    # The meaningful "multiplication count" for algorithm comparison is algo_data['total_algo_multiplications'] (the rank).


def main():
    parser = argparse.ArgumentParser(description="Convert tensor decomposition to LaTeX algorithm and optionally verify/compile.")
    # ... (Arguments same as before) ...
    parser.add_argument("--decomp-file", type=str, help="Path to Python file with decomposition data.")
    parser.add_argument("--decomp-var", type=str, help="Variable name for the (U,V,W) tuple in the file.")
    parser.add_argument("--n", type=int, help="Dimension n (rows of A).")
    parser.add_argument("--m", type=int, help="Dimension m (cols of A / rows of B).")
    parser.add_argument("--p", type=int, help="Dimension p (cols of B).")
    parser.add_argument("--rank", type=int, help="Rank of the decomposition.")

    parser.add_argument("--example", type=str,
                        choices=["strassen", "minimal_float", "standard_121"],
                        help="Run a built-in example from my_decompositions.py (requires the file).")

    parser.add_argument("--output-file", "-o", type=str, help="Base name for output .tex file (e.g., 'my_algo'). If not specified, uses the same basename as the input file.")
    parser.add_argument("--verify", action='store_true', help="Perform symbolic verification of the decomposition.")
    parser.add_argument("--compile-latex", action='store_true', help="Attempt to compile the .tex file to .pdf.")
    parser.add_argument("--latex-compiler", type=str, default="pdflatex", help="LaTeX compiler command.")
    parser.add_argument("--show-counts", action='store_true', help="Show detailed SymPy multiplication counts.")


    args = parser.parse_args()

    decomposition_data = None
    n_val, m_val, p_val, rank_val = args.n, args.m, args.p, args.rank
    title = "Matrix multiplication decomposition"

    # --- Load Decomposition ---
    if args.example:
        if not os.path.exists("my_decompositions.py"):
            print("Error: my_decompositions.py not found (required for --example).", file=sys.stderr)
            return 1

        spec = importlib.util.spec_from_file_location("my_decompositions_module", "my_decompositions.py")
        decompositions_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(decompositions_module)

        # Make sure your my_decompositions.py defines these structures
        example_map = {
            "strassen": (getattr(decompositions_module, 'decomposition_222', None), # Assuming decomposition_222 is Strassen
                         getattr(decompositions_module, 'strassen_metadata', None)),
            "minimal_float": (getattr(decompositions_module, 'minimal_float_decomposition', None),
                              getattr(decompositions_module, 'minimal_float_metadata', None)),
            "standard_121": (getattr(decompositions_module, 'standard_121_decomposition', None),
                             getattr(decompositions_module, 'standard_121_metadata', None)),
        }
        if args.example not in example_map or example_map[args.example][0] is None or example_map[args.example][1] is None:
            print(f"Error: Example '{args.example}' or its metadata not properly defined in my_decompositions.py.", file=sys.stderr)
            return 1

        decomposition_data, metadata = example_map[args.example]
        n_val, m_val, p_val, rank_val = metadata['n'], metadata['m'], metadata['p'], metadata['rank']
        title = metadata.get('title', f"Algorithm for {args.example}")


    elif args.decomp_file:
        if not os.path.exists(args.decomp_file):
            print(f"Error: File not found: {args.decomp_file}", file=sys.stderr)
            return 1

        module_name = os.path.splitext(os.path.basename(args.decomp_file))[0]
        spec = importlib.util.spec_from_file_location(module_name, args.decomp_file)
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)

        if n_val is None: n_val = getattr(custom_module, 'n', None)
        if m_val is None: m_val = getattr(custom_module, 'm', None)
        if p_val is None: p_val = getattr(custom_module, 'p', None)
        if rank_val is None: rank_val = getattr(custom_module, 'rank', None)

        decomp_var_name_to_load = args.decomp_var
        if not decomp_var_name_to_load:
            if all(isinstance(x, int) and x > 0 for x in [n_val, m_val, p_val]): # n,m,p must be known
                potential_name = f"decomposition_{n_val}{m_val}{p_val}"
                if hasattr(custom_module, potential_name):
                    decomp_var_name_to_load = potential_name
                    print(f"Using inferred decomposition variable: {decomp_var_name_to_load}")

        if not decomp_var_name_to_load:
            print(f"Error: Could not determine decomposition variable name.", file=sys.stderr)
            print("  Specify with --decomp_var or ensure n,m,p are known to infer 'decomposition_NMP'.", file=sys.stderr)
            return 1

        if not hasattr(custom_module, decomp_var_name_to_load):
            print(f"Error: Variable '{decomp_var_name_to_load}' not found in {args.decomp_file}.", file=sys.stderr)
            return 1
        decomposition_data = getattr(custom_module, decomp_var_name_to_load)
        title = fr"$\langle{n_val},{m_val},{p_val}\rangle$ decomposition"

    else:
        parser.print_help()
        print("\nNo decomposition source specified. Use --example or --decomp_file.", file=sys.stderr)
        return 1

    # --- Validate dimensions ---
    if not all(isinstance(x, int) and x > 0 for x in [n_val, m_val, p_val]):
        print("Error: Dimensions n, m, p must be positive integers.", file=sys.stderr)
        print(f"  Got: n={n_val}, m={m_val}, p={p_val}", file=sys.stderr)
        return 1

    if rank_val is None and decomposition_data:
        try:
            rank_val = decomposition_data[0].shape[1]
            print(f"Inferred rank from decomposition: {rank_val}")
        except (TypeError, IndexError, AttributeError):
             print("Error: Could not infer rank from decomposition. Please specify --rank.", file=sys.stderr)
             return 1
    if not isinstance(rank_val, int) or rank_val < 0:
        print(f"Error: Rank must be a non-negative integer. Got: {rank_val}", file=sys.stderr)
        return 1

    # --- Generate Algorithm Data ---
    # Ensure decomposition matrices are NumPy arrays if loaded from simple lists/tuples
    u, v, w = decomposition_data
    u_np = np.array(u)
    v_np = np.array(v)
    w_np = np.array(w)
    decomposition_data_np = (u_np, v_np, w_np)

    algo_data = tensor_decomposition_to_algorithm_data(decomposition_data_np, n_val, m_val, p_val, rank_val)

    # --- Process and output algorithm data ---
    # Generate LaTeX content for compilation if needed
    latex_content = print_latex_document(
        algo_data['latex_steps'],
        title,
        multiplications=algo_data['total_algo_multiplications']
    )

    # Determine output file basename
    if args.output_file:
        output_basename = args.output_file
    elif args.decomp_file:
        # Use the same basename as the input file
        output_basename = os.path.splitext(os.path.basename(args.decomp_file))[0]
    elif args.example:
        # Use the example name
        output_basename = args.example
    else:
        # Fallback for other cases
        output_basename = 'out'

    if args.compile_latex:
        tex_filename = output_basename if output_basename.endswith(".tex") else output_basename + ".tex"
        try:
            with open(tex_filename, "w") as f:
                f.write(latex_content)
            print(f"LaTeX algorithm written to {tex_filename}")

            print(f"Attempting to compile {tex_filename} with {args.latex_compiler}...")
            # For Windows, shell=True might be needed if pdflatex is not directly in PATH in a complex way
            # but generally, direct command is better.
            # Use a temporary directory for compilation to keep main dir clean from aux files.
            # However, for simplicity here, compiling in current dir.
            # On Linux/macOS, os.path.dirname(os.path.abspath(tex_filename)) or "." is fine for cwd

            # Compile twice
            success = False
            for i in range(2):
                compile_process = subprocess.run(
                    [args.latex_compiler, "-interaction=nonstopmode", tex_filename],
                    capture_output=True, text=True,
                    cwd=os.path.dirname(os.path.abspath(tex_filename)) or "." # Run in file's dir
                )
                if compile_process.returncode == 0:
                    success = True
                else:
                    success = False
                    break # Stop if first pass fails

            if success:
                pdf_filename = tex_filename.replace(".tex", ".pdf")
                print(f"Compilation successful. Output: {pdf_filename}")
            else:
                print(f"LaTeX compilation failed (return code {compile_process.returncode}).", file=sys.stderr)
                print("stdout:\n" + compile_process.stdout, file=sys.stderr)
                print("stderr:\n" + compile_process.stderr, file=sys.stderr)
                log_filename = tex_filename.replace(".tex", ".log")
                if os.path.exists(log_filename):
                    print(f"\n--- Contents of {log_filename} (last 2000 chars) ---", file=sys.stderr)
                    try:
                        with open(log_filename, 'r') as log_f:
                            log_content = log_f.read()
                            print(log_content[-2000:], file=sys.stderr)
                    except Exception as log_e:
                        print(f"Error reading log file: {log_e}", file=sys.stderr)
        except IOError as e:
            print(f"Error writing to {tex_filename}: {e}", file=sys.stderr)
            return 1
    else:
        # Use SymPy's pretty printing instead of LaTeX output
        if not args.show_counts and not args.verify:
            print(f"\n--- {title} ({algo_data['total_algo_multiplications']} Multiplications) ---")

            # Print intermediate products
            print("\nIntermediate products M_r:")
            for r_idx in range(algo_data['total_algo_multiplications']):
                if r_idx < len(algo_data['M_symbols']):
                    current_M_sym = algo_data['M_symbols'][r_idx]
                    current_M_def = algo_data['M_def_exprs'][r_idx]

                    # Only print if the expression isn't zero
                    if current_M_def != sympy.S.Zero:
                        print(f"\nM_{r_idx} =")
                        sympy.pprint(current_M_def)

            # Print C matrix elements
            print("\nElements of C = A B:")
            C_matrix = algo_data['C_matrix_expr']

            for i in range(n_val):
                for j in range(p_val):
                    c_elem = C_matrix[i, j]
                    if c_elem != sympy.S.Zero:
                        print(f"\nC[{i},{j}] =")
                        sympy.pprint(c_elem)

            print("\n------------------------------------------------")

    # --- Optional Steps ---
    if args.show_counts:
        count_multiplications_in_steps(algo_data)

    if args.verify:
        verify_symbolically(n_val, m_val, p_val, algo_data)

    return 0

if __name__ == '__main__':
    sys.exit(main())
