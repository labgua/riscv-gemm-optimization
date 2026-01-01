import sys
import numpy as np
import re

def extract_full_matrix(lines, start_idx):
    """Extract the full matrix as a NumPy array from the log lines."""
    matrix = []
    i = start_idx
    while i < len(lines):
        line = lines[i].strip()
        # Stop if we encounter the start of another matrix section
        if re.match(r'^[A-Z]>', line) and 'Print Matrix' in line:
            if matrix:  # Already started reading → this is the next matrix
                break
        # Skip non-numeric lines (headers, separators, etc.)
        if line and not line.startswith(('->', 'EMU', '-', 'Execution', '>', 'Testing', 'BENCHMARK')):
            try:
                row = list(map(float, line.split()))
                if row:
                    matrix.append(row)
            except ValueError:
                pass  # Ignore lines that aren't numeric
        i += 1
    return np.array(matrix, dtype=np.float32), i

def format_tile(matrix, max_rows=10, max_cols=10):
    """Return a list of strings representing the top-left [max_rows x max_cols] tile,
       with ellipses ('...') to indicate truncation."""
    tile_lines = []
    rows, cols = matrix.shape
    show_ellipsis_rows = rows > max_rows
    show_ellipsis_cols = cols > max_cols

    for i in range(min(rows, max_rows)):
        row = matrix[i]
        # Format only the first `max_cols` values
        formatted_vals = ["{:.1f}".format(x) for x in row[:max_cols]]
        line = "\t".join(formatted_vals)
        if show_ellipsis_cols:
            line += "\t..."
        tile_lines.append(line)
    
    if show_ellipsis_rows:
        # Add an ellipsis row to indicate more rows exist
        ellipsis_line = "..." + ("\t..." if show_ellipsis_cols else "")
        tile_lines.append(ellipsis_line)
    
    return tile_lines

def is_matrix_header_line(line):
    """Check if the line marks the start of a matrix section (A, B, or C)."""
    return any(marker in line for marker in ['A> Print Matrix', 'B> Print Matrix', 'C> Print Matrix'])

def main():
    original_lines = sys.stdin.readlines()

    # Locate the starting line indices for matrices A, B, and C
    a_idx = b_idx = c_idx = None
    for i, line in enumerate(original_lines):
        if 'A> Print Matrix' in line:
            a_idx = i
        elif 'B> Print Matrix' in line:
            b_idx = i
        elif 'C> Print Matrix' in line:
            c_idx = i

    # Extract full matrices (needed for correctness verification)
    A_full, _ = extract_full_matrix(original_lines, a_idx) if a_idx is not None else (None, None)
    B_full, _ = extract_full_matrix(original_lines, b_idx) if b_idx is not None else (None, None)
    C_full, _ = extract_full_matrix(original_lines, c_idx) if c_idx is not None else (None, None)

    # Print original log, but replace full matrices with 10x10 tiles
    i = 0
    while i < len(original_lines):
        line = original_lines[i]
        if 'A> Print Matrix' in line:
            sys.stdout.write(line)
            if A_full is not None:
                for tline in format_tile(A_full):
                    sys.stdout.write(tline + "\n")
            # Skip the original matrix lines
            _, next_i = extract_full_matrix(original_lines, i)
            i = next_i
            continue
        elif 'B> Print Matrix' in line:
            sys.stdout.write(line)
            if B_full is not None:
                for tline in format_tile(B_full):
                    sys.stdout.write(tline + "\n")
            _, next_i = extract_full_matrix(original_lines, i)
            i = next_i
            continue
        elif 'C> Print Matrix' in line:
            sys.stdout.write(line)
            if C_full is not None:
                for tline in format_tile(C_full):
                    sys.stdout.write(tline + "\n")
            _, next_i = extract_full_matrix(original_lines, i)
            i = next_i
            continue
        else:
            sys.stdout.write(line)
        i += 1

    # --- Verify matrix multiplication correctness ---
    if a_idx is None or b_idx is None or c_idx is None:
        print("❌ Error: one or more matrices (A, B, or C) not found.")
        sys.exit(1)

    if A_full.shape[1] != B_full.shape[0]:
        print("❌ Error: incompatible dimensions for matrix multiplication.")
        sys.exit(1)

    if C_full.shape != (A_full.shape[0], B_full.shape[1]):
        print("❌ Error: output matrix C does not match the expected shape of A×B.")
        sys.exit(1)

    # Compute the correct result
    C_true = np.dot(A_full, B_full).astype(np.float32)

    # Compare with tolerance for floating-point errors
    if np.allclose(C_full, C_true, rtol=1e-5, atol=1e-5):
        print("✅ Matrix multiplication is CORRECT!")
    else:
        print("❌ Matrix multiplication is INCORRECT.")
        diff = np.abs(C_full - C_true)
        print(f"   Maximum absolute difference: {diff.max():.6f}")

# Example usage
# ./emu.sh build/qemu/rvv_smatmulop_reordered_tiling 128 8 | python check_matmul.py

if __name__ == "__main__":
    main()