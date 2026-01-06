import os
import numpy as np

def load_x(subject_name: str) -> np.ndarray:
    """
    Load x (diff of rpeaks / fs) from a pre-saved .npy file.

    Parameters
    ----------
    subject_name : str
        Base name of the subject (without extension).
        e.g. load_x("S1") expects "S1.npy" in the project root directory.

    Returns
    -------
    x : np.ndarray
        The precomputed x array.
    """
    # Get path to this file's directory (src)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    # Get project root (parent of src)
    project_dir = os.path.dirname(src_dir)
    
    # Build path to the .npy file in the data directory
    file_name = subject_name + ".npy"
    data_path = os.path.abspath(os.path.join(project_dir, "data", file_name))

    # Load the x array directly
    x = np.load(data_path)

    return x

# Example usage:
# x = load_x("S1")
# print(x.shape, x[:10])