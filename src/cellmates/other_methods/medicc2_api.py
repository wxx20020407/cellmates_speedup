import subprocess


def load_medicc2_tsv_file(medicc2_tsv_path):
    """
    Load MEDICC2 formatted TSV file into a suitable data structure.
    Parameters
    ----------
    medicc2_tsv_path : str
        Path to the MEDICC2 formatted TSV file.

    Returns
    -------
    dict
        A dictionary where keys are cell IDs and values are lists of tuples (chromosome, start, end, cn_a, cn_b).
    """
    data = {}


def run_medicc2(dataset_path, out_dir_path, topology_only=False, num_proc=1):
    # Prepare command to run MEDICC2
    medicc2_command = f'medicc2 {dataset_path} {out_dir_path} -j {num_proc} --no-plot'
    medicc2_command += '--topology-only ' if topology_only else ''
    # Run MEDICC2
    subprocess.run(medicc2_command, shell=True)


def load_medicc2_tree(out_dir):
    return None