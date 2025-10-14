def main(snakemake):
    import matplotlib.pyplot as plt
    import pandas as pd

    # Load data
    df = pd.read_csv(snakemake.input[0])
    # format:
    # seed, ru_mse, uv_mse, uw_mse, rf, urf, nrf, f1_gt, f1_em, n_cells, n_bins,
    # average over seeds



if __name__=="__main__":
    main(snakemake)