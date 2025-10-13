def main(snakemake):
    import matplotlib.pyplot as plt
    import pandas as pd

    # Load data
    df = pd.read_csv(snakemake.input[0])

    # Plot data
    # placeholder plot
    plt.figure(figsize=(10,6))
    plt.plot(df['metric1'], df['metric2'], marker='o')

    # Save plot
    plt.savefig(snakemake.output[0])

if __name__=="__main__":
    main(snakemake)