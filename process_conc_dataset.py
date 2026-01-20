import pandas as pd

# Load the CSV file
df = pd.read_csv("datasets/solder_ball_conc.csv", header=None)

df.columns = [
    "d_pad",
    "t_pad",
    "d_us",
    "d_rep1",
    "t_ubm",
    "del_d",
    "h_ball",
    "max_conc",
    "min_conc",
]

# Remove every odd row
df_even_rows = df.iloc[1::2].reset_index(drop=True)

# Save the modified DataFrame to a new CSV file
df_even_rows.to_csv("solder_ball_conc.csv")
