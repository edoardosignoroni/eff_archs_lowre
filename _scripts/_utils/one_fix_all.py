import itertools
import pandas as pd

# Lists of values for each parameter
enc_layers = [2, 4, 6, 8, 12, 16, 24, 32]
dec_layers = [2, 4, 6, 8, 12, 16, 24, 32]
emb = [256, 512, 1024, 2048, 4096]
ffw = [256, 512, 1024, 2048, 4096]

# Fixed values for all parameters
default_enc_layers = enc_layers[0]
default_dec_layers = dec_layers[0]
default_emb = emb[0]
default_ffw = ffw[0]

# Function to generate combinations where one parameter changes
def generate_combinations(enc_layers, dec_layers, emb, ffw):
    combinations = []

    # Change `enc_layers` only
    for enc in enc_layers:
        combinations.append((enc, default_dec_layers, default_emb, default_ffw))

    # Change `dec_layers` only
    for dec in dec_layers:
        combinations.append((default_enc_layers, dec, default_emb, default_ffw))

    # Change `emb` only
    for e in emb:
        combinations.append((default_enc_layers, default_dec_layers, e, default_ffw))

    # Change `ffw` only
    for f in ffw:
        combinations.append((default_enc_layers, default_dec_layers, default_emb, f))

    return combinations

# Generate all combinations
combinations = generate_combinations(enc_layers, dec_layers, emb, ffw)

for combination in combinations:
    print('_'.join(list(combination)+"_2")

# Convert to a DataFrame for easy visualization and export
columns = ["Enc_Layers", "Dec_Layers", "Emb", "FFW"]
df = pd.DataFrame(combinations, columns=columns)

# Save to a CSV file
output_file = "one_change_combinations.csv"
df.to_csv(output_file, index=False)

print(f"Generated combinations saved to {output_file}")
