import argparse
import pandas as pd

# Argument parser
parser = argparse.ArgumentParser(
    prog='params',
    description='Get model parameter combinations',
)

parser.add_argument('-m', '--mode', action='store', help='mode (enc_dec, simple, grow)')
parser.add_argument('-p', '--params', type=int, help='target params')
parser.add_argument('-min_p', '--min_params', type=int, help='lower bound target params')

args = parser.parse_args()

# Default parameter values
project_dir = "/nlp/projekty/mtlowre/new_tokeval"
scripts_dir = project_dir+"/_scripts"

# Transformer large = 176357376
# Transformer base = 44138496
# Transformer Small = 9416704

LOW_PARAMS = args.min_params if args.min_params is not None else 0
TARGET_PARAMS = args.params if args.params is not None else 0
TOLL = 0.1

pairs = ['eng_wiki-ita_wiki', 'deu-dsb', 'eng-mni', 'eng-akk']

# Model configuration options
voc = 4000
num_enc_layers = [2, 4, 6, 8, 12, 16, 24, 32]
num_dec_layers = [2, 4, 6, 8, 12, 16, 24, 32]
embeds = [256, 512, 1024, 2048, 4096]
#embeds = [512]
ffws = [256, 512, 1024, 2048, 4096]
attn_h = [2]

# Dictionary to store all possible configurations
all_combinations = {}

# Allow independent looping of encoder and decoder layers
for enc_layers in num_enc_layers:
    for dec_layers in num_dec_layers:
        for emb in embeds:
            for ffw in ffws:
                for head in attn_h:
                    # Ensure the embedding dimension is divisible by the number of attention heads
                    while emb / head < 32 and head >= 2:
                        head -= 2

                    if emb % head == 0:
                        # Calculate the number of parameters for each component
                        # https://towardsdatascience.com/how-to-estimate-the-number-of-parameters-in-transformer-models-ca0f57d8dff0
                        # (2*emb*voc)+((4*(emb^2)+2*emb*ffw+9*emb+ffw)+(8*(emb^2)+2*emb*ffw+15*emb+ffw))*num_layers

                        # Vocabulary parameters
                        voc_params = 2 * emb * voc

                        # Encoder parameters
                        enc_attention = 4 * (emb ** 2 + emb)
                        enc_feed_forward = 2 * emb * ffw + emb + ffw
                        enc_layer_norm = 2 * emb
                        encoder_params = enc_layers * (enc_attention + enc_feed_forward + 2 * enc_layer_norm)

                        # Decoder parameters
                        dec_attention = 4 * (emb ** 2 + emb)
                        dec_feed_forward = 2 * emb * ffw + emb + ffw
                        dec_layer_norm = 2 * emb
                        decoder_params = dec_layers * (2 * dec_attention + dec_feed_forward + 3 * dec_layer_norm)

                        # Total parameters
                        total_params = encoder_params + decoder_params + voc_params

                        # Create a unique combination key
                        combination = f"{enc_layers}_{dec_layers}_{emb}_{ffw}_{head}"
                        all_combinations[combination] = total_params

# Filter valid combinations based on parameter range
valid_combinations = {}

# Define parameter boundaries
if args.mode == 'grow':
    lower_bound = LOW_PARAMS * (1 - TOLL)
    upper_bound = TARGET_PARAMS * (1 + TOLL)
else:
    lower_bound = TARGET_PARAMS * (1 - TOLL)
    upper_bound = TARGET_PARAMS * (1 + TOLL)

# Filtering based on boundaries
for key, value in all_combinations.items():
    if lower_bound <= value <= upper_bound:
        valid_combinations[key] = value

print("Number of valid combinations:", len(valid_combinations))
print("Total combinations generated:", len(all_combinations))

### TEMP FIX TO GET RULES MANUALLY
# import csv
# reader = csv.reader(open('/nlp/projekty/mtlowre/new_tokeval/_scripts/_combinations/fix_all_but_one_combinations.csv', 'r'))
# d = {}
# for row in reader:
#    params, enc, dec, emb, ffw, heads = row
#    d[f"{enc}_{dec}_{emb}_{ffw}_{heads}"] = params

# valid_combinations = d

# Writing results to files
with open(f"{scripts_dir}/_rules/{args.params}_train_rules.txt", 'w+') as rules, open(f"{scripts_dir}/_rules/{args.params}_new_trainpairs.tsv", "w+") as experiment:
    for key in valid_combinations.keys():
        print(key)
        keylist = key.split('_')
        rules.write(f"""{key})
                    encoder_layers={keylist[0]}
                    decoder_layers={keylist[1]}
                    encoder_embed_dim={keylist[2]}
                    decoder_embed_dim={keylist[2]}
                    encoder_ffn_embed_dim={keylist[3]}
                    decoder_ffn_embed_dim={keylist[3]}
                    encoder_attention_heads={keylist[4]}
                    decoder_attention_heads={keylist[4]}
                ;;
                """)
        for pair in pairs:
            src, tgt = pair.split('-')
            experiment.write(f"4000,bpe.nof,{src},{tgt},{key},50000,1e-4,5000,0.1,0.1,16000,20,1\n")

# Save all combinations to CSV
df = pd.DataFrame.from_dict(all_combinations, orient='index', columns=['Total_Params'])
df.reset_index(inplace=True)
df[['Enc_Layers', 'Dec_Layers', 'Emb', 'FFW', 'Heads']] = df['index'].str.split('_', expand=True)
df.drop(columns=['index'], inplace=True)
df.to_csv(f"{scripts_dir}/_combinations/{args.mode}_{args.params}_all_combinations.csv", index=False)

# Save valid combinations to CSV
df = pd.DataFrame.from_dict(valid_combinations, orient='index', columns=['Total_Params'])
df.reset_index(inplace=True)
df[['Enc_Layers', 'Dec_Layers', 'Emb', 'FFW', 'Heads']] = df['index'].str.split('_', expand=True)
df.drop(columns=['index'], inplace=True)
df.to_csv(f"{scripts_dir}/_combinations/{args.mode}_{args.params}_combinations.csv", index=False)

