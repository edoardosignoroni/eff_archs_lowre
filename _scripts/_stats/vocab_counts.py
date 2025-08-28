import os

pairs = ["eng.25000-deu.25000","eng.50000-deu.50000","eng.75000-deu.75000",
         "eng.100000-deu.100000","eng.150000-deu.150000","eng.200000-deu.200000",
         "eng.275000-deu.275000","eng.350000-deu.350000","eng.425000-deu.425000","eng.500000-deu.500000"]
vocs = [1000, 2000, 4000, 8000, 16000, 32000]

results_dir = "/nlp/projekty/mtlowre/new_tokeval/_results"
data_dir = "/nlp/projekty/mtlowre/new_tokeval/_data"

for pair in pairs:
    langs = pair.split('-')
    src = langs[0]
    tgt = langs[1]
    for voc in vocs:
        for lang in langs:
            # dict_filename = os.path.join(data_dir, f'{pair}/{voc}-bpe.nof/{lang}.vocab')
            dict_filename = os.path.join(data_dir, f'{pair}/{voc}-bpe.nof/databin/dict.{lang}.txt')
            train_filename = os.path.join(data_dir, f'{pair}/{voc}-bpe.nof/train.toks.{lang}')
            output_filename = os.path.join(results_dir, f'{pair}-{voc}-bpe.nof-vocab_counts.csv')

            train = open(train_filename, encoding='utf-8')

            train_lines = train.readlines()

            train_words = set()

            for line in train_lines:
                parts = line.split("▁")
                for part in parts:
                    if part.count(" ")==1 and part.endswith(" "):
                        train_words.add(part.strip(" "))

            f = open(dict_filename, encoding="utf-8")
            lines = f.readlines()

            tot_lines = len(lines)+1

            i=0
            count_chars = 0
            count_subwords = 0
            count_words = 0

            for line in lines:
                parts = line.split(" ")
                line = parts[0]
                if "madeupword0000" in line:
                    continue
                
                if len(line) == 1:
                    count_chars = count_chars + 1
                    continue
                
                if len(line) == 2:
                    if line.startswith("▁"):
                        count_chars = count_chars + 1             
                    else:
                        count_subwords = count_subwords + 1
                    continue


                if line.startswith("▁"):
                    line = line.strip("▁")
                    if line in train_words:
                        count_words = count_words + 1
                    else:
                        count_subwords = count_subwords + 1
                    continue
                else:
                    count_subwords = count_subwords + 1
                    continue

            Voc_Size = voc
            Real_VocSize = tot_lines
            Prc_Char = count_chars/tot_lines*100
            Num_Char = count_chars
            Prc_Subword = count_subwords/tot_lines*100
            Num_Subword = count_subwords
            Prc_Word = count_words/tot_lines*100
            Num_Word = count_words

            with open(output_filename, "w+") as out:
                out.write(f"{src},{tgt},{Voc_Size},{Real_VocSize},{Prc_Char},{Num_Char},{Prc_Subword},{Num_Subword},{Prc_Word},{Num_Word}\n")

            # print(count_chars)
            # print(count_subwords)
            # print(count_words)

# dir = os.path.dirname(__file__)
# dict_filename = os.path.join(dir, 'dict.eng.txt')
# train_filename = os.path.join(dir, 'train.toks.eng')


