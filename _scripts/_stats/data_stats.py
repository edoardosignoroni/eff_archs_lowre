"""
COLLECT STATISTICS ON UNTOKENIZED DATASETS
"""
import argparse

parser = argparse.ArgumentParser(
    prog='run_eval',
    description='evaluates translations from Flores test sets',
    epilog=''
)

parser.add_argument('-f','--file', action='store',
                    help='file to process')

args = parser.parse_args()

def count_sentences(line):
    """
    Counts sentences by splitting line at the .
    """
    line_sents = line.split('.')
    line_sents_no_blank = [ sent.strip() for sent in line_sents if sent.strip() ]    

    return ({"text" : line_sents_no_blank, "count" : len(line_sents_no_blank)})

def count_words(line):
    """
    Count white space words
    """
    line_words = line.split(' ')

    return ({"text" : line_words, "count" : len(line_words)})

def count_chars(line):
    """
    Counts characters
    """
    line_chars = list(line)

    return ({"text" : line_chars, "count" : len(line_chars)})

def get_stats_lists(lines_list):
    """
    Gets counts and lists of sentences, words, and chars
    """
    
    lines = {"text" : lines_list, "count" : len(lines_list)}
    sentences = {"text" : [], "count" : 0}
    words = {"text" : [], "count" : 0}
    chars = {"text" : [], "count" : 0}
    avgs = {"line" : 0, "sentence" : 0, "word" : 0}

    for line in lines_list:

        # count
        line_sentences = count_sentences(line)
        line_words = count_words(line)
        line_chars = count_chars(line)
        # averages
        line_len = line_words["count"]
        if line_sentences["count"] == 0:
            # there clearly is a line with content, but only dots. Counts as 1 sent
            line_sentences["text"] = line
            line_sentences["count"] = 1
        line_avg_sent_len = sum([len(sent.split(' ')) for sent in line_sentences["text"]])/line_sentences["count"]
        line_avg_word_len = sum([len(word) for word in line_words["text"]])/line_words["count"]
        # update stats
        sentences["text"].extend(line_sentences["text"])
        sentences["count"] += line_sentences["count"]
        words["text"].extend(line_words["text"])
        words["count"] += line_words["count"]
        chars["text"].extend(line_chars["text"])
        chars["count"]+=line_chars["count"]
        # accumulte sum of lens, will divide by len(lines_list) before return
        avgs["line"] += line_len
        avgs["sentence"] += line_avg_sent_len
        avgs["word"] += line_avg_word_len
    
    avgs["line"] = avgs["line"]/len(lines_list)
    avgs["sentence"] = avgs["sentence"]/len(lines_list)
    avgs["word"] = avgs["word"]/len(lines_list)
    
    return ({"lines" : lines, "sentences" : sentences, "words" : words, "chars" : chars, "averages" : avgs})

project_dir = f"/nlp/projekty/mtlowre/new_tokeval"
# data_dir = f"{project_dir}/_data"
# dataset_dir = f"{data_dir}/{args.dataset}"

with open(args.file, 'r') as file:
    
    file_lines = file.readlines()

    stats = get_stats_lists(file_lines)


stat_dict = {
                "lines" : stats["lines"]["count"],                                 # num lines separated at new_line
                "sents" : stats["sentences"]["count"],                         # num sentences separated at .
                "words" : stats["words"]["count"],                                # num white space separated words
                "chars" : stats["chars"]["count"],                               # num chars
                "avg_line_length" : round(stats["averages"]["line"], 3),                    # avg len of lines in words
                "avg_sent_length" : round(stats["averages"]["sentence"], 3),                   # avg len of sent in words
                "avg_word_length" : round(stats["averages"]["word"], 3),                     # avg len of words in chars
                "uniq_lines" : len(set(stats["lines"]["text"])),                          # num of unique lines
                "uniq_sents" : len(set(stats["sentences"]["text"])),                           # num of unique sentences
                "uniq_words" : len(set(stats["words"]["text"])),                           # num of unique word forms
                "uniq_chars" : len(set(stats["chars"]["text"]))                           # num of unique chars
}

#print(','.join(list(stat_dict.keys())))
print(f'{stat_dict["lines"]},{stat_dict["sents"]},{stat_dict["words"]},{stat_dict["chars"]},{stat_dict["avg_line_length"]},{stat_dict["avg_sent_length"]},{stat_dict["avg_word_length"]},{stat_dict["uniq_lines"]},{stat_dict["uniq_sents"]},{stat_dict["uniq_words"]},{stat_dict["uniq_chars"]}')