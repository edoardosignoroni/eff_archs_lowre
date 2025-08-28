# Evaluates translations

import argparse
import random
import evaluate
import statsmodels.stats.api as sms
from comet import download_model, load_from_checkpoint

# Argument parser
parser = argparse.ArgumentParser(
    prog='run_eval',
    description='evaluates translations from test sets',
    epilog=''
)

parser.add_argument('-src','--source_lang', action='store',
                    help='source language')
parser.add_argument('-tgt','--target_lang', action='store',
                    help='target language')
parser.add_argument('-voc','--vocab_size', action='store',
                    help='vocab size')
parser.add_argument('-tok','--tokenizer', action='store',
                    help='tokenizer')
parser.add_argument('-m','--model', action='store',
                    help='name of the model')
parser.add_argument('-e','--experiment', action='store',
                    help='name of the experiment')
# parser.add_argument('-b', '--backups', action='store',
#                     help='backup save strategy')
parser.add_argument('-boot', '--bootstrap_eval', action='store_true',
                    help='perform bootstrap evaluation')

args = parser.parse_args()

def compute_sys_bleu(translations, references, use_effective_order=False):
    results = bleu.compute(predictions=translations, references=references, use_effective_order=use_effective_order)
    return results['score']

def compute_sent_bleu(translations, references):
    scores = []
    for sent, ref in zip(translations, [str(reference) for reference in references]):
        scores.append(compute_sys_bleu(translations=[sent], references=[ref], use_effective_order=True))
    return scores

def compute_sys_chrf2(translations, references):
    results = chrf.compute(predictions=translations, references=references, word_order=2, eps_smoothing=True)
    return results['score']

def compute_sent_chrf2(translations,references):
    scores = []
    for sent, ref in zip(translations, [str(reference) for reference in references]):
        scores.append(compute_sys_chrf2(translations=[sent], references=[ref]))
    return scores

def compute_sys_chrf(translations, references):
    results = chrf.compute(predictions=translations, references=references, word_order=0, eps_smoothing=True)
    return results['score']

def compute_sent_chrf(translations,references):
    scores = []
    for sent, ref in zip(translations, [str(reference) for reference in references]):
        scores.append(compute_sys_chrf(translations=[sent], references=[ref]))
    return scores

def load_data_for_comet(sources, preds, refs):
    data= []
    for src,mt,ref in zip(sources, preds, refs):
        data.append({"src": src, "mt": mt, "ref": ref})
    return data

def compute_comet(sources, preds, refs):
    data = load_data_for_comet(sources, preds, refs)
    model_output = comet_model.predict(data, batch_size=8, gpus=1)
    print (model_output)
    return model_output

src = args.source_lang
tgt = args.target_lang
voc = args.vocab_size
tok = args.tokenizer
model = args.model

#load metrics
chrf = evaluate.load('chrf')
#chrf = CHRF()
comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)
bleu=evaluate.load('sacrebleu')

#set directories
project_dir="/nlp/projekty/mtlowre/new_tokeval"
data_dir=f"{project_dir}/_data"
models_dir=f"{project_dir}/_models/{args.experiment}"
outputs_dir=f"{project_dir}/_outputs/{args.experiment}"
logs_dir=f"{project_dir}/_logs/{args.experiment}"

#model name
model_name = f'{src}-{tgt}-{voc}-{tok}-{model}'

#paths
src_path = f"{outputs_dir}/{src}-{tgt}.{src}"
ref_path = f"{outputs_dir}/{src}-{tgt}.{tgt}"
hyp_path = f"{outputs_dir}/{model_name}.{tgt}"
scr_path = f"{outputs_dir}/{model_name}-scr.csv"

#open and preprocess
with open(src_path, 'r') as src_file, open(ref_path, 'r') as ref_file, open(hyp_path, 'r') as hyp_file, open(scr_path, 'w+') as scr_file:

    sources = src_file.readlines()
    references = ref_file.readlines()
    translations = hyp_file.readlines()

    print(f"Loaded {len(sources)} sources...")
    print(f"Loaded {len(references)} references...")
    print(f"Loaded {len(translations)} translations...")

#scoring
    comet_scores = compute_comet(sources, translations, references)
    
    score_dict = {
                    'sys_chrf++': compute_sys_chrf2(translations,references), 
                    'sent_chrf++': compute_sent_chrf2(translations,references),
                    'sys_comet': comet_scores['system_score'],
                    'sent_comet': comet_scores['scores'], 
                    'sys_bleu': compute_sys_bleu(translations, references),
                    'sent_bleu': compute_sent_bleu(translations, references),
                    'sys_chrf': compute_sys_chrf(translations, references),
                    'sent_chrf': compute_sent_chrf(translations,references),
                    }
    
    if args.bootstrap_eval:
        #bootstrap_eval parameters
        iterations = 200
        batch_size = 400

        boot_bleus = []
        boot_chrfs = []
        boot_chrf2s = []
        boot_comets = []

        for i in range(iterations):
            random.seed(i)
            indexes = random.sample(range(0, len(translations)), batch_size)
            
            iter_bleu = sum([score_dict["sent_bleu"][index] for index in indexes])/len(indexes)
            iter_chrf = sum([score_dict["sent_chrf"][index] for index in indexes])/len(indexes)
            iter_chrf2 = sum([score_dict["sent_chrf++"][index] for index in indexes])/len(indexes)
            iter_comet = sum([score_dict["sent_comet"][index] for index in indexes])/len(indexes)

            boot_bleus.append(iter_bleu)
            boot_chrfs.append(iter_chrf)
            boot_chrf2s.append(iter_chrf2)
            boot_comets.append(iter_comet)
        
        avg_bleu = sum(boot_bleus)/len(boot_bleus)
        avg_chrf = sum(boot_chrfs)/len(boot_chrfs)
        avg_chrf2 = sum(boot_chrf2s)/len(boot_chrf2s)
        avg_comet = sum(boot_comets)/len(boot_comets)
        
        conf_bleu= sms.DescrStatsW(boot_bleus).tconfint_mean()
        conf_chrf= sms.DescrStatsW(boot_chrfs).tconfint_mean()
        conf_chrf2= sms.DescrStatsW(boot_chrf2s).tconfint_mean()
        conf_comet= sms.DescrStatsW(boot_comets).tconfint_mean()
        low_conf_bleu = conf_bleu[0]
        high_conf_bleu = conf_bleu[1]
        low_conf_chrf = conf_chrf[0]
        high_conf_chrf = conf_chrf[1]
        low_conf_chrf2 = conf_chrf2[0]
        high_conf_chrf2 = conf_chrf2[1]
        low_conf_comet = conf_comet[0]
        high_conf_comet = conf_comet[1]

        model_l = model.split("_")
        layers = model_l[0]
        embs = model_l[1]
        ffw = model_l[2]
        heads = model_l[3]
                  #ADD ROUND 0.00X
        scr_file.write(f"{src},{tgt},{voc},{tok},{model},{avg_bleu},{low_conf_bleu},{high_conf_bleu},{avg_chrf},{low_conf_chrf},{high_conf_chrf},{avg_chrf2},{low_conf_chrf2},{high_conf_chrf2},{avg_comet},{low_conf_comet},{high_conf_comet},{score_dict['sys_comet']*100},{layers},{embs},{ffw},{heads}\n")
    else:
        scr_file.write(f"{src},{tgt},{voc},{tok},{model},{score_dict['sys_bleu']},{score_dict['sys_chrf++']},{score_dict['sys_chrf']},{score_dict['sys_comet']},{score_dict['sys_comet']*100}\n")

# print(score_dict)



