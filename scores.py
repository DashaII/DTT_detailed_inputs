#!/usr/bin/env python3

import sacrebleu
from bleurt import score
import configs
import datasets
import warnings

warnings.filterwarnings("ignore")


def read_from_file(filename):
    with open(filename, mode="r", encoding="utf-8") as file:
        result = [line.strip() for line in file]
    return result


def get_reference(data):
    return [item['references'] for item in data]


if __name__ == '__main__':
    test_dataset = datasets.load_dataset(path='GEM/web_nlg', name=configs.LANG, split=configs.TEST_SPLIT)
    references = get_reference(test_dataset)

    results_files = ['results/results_generated_10.txt', 'results/results_generated_12.txt',
                     'results/results_generated_14.txt', 'results/results_generated_14_1.txt',
                     'results/results_generated_14_3.txt', 'results/results_generated_14_5.txt',
                     'results/results_generated_14_7.txt', 'results/results_generated_14_13.txt']

    results = []
    for file_name in results_files:
        results.append(read_from_file(file_name))

    # BLEU and CHRF
    bleu = sacrebleu.metrics.BLEU()
    chrf = sacrebleu.metrics.CHRF()

    for i, result in enumerate(results):
        bleu_score1 = bleu.corpus_score(result, references)
        bleu_score2 = sacrebleu.raw_corpus_bleu(result, references, 0.0).score
        chrf_score = chrf.corpus_score(result, references)

        print("\n", results_files[i], ">")
        print("BLEU score v1", round(bleu_score1.score, 2))
        print("BLEU score v2", round(bleu_score2, 2))
        print("CHRF score", chrf_score)

    # BLEURT
    # cand = ['Municipal Coaracy da Mata Fonseca is located in Arapiraca and is the home ground of the Agremiacao Sportiva Arapiraquense. The club play in the Campeonato Brasileiro Série C league in Brazil and the nickname of the player is Alvinegro.']
    # ref1 = ['Estádio Municipal Coaracy da Mata Fonseca is the name of the ground of Agremiação Sportiva Arapiraquense in Arapiraca. Agremiação Sportiva Arapiraquense, nicknamed "Alvinegro", lay in the Campeonato Brasileiro Série C league from Brazil.']
    # ref2 = ['Estádio Municipal Coaracy da Mata Fonseca is the name of the ground of Agremiação Sportiva Arapiraquense in Arapiraca. Alvinegro, the nickname of Agremiação Sportiva Arapiraquense, play in the Campeonato Brasileiro Série C league from Brazil.']
    #
    #
    # checkpoint = "bleurt/BLEURT-20"
    # scorer = score.BleurtScorer(checkpoint)
    # score1 = scorer.score(references=ref1, candidates=cand)
    # score2 = scorer.score(references=ref2, candidates=cand)
    #
    # print(score1, score2)
