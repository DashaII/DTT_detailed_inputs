from transition_amr_parser.parse import AMRParser
from tqdm import tqdm
from sentence_splitter import SentenceSplitter

# Steps to run IBM transition AMR parser
# 	1. Install old libraries from setup.py file
# 	2. Install fairseq==0.10.0 (0.10.2 version doesn’t work)
# 	3. Install torch-scratter like pip install torch-scatter --no-index -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
# 	4. Install parser like pip install git+https://github.com/IBM/transition-amr-parser.git
#
# Note 1 - doesn't work on windows due to ":" delimiter parsing for ensembles (windows path contains ":", like C:\…)
# Note 2 - https://github.com/IBM/transition-amr-parser/tree/master
#
# How to run your script Dasha:
#
# 	1. Using Linux console go to linux venv location: home/python_projects/ini_IBM_AMR
# 	2. Activate environment:  source bin/activate
# 	3. Return back to python file location: /mnt/d/UNI/Thesis/uni_IBM_AMR_3
# 	4. Run the file: python3 amr_parser


def test_func():
    # Download and save a model named AMR3.0 to cache
    parser = AMRParser.from_pretrained('AMR3-structbart-L')
    tokens, positions = parser.tokenize('The queen has tweeted her thanks to people who sent her 90th birthday messages on social media')

    # Use parse_sentence() for single sentences or parse_sentences() for a batch
    annotations, machines = parser.parse_sentence(tokens)

    # Print Penman notation
    # print(annotations)

    # Print Penman notation without JAMR, with ISI
    amr = machines.get_amr()
    # print(amr.get_metadata())
    print(amr.to_penman(jamr=False, isi=True))
    # Plot the graph (requires matplotlib)
    # amr.plot()


def parse_to_file(from_filename: str, to_filename: str):
    from_file = open(from_filename, "r")
    sentences = from_file.readlines()

    # Download and save a model named AMR3.0 to cache
    parser = AMRParser.from_pretrained('AMR3-structbart-L')

    with open(to_filename, "w", encoding='utf-8') as file:
        for i, sent in enumerate(tqdm(sentences)):
            # parse sent into sentences if it's a paragraph
            sents = parse_into_sent(sent)

            for s in sents:
                tokens, positions = parser.tokenize(s)
                # Use parse_sentence() for single sentences or parse_sentences() for a batch
                annotations, machines = parser.parse_sentence(tokens)
                file.write("%d: %s\n" % (i+1, annotations))


def parse_into_sent(sent):
    splitter = SentenceSplitter(language='en')
    return splitter.split(sent)


if __name__ == '__main__':
    parse_to_file("/mnt/d/UNI/Thesis/uni_IBM_AMR_3/webnlg_data_targets_test.txt", "/mnt/d/UNI/Thesis/uni_IBM_AMR_3/amr_parser_results_test.txt")
