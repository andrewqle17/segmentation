# dataset: https://norvig.com/ngrams/
# TODO:
# 1. Determine how to deal with non-lowercase ASCII characters in URLs
# instant segment; currently, they are just being removed
# 2. Implement a better metric for evaluating segmentation similarity;
# currently, validate() just checks for exact matches
# 3. Need to implement logic for set_ngrams that handles cases where n > 2;
# currently, only n=1 and n=2 work.

import instant_segment as seg
import pandas as pd


def main():
    data = pd.read_csv("manual_label_sample_full.csv",
                       skiprows=13, header=None)
    data.iloc[:, 2] = data.iloc[:, 2].astype(dtype="str")
    data.iloc[:, 2] = data.iloc[:, 2].apply(
        lambda x: ''.join([i if 97 <= ord(i) <= 122 else '' for i in x]))
    unigrams = set_ngrams("count_1w.txt")
    bigrams = set_ngrams("count_2w.txt", n=2)
    is_results = pd.DataFrame(
        is_segment(list(data.iloc[:, 2]), unigrams, bigrams)
    )
    validation_data = pd.read_csv("manual_label_sample_full_segmented.csv",
                                  skiprows=1, header=None)
    validation_results = pd.DataFrame(
        validation_data.iloc[:, 2].str.split('|', expand=True))
    print(validation_results)
    print(is_results)
    print("Instant Segment Results: ", validate(is_results, validation_results))


def set_ngrams(file, n=1):
    df = pd.read_csv(file, sep="\t", header=None)
    df.iloc[:, 0] = df.iloc[:, 0].astype(dtype="str")
    if n == 2:
        df.iloc[:, 0] = df.iloc[:, 0].str.split(" ").apply(tuple)
    return iter(list(df.itertuples(index=False, name=None)))

# Uses the Instant Segment algorithm to segment a list of words (https://github.com/djc/instant-segment)


def is_segment(words, unigrams, bigrams):
    segmentations = []
    unigrams = unigrams
    bigrams = bigrams
    segmenter = seg.Segmenter(unigrams, bigrams)
    search = seg.Search()
    for word in words:
        print(word)
        segmenter.segment(word, search)
        segmentations.append([word for word in search])
    return segmentations
# Returns the proportion of exact segmentation matches


def validate(seg_results, validation):
    if len(seg_results) != len(validation):
        print("Length of inputs must match.")
        print(len(seg_results), len(validation))
        return
    total = 0
    correct = 0
    for i in range(len(seg_results)):
        seg_result = seg_results.iloc[i, :].dropna()
        val_result = validation.iloc[i, :].dropna()
        total += 1
        if seg_result.equals(val_result):
            correct += 1
    return correct/total


main()
