# dataset: https://norvig.com/ngrams/
# TODO:
# 1. Determine how to deal with non-lowercase ASCII characters in URLs
# instant segment; currently, they are just being removed
# 2. Implement a better metric for evaluating segmentation similarity;
# currently, validate() just checks for exact matches
# 3. Need to implement logic for set_ngrams that handles cases where n > 2;
# currently, only n=1 and n=2 work.

import instant_segment as seg
import numpy as np
import pandas as pd


def main():
    data = pd.read_csv("manual_label_sample_combined.csv",
                       skiprows=502, header=None)
    unigrams = set_ngrams("count_1w.txt")
    bigrams = set_ngrams("count_2w.txt", n=2)
    is_results_full = pd.DataFrame(
        is_segment(data.iloc[:, 2], unigrams, bigrams)
    )
    for i in range(len(is_results_full)):
        is_results_full.iloc[i, :] = combine_segments(
            is_results_full.iloc[i, :])
    is_results_full = is_results_full.iloc[:, 0]

    data = pd.concat([data, is_results_full], axis=1, ignore_index=True)
    high_confidence = data.loc[data.iloc[:, 2] == data.iloc[:, 3]]
    low_confidence = data.loc[data.iloc[:, 2] != data.iloc[:, 3]]
    # high_confidence.to_csv("low_confidence_segmentations.csv")

    segmentation_methods = ["Manual1", "Manual2",
                            "ChatGPT_4o", "Instant_Segment"]
    high_confidence_sim_mat = pd.DataFrame(
        0, segmentation_methods, segmentation_methods, dtype=float)
    for i in range(len(high_confidence_sim_mat)):
        for j in range(len(high_confidence_sim_mat)):
            scores = []
            for k in range(len(high_confidence)):
                scores.append(calc_jaccard(high_confidence.iloc[k, i+2],
                                           high_confidence.iloc[k, j+2]))
            high_confidence_sim_mat.iloc[i, j] = np.mean(
                pd.Series(scores).dropna())
    print(high_confidence_sim_mat)

    full_sim_mat = pd.DataFrame(
        0, segmentation_methods, segmentation_methods, dtype=float)
    for i in range(len(full_sim_mat)):
        for j in range(len(full_sim_mat)):
            scores = []
            for k in range(len(data)):
                scores.append(calc_jaccard(data.iloc[k, i+2],
                                           data.iloc[k, j+2]))
            full_sim_mat.iloc[i, j] = np.mean(pd.Series(scores).dropna())
    print(full_sim_mat)

    # validation_data = pd.read_csv("manual_label_sample_combined.csv",
    #                               skiprows=502, header=None)
    # validation_results1 = pd.DataFrame(
    #     validation_data.iloc[:, 2].str.split('|', expand=True))
    # validation_results2 = pd.DataFrame(
    #     validation_data.iloc[:, 3].str.split('|', expand=True))
    # pipe_distances = data.apply(
    #     lambda row: calc_pipe_distance(row[2], row[3]), axis=1)
    # pipe_distances = pipe_distances.drop(
    #     labels=pipe_distances[pipe_distances < 0].index)
    # print(pipe_distances)
    # print(np.average(pipe_distances))
    # print("Full Instant Segment Results: ", validate(
    #     is_results_full, validation_results1))
    # print("Chunk B Instant Segment Results: ", validate(
    #     is_results_full.iloc[0:499], validation_results2.iloc[0:499]
    # ))
    # print("Chunk C Instant Segment Results: ", validate(
    #     is_results_full.iloc[500:998], validation_results2.iloc[500:998]
    # ))
    # print("Chunk D Instant Segment Results: ", validate(
    #     is_results_full.iloc[999:], validation_results2.iloc[999:]
    # ))
    # print("Chunk B Similarity: ", validate(
    #     validation_results1[0:499], validation_results2.iloc[0:499]
    # ))
    # print("Chunk C Similarity: ", validate(
    #     validation_results1[500:998], validation_results2.iloc[500:998]
    # ))
    # print("Chunk D Similarity: ", validate(
    #     validation_results1[999:], validation_results2.iloc[999:]
    # ))


def set_ngrams(file, n=1):
    df = pd.read_csv(file, sep="\t", header=None)
    df.iloc[:, 0] = df.iloc[:, 0].astype(dtype="str")
    if n == 2:
        df.iloc[:, 0] = df.iloc[:, 0].str.split(" ").apply(tuple)
    return iter(list(df.itertuples(index=False, name=None)))

# Uses the Instant Segment algorithm to segment a list of words (https://github.com/djc/instant-segment)


def is_segment(words, unigrams, bigrams):
    words = words.astype(dtype="str")
    words = words.apply(
        lambda x: ''.join([i if 97 <= ord(i) <= 122 else '' for i in x]))
    words = list(words)
    segmentations = []
    unigrams = unigrams
    bigrams = bigrams
    segmenter = seg.Segmenter(unigrams, bigrams)
    search = seg.Search()
    for word in words:
        segmenter.segment(word, search)
        segmentations.append([word for word in search])
    return segmentations
# Returns the proportion of exact segmentation matches

# Takes a list of segments and combines it into a single segmentation in the form "seg1|seg2|..."


def combine_segments(segments):
    combined = segments[0]
    if combined == None:
        return None
    segments = segments.dropna()
    for i in range(len(segments)-1):
        combined += "|"+segments[i+1]
    return combined

# Takes a segmentationin the form "seg1|seg2|..." and splits it into a Series


def split_segmentation(segmentation):
    segments = segmentation.split('|')
    return segments


def validate(seg_results, validation):
    if len(seg_results) != len(validation):
        print("Length of inputs must match.")
        print(len(seg_results), len(validation))
        return
    total = 0
    correct = 0
    for i in range(len(seg_results)):
        seg_result = seg_results.iloc[i, :]
        val_result = validation.iloc[i, :]
        total += 1
        if seg_result == val_result:
            correct += 1
    return correct/total


def calc_pipe_distance(word1, word2):
    word1_pipes = np.array(
        [pos for pos, char in enumerate(word1) if char == "|"])
    word2_pipes = np.array(
        [pos for pos, char in enumerate(word2) if char == "|"])
    if len(word1_pipes) != len(word2_pipes) or len(word1_pipes) == 0:
        return -1
    distance_array = np.absolute(word1_pipes - word2_pipes)
    return sum(distance_array)


def calc_jaccard(segmentation1, segmentation2):
    if segmentation1 == None or segmentation2 == None:
        return None
    segmentation1 = set(split_segmentation(segmentation1))
    segmentation2 = set(split_segmentation(segmentation2))
    intersection = segmentation1.intersection(segmentation2)
    union = segmentation1.union(segmentation2)
    return len(intersection)/len(union)


main()
