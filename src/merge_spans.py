import logging
import argparse

import numpy as np

logger = logging.getLogger(__name__)

def main(args):
    spans_file = args.spans_file

    cur_art_id = None
    cur_spans = []
    with open(spans_file, 'r') as lines:
        for line in lines:
            art_id, start, end = line.strip().split('\t')
            start = int(start)
            end = int(end)

            # First article
            if cur_art_id == None:
                cur_art_id = art_id

            if art_id != cur_art_id:
                merged_spans = _merge_spans(cur_spans)
                with open(spans_file[:-4] + "_merged.txt", 'a') as file:
                    for merged_start, merged_end in merged_spans:
                        file.write(f"{cur_art_id}\t{merged_start}\t{merged_end}\n")

                cur_art_id = art_id
                cur_spans = []
            cur_spans.append((start, end))
        
        # Last article
        merged_spans = _merge_spans(cur_spans)
        with open(spans_file[:-4] + "_merged.txt", 'a') as file:
            for merged_start, merged_end in merged_spans:
                file.write(f"{cur_art_id}\t{merged_start}\t{merged_end}\n")

def _merge_spans(spans):
    """
    Merge overlapping spans in the given span tensor.
    :param prop_spans: spans to be merged
    :return: tensor contained only non-overlapping spans
    """
    # For each span in the sorted list, check for intersection with rightmost span analyzed
    merged_spans = [spans[0]]
    for span in spans[1:]:
        # If the current interval does not overlap with the stack top, push it
        if span[0] > merged_spans[-1][1]:
            merged_spans.append(span)
        # If the current interval overlaps with stack top and ending time of current interval is more than that of stack top, 
        # update stack top with the ending time of current interval
        elif span[1] >  merged_spans[-1][1]:
            merged_spans[-1] = (merged_spans[-1][0], span[1])
    return merged_spans

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Merge spans")
    parser.add_argument('-s', '--spans-file', dest='spans_file', required=True, 
                        help="file with predicted spans")

    main(parser.parse_args())