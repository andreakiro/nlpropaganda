import os
import pickle

def main():
    labels = {}
    with os.scandir("labels-tc/") as articles:
        for a in articles:
            if not a.name.endswith(".labels"):
                continue
            content: str = None
            with open(a, 'r') as lines:
                for line in lines:
                    article_id, category, span_start, span_end = line.strip().split()
                    category = category.split(",")
                    if article_id not in labels:
                        labels[article_id] = [[int(span_start), int(span_end), category]]
                    else:
                        labels[article_id].append([int(span_start), int(span_end), category])
    
    merged = {}
    for key, value in labels.items():
        merged[key] = split_spans(list(value))

    # for (id, info), (_, merged ) in zip(labels.items(), merged.items()):
    #     print("\n\n")
    #     print("Article ", id)
    #     for elem in info:
    #         print(elem[0], elem[1], elem[2])
    #     print("-- merge --")
    #     for elem in merged:
    #         print(elem[0], elem[1], elem[2])
    
    print(merged)

    a_file = open("gold_split.pkl", "wb")
    pickle.dump(merged, a_file)
    a_file.close()

def split_spans(spans):
    splitted = []
    for s in spans:
        if len(s[2]) > 1:
            for cat in s[2]:
                splitted.append([s[0], s[1], cat])
        else:
            splitted.append([s[0], s[1], s[2][0]])
    return splitted

def merge_spans(spans):
    spans.sort(key=lambda y: y[0])
    merged = []
    for s in spans:
        if merged and s[0] < merged[-1][1]:
            merged[-1][1] = s[1]
            for cat in s[2]:
                if cat not in merged[-1][2]:
                    merged[-1][2].append(cat)
        else:
            merged.append(s)            
    return merged

def resave_pred():
    labels = {}
    with open("output1.txt", 'r') as lines:
        for line in lines:
            article_id, span_start, span_end = line.strip().split()
            if article_id not in labels:
                labels[article_id] = [[int(span_start), int(span_end)]]
            else:
                labels[article_id].append([int(span_start), int(span_end)])
    a_file = open("pred_original.pkl", "wb")
    pickle.dump(labels, a_file)
    a_file.close()

resave_pred()