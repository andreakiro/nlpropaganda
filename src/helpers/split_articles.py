import os

def main(filename):
    spans = {}
    with open(filename, 'r') as lines:
        for line in lines:
            article_id, _, span_start, span_end = line.strip().split("\t")
            if not article_id in spans:
                spans[article_id] = []
            spans[article_id].append((span_start, span_end))
    
    for id in spans:
        s = "article" + str(id) + ".task-flc-tc.labels"
        with open(os.path.join("data", "data-test-tc", "labels", s), 'w') as f:
            for sp in spans[id]:
                ss = str(id) + "\t" + str(sp[0]) + "\t" + str(sp[1]) + "\n"
                f.write(ss)

main(os.path.join("data", "test-task-tc-template.out"))