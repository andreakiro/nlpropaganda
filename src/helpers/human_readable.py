def main():
    prop_spans = []
    with open("output1_merged.txt", 'r') as lines:
        for line in lines:
            art_id, start, end = line.strip().split('\t')
            prop_spans.append((int(start),int(end)))

    with open('article000.txt', 'r') as file:
        data = file.read().replace('\n', ' ')

    for span in prop_spans:
        print(data[span[0]-1:span[1]+1])

main()