

def read_dataset(dataset_file):
    with open(dataset_file, 'r') as file_handle:
        current_sentance_num = '1.0'
        current_sentance = []
        for raw_line in file_handle:

            line_contents = raw_line.split()
            sentance_num, word, pos, tag = line_contents[1:]
            if (word, pos, tag) == ('Word', 'POS', 'Tag'):
                continue
            current_sentance.append((word, pos, tag))
            if sentance_num != current_sentance_num:
                current_sentance_num = sentance_num
                full_sentance = current_sentance
                current_sentance = []
                yield [((w, t), iob) for w, t, iob in full_sentance]


if __name__ == '__main__':
    data_file = "/home/eoshea/sflintro/data/dataset_22nov17.txt" 
    data = list(read_dataset(data_file))
    print(data[0])
    print(sum([len(i) for i in data]))