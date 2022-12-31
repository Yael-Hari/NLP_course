import torch

from generate_comp_tagged import write_to_tagged_file


def get_sentences(raw_lines, tagged):
    """
    input:
        tagged or not and
        raw lines looks like:
        __________________________________________________________________________
        |     0        |   1    | 2 |     3     | 4 | 5 |      6     | 7 | 8 | 9 |
        __________________________________________________________________________
        token_counter  | token  | - | token POS | - | - | token head | - | - | - |
        __________________________________________________________________________
        columns are splitted by \t
        raws are splitted by \n
    output:
        sentences_words_and_pos - list of couples:  (token, token_POS)
        sentences_deps - list of couples:           tensor(token_counter, token_head)
    """
    # init
    sentences_words_and_pos = []
    sentences_deps = []
    curr_s_words_and_pos = []
    curr_s_deps = []
    # empty lines
    EOF = "\ufeff"
    empty_lines = ["", "\t", EOF]

    for raw_line in raw_lines.split("\n"):
        if raw_line not in empty_lines:
            input_values = raw_line.split("\t")
            curr_s_words_and_pos.append((input_values[1], input_values[3]))
            if tagged:
                curr_s_deps.append(
                    torch.tensor([int(input_values[0]), int(input_values[6])])
                )
        else:
            # got empty line -> finish current sentence
            if len(curr_s_words_and_pos) > 0:
                sentences_words_and_pos.append(curr_s_words_and_pos)
                if tagged:
                    curr_s_deps_tensor = torch.stack(curr_s_deps)
                    sentences_deps.append(curr_s_deps_tensor)
            # init for next sentence
            curr_s_words_and_pos = []
            curr_s_deps = []
    return sentences_words_and_pos, sentences_deps


if __name__ == "__main__":
    train_path = "train.labeled"
    with open(train_path, "r", encoding="utf8") as f:
        raw_lines = f.read()
    tagged = True
    sentences_words_and_pos, sentences_deps = get_sentences(raw_lines, tagged)
    preds = torch.concat(sentences_deps)

    new_train_path = "genreated_labels_train.labeled"
    write_to_tagged_file(preds, new_train_path, train_path)
