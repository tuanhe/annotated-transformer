import torch
from src.layers.utils import subsequent_mask
from src.model.transformer import make_model

#def make_modelHubin(src_vocab_size, tgt_vocab_size, d_model=512, d_ff=2048, n_head=8, 
#                    n_blocks=6, max_len = 5000, dropout=0.1):

def inference_test():
    test_model = make_model(src_vocab = 11, tgt_vocab = 11, N = 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()

if __name__ == "__main__":
    run_tests()