import sentencepiece
import torch


def get_tokenizer(mode, dataset_cfg):
    name = dataset_cfg.tokenizer.name
    if name == 'spm':
        return SPMTokenizer(mode, dataset_cfg)
    else:
        raise ValueError('Invalid choice of tokenizer')


class SPMTokenizer:
    """Mapping Dictionary Class for SentencePiece tokenization."""

    def __init__(self, mode, dataset_cfg):
        sp_model_path = dataset_cfg.tokenizer.model.replace('{mode}', mode)
        dict_path = dataset_cfg.tokenizer.dict.replace('{mode}', mode)

        self.spm = sentencepiece.SentencePieceProcessor(model_file=sp_model_path)
        self.mode = mode

        units = open(dict_path, encoding='utf8').read().splitlines()
        self.hashmap = {unit.split()[0]: unit.split()[-1] for unit in units}
        self.token_list = ["<blank>"] + list(self.hashmap.keys()) + ["<eos>"]
        self.ignore_id = -1

    def word2token(self, text):
        tokens = self.spm.EncodeAsPieces(text)
        token_ids = [self.hashmap.get(token, self.hashmap["<unk>"]) for token in tokens]
        return torch.tensor(list(map(int, token_ids)))

    def token2word(self, token_ids):
        valid_ids = []
        for token in token_ids:
            if token != -1:
                valid_ids.append(token)
        token_ids = valid_ids
        text = self._ids_to_str(token_ids, self.token_list)
        text = text.replace("\u2581", " ").strip()
        text = text.replace(" #", "")
        return text

    def _ids_to_str(self, token_ids, char_list):
        token_as_list = [char_list[idx] for idx in token_ids]
        return "".join(token_as_list).replace("<space>", " ").replace("▁", " ")\
                .replace("<eos>", "").replace(' #', '').strip()
