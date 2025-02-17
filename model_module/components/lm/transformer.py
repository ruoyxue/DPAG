"""Transformer language model."""

import logging
from typing import Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_module.components.utils.lm_interface import LMInterface
from model_module.components.transformer.embedding import PositionalEncoding
from model_module.components.transformer.encoder import Encoder
from model_module.components.transformer.mask import subsequent_mask
from model_module.components.utils.scorer_interface import BatchScorerInterface
from model_module.components.utils.cli_utils import strtobool


class TransformerLM(nn.Module, LMInterface, BatchScorerInterface):
    """Transformer language model."""

    @staticmethod
    def add_arguments(parser):
        """Add arguments to command line argument parser."""
        parser.add_argument(
            "--layer", type=int, default=4, help="Number of hidden layers"
        )
        parser.add_argument(
            "--unit",
            type=int,
            default=1024,
            help="Number of hidden units in feedforward layer",
        )
        parser.add_argument(
            "--att-unit",
            type=int,
            default=256,
            help="Number of hidden units in attention layer",
        )
        parser.add_argument(
            "--embed-unit",
            type=int,
            default=128,
            help="Number of hidden units in embedding layer",
        )
        parser.add_argument(
            "--head", type=int, default=2, help="Number of multi head attention"
        )
        parser.add_argument(
            "--dropout-rate", type=float, default=0.5, help="dropout probability"
        )
        parser.add_argument(
            "--att-dropout-rate",
            type=float,
            default=0.0,
            help="att dropout probability",
        )
        parser.add_argument(
            "--emb-dropout-rate",
            type=float,
            default=0.0,
            help="emb dropout probability",
        )
        parser.add_argument(
            "--tie-weights",
            type=strtobool,
            default=False,
            help="Tie input and output embeddings",
        )
        parser.add_argument(
            "--pos-enc",
            default="sinusoidal",
            choices=["sinusoidal", "none"],
            help="positional encoding",
        )
        return parser

    def __init__(self, n_vocab, args):
        """Initialize class.

        Args:
            n_vocab (int): The size of the vocabulary
            args (argparse.Namespace): configurations. see py:method:`add_arguments`

        """
        nn.Module.__init__(self)

        # NOTE: for a compatibility with less than 0.9.7 version models
        emb_dropout_rate = getattr(args, "emb_dropout_rate", 0.0)
        # NOTE: for a compatibility with less than 0.9.7 version models
        tie_weights = getattr(args, "tie_weights", False)
        # NOTE: for a compatibility with less than 0.9.7 version models
        att_dropout_rate = getattr(args, "att_dropout_rate", 0.0)

        if args.pos_enc == "sinusoidal":
            pos_enc_class = PositionalEncoding
        elif args.pos_enc == "none":

            def pos_enc_class(*args, **kwargs):
                return nn.Sequential()  # indentity

        else:
            raise ValueError(f"unknown pos-enc option: {args.pos_enc}")

        self.embed = nn.Embedding(n_vocab, args.embed_unit)

        if emb_dropout_rate == 0.0:
            self.embed_drop = None
        else:
            self.embed_drop = nn.Dropout(emb_dropout_rate)

        self.encoder = Encoder(
            idim=args.embed_unit,
            attention_dim=args.att_unit,
            attention_heads=args.head,
            linear_units=args.unit,
            num_blocks=args.layer,
            dropout_rate=args.dropout_rate,
            attention_dropout_rate=att_dropout_rate,
            input_layer="linear2",
            pos_enc_class=pos_enc_class,
        )
        self.decoder = nn.Linear(args.att_unit, n_vocab)

        logging.info("Tie weights set to {}".format(tie_weights))
        logging.info("Dropout set to {}".format(args.dropout_rate))
        logging.info("Emb Dropout set to {}".format(emb_dropout_rate))
        logging.info("Att Dropout set to {}".format(att_dropout_rate))

        if tie_weights:
            assert (
                args.att_unit == args.embed_unit
            ), "Tie Weights: True need embedding and final dimensions to match"
            self.decoder.weight = self.embed.weight

        model_dict = args.model_token_dict
        lm_dict = args.lm_token_dict

        model_to_lm = { model_dict[key]: lm_dict.get(key, lm_dict["<unk>"]) for key in model_dict }

        common_keys = set(model_dict.keys()).intersection(lm_dict.keys())
        print(f'The inference model has {len(common_keys)} same tokens embeddings as the language model')

        model_tokens = torch.tensor(list(model_to_lm.keys()), dtype=torch.long)
        lm_tokens = torch.tensor(list(model_to_lm.values()), dtype=torch.long)

        self.model_to_lm_mapping = torch.full((len(model_dict),), lm_dict["<unk>"], dtype=torch.long)
        self.model_to_lm_mapping[model_tokens] = lm_tokens
        self.model_to_lm_mapping = self.model_to_lm_mapping.cuda()

        self.lm_to_model_mapping = torch.full((len(lm_dict),), model_dict["<unk>"], dtype=torch.long).cuda()
        for token, lm_id in lm_dict.items():
            if token in model_dict:
                self.lm_to_model_mapping[lm_id] = model_dict[token]

        self.model_voca_size = len(model_dict)

    def _target_mask(self, ys_in_pad):
        ys_mask = ys_in_pad != 0
        m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
        return ys_mask.unsqueeze(-2) & m

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute LM loss value from buffer sequences.

        Args:
            x (torch.Tensor): Input ids. (batch, len)
            t (torch.Tensor): Target ids. (batch, len)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of
                loss to backward (scalar),
                negative log-likelihood of t: -log p(t) (scalar) and
                the number of elements in x (scalar)

        Notes:
            The last two return values are used
            in perplexity: p(t)^{-n} = exp(-log p(t) / n)

        """
        xm = x != 0

        if self.embed_drop is not None:
            emb = self.embed_drop(self.embed(x))
        else:
            emb = self.embed(x)

        h, _ = self.encoder(emb, self._target_mask(x))
        y = self.decoder(h)
        loss = F.cross_entropy(y.view(-1, y.shape[-1]), t.view(-1), reduction="none")
        mask = xm.to(dtype=loss.dtype)
        logp = loss * mask.view(-1)
        logp = logp.sum()
        count = mask.sum()
        return logp / count, logp, count

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token.

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                torch.float32 scores for next token (n_vocab)
                and next state for ys

        """
        y = y.unsqueeze(0)
        
        if self.embed_drop is not None:
            emb = self.embed_drop(self.embed(y))
        else:
            emb = self.embed(y)

        h, _, cache = self.encoder.forward_one_step(
            emb, self._target_mask(y), cache=state
        )
        h = self.decoder(h[:, -1])
        logp = h.log_softmax(dim=-1).squeeze(0)
        return logp, cache

    def convert_model_token_to_lm_token(self, model_tokens):
        return self.model_to_lm_mapping[model_tokens]

    def convert_lm_prob_to_model_prob(self, lm_probs):
        model_probs = torch.zeros((lm_probs.shape[0], self.model_voca_size), device=lm_probs.device)
        model_probs.index_add_(1, self.lm_to_model_mapping, lm_probs)
        return model_probs

    # batch beam search API (see BatchScorerInterface)
    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch (required).

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor): The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        ys = self.convert_model_token_to_lm_token(ys)

        n_batch = len(ys)
        n_layers = len(self.encoder.encoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        if self.embed_drop is not None:
            emb = self.embed_drop(self.embed(ys))
        else:
            emb = self.embed(ys)

        # batch decoding
        h, _, states = self.encoder.forward_one_step(
            emb, self._target_mask(ys), cache=batch_state
        )

        h = self.decoder(h[:, -1])

        logp = h.log_softmax(dim=-1)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]

        logp = self.convert_lm_prob_to_model_prob(logp)

        return logp, state_list
