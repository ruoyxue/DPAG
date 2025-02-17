import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Qwen2Config, Qwen2ForCausalLM

from model_module.components.utils.asr_utils import torch_load, get_model_conf
from model_module.components.utils.batch_beam_search import BatchBeamSearch
from model_module.components.utils.lm_interface import dynamic_import_lm
from model_module.components.utils.scorers.length_bonus import LengthBonus


def get_beam_search_decoder(
    token_list,
    scorers,
    sos,
    eos,
    rnnlm=False,
    rnnlm_conf=None,
    length_bonus=0,
    ctc_weight=0.1,
    lm_weight=0.0,
    beam_size=40,
    device=None
):
    # sos = model.odim - 1
    # eos = model.odim - 1
    # scorers = model.scorers()

    if rnnlm:
        lm_args = get_model_conf(rnnlm, rnnlm_conf)
        lm_model_module = getattr(lm_args, "model_module", "default")
        lm_token_dict = lm_args.char_list_dict
        lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
        model_token_dict = {key: value for value, key in enumerate(token_list)}
        lm_args.model_token_dict = model_token_dict
        lm_args.lm_token_dict = lm_token_dict
        lm = lm_class(len(lm_token_dict), lm_args)
        torch_load(rnnlm, lm)
        lm.eval()
        lm = lm.to(device)
    else:
        lm = None

    scorers["lm"] = lm
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": lm_weight,
        "length_bonus": length_bonus
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )

