import torch

from model_module.components.visual_frontend import get_visual_frontend
from model_module.components.transformer.apply_lm_scorer import apply_lm
from model_module.components.transformer.beam_search_decoder import get_beam_search_decoder
from model_module.components.utils.asr_utils import parse_hypothesis
from model_module.components.utils.ctc import CTC
from model_module.components.utils.scorers.ctc import CTCPrefixScorer
from model_module.components.lm import LMScorer
from model_module.components.transformer.decoder import Decoder
from data_module.transforms import RandomHorizontalFlip

from .symmetric_face_embedding import SymmetricFaceEmbedding
from .encoder import DPAEncoder


class DPAG(torch.nn.Module):
    def __init__(self, in_ch, token_list, model_cfg, decode_cfg, transform, ignore_id=-1):
        torch.nn.Module.__init__(self)
        self.visual_frontend = get_visual_frontend(in_ch, model_cfg.visual_frontend)
        self.odim = len(token_list)
        self.token_list = token_list
        self.model_cfg = model_cfg
        self.sos = self.odim - 1
        self.eos = self.odim - 1
        self.ignore_id = ignore_id
        self.transform = transform
        self.decode_cfg = decode_cfg
        
        tcfg = model_cfg.chinese_extractor

        self.sfe = SymmetricFaceEmbedding(
            image_size=model_cfg.visual_frontend.image_size,
            feat_size=model_cfg.visual_frontend.feat_size,
            feat_ch=model_cfg.visual_frontend.feat_channel,
            ratio=model_cfg.visual_frontend.ratio,
            dropout_rate=tcfg.dropout_rate
        )

        self.encoder = DPAEncoder(
            idim=model_cfg.visual_frontend.feat_channel,
            node_dim=tcfg.node_dim,
            edge_dim=tcfg.edge_dim,
            heads=tcfg.eheads,
            layers=tcfg.elayers,
            dropout_rate=tcfg.dropout_rate,
            attn_drop_rate=tcfg.attn_drop_rate,
            pga_node_dim=tcfg.pga_node_dim,
            pgru_scale=tcfg.pgru_scale,
            macaron_style=tcfg.macaron_style,
            tv_local_width=tcfg.tv_local_width
        )

        self.decoder = Decoder(
            odim=self.odim,
            attention_dim=tcfg.pga_node_dim,
            attention_heads=tcfg.dheads,
            linear_units=tcfg.dlinear_units,
            num_blocks=tcfg.dlayers,
            dropout_rate=tcfg.dropout_rate,
            positional_dropout_rate=tcfg.dropout_rate,
            self_attention_dropout_rate=tcfg.attn_drop_rate,
            src_attention_dropout_rate=tcfg.attn_drop_rate
        )

        self.mtlalpha = tcfg.mtlalpha
        assert 0 <= self.mtlalpha <= 1
        if self.mtlalpha > 0:
            self.ctc = CTC(
                odim=self.odim, 
                eprojs=tcfg.pga_node_dim, 
                dropout_rate=tcfg.dropout_rate,
                ctc_type=tcfg.ctc_type, 
                reduce=True
            )

        self.lm_scorer = None
        self.hflip = RandomHorizontalFlip(1)

    def test_step(self, sample):
        video, landmark = sample['video'],  sample['landmark']

        video_length = video.shape[0]
        
        video = video.unsqueeze(0)
        landmark = landmark.unsqueeze(0)

        video_feat, _ = self.visual_frontend(video, flatten=False)  # (B, T, d_model)
        node_feat = self.sfe(video_feat, landmark)
        frame_feat = self.encoder(node_feat, None)
        
        scorers = dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))
        
        if not hasattr(self, 'beam_search'):
            self.beam_search = get_beam_search_decoder(
                token_list=self.token_list,
                scorers=scorers,
                sos=self.sos,
                eos=self.eos,
                ctc_weight=self.model_cfg.chinese_extractor.test_ctc_weight,
                beam_size=self.decode_cfg.test_beam_size,
                rnnlm=self.decode_cfg.rnn_lm,
                rnnlm_conf=self.decode_cfg.rnn_lm_conf,
                length_bonus=self.decode_cfg.length_bonus,
                lm_weight=self.decode_cfg.lm_weight,
                device=video.device
            )
        
        nbest_hyps = self.beam_search(frame_feat.squeeze(0))
        
        if self.decode_cfg.tta:
            tta_dict = self.hflip({
                "video": video,
                "landmark": landmark
            })
            flip_video, flip_landmark = tta_dict['video'], tta_dict['landmark']

            flip_video_feat, _ = self.visual_frontend(flip_video, flatten=False)  # (B, T, d_model)
            flip_node_feat = self.sfe(flip_video_feat, flip_landmark)
            flip_frame_feat = self.encoder(flip_node_feat, None)
            flip_nbest_hyps = self.beam_search(flip_frame_feat.squeeze(0))
            nbest_hyps.extend(flip_nbest_hyps)
            nbest_hyps.sort(key=lambda hyp: hyp.score, reverse=True)
        
        if self.decode_cfg.llm_path:
            if self.lm_scorer is None:
                self.lm_scorer = LMScorer(
                    llm_path=self.decode_cfg.llm_path,
                    lm_largest_texts_length=self.decode_cfg.llm_largest_texts_length
                )
            
            nbest_hyps = apply_lm(
                nbest_hyps, 
                self.lm_scorer, 
                length_bonus=self.decode_cfg.llm_length_bonus,
                lm_weight=self.decode_cfg.llm_weight, 
                transform=self.transform
            )

            nbest_hyps.sort(key=lambda hyp: hyp.score, reverse=True)  # resort the texts with new scores (consider lm score)

        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]

        text, token, _ = parse_hypothesis(nbest_hyps[0], self.token_list)
        text = text.replace("▁", " ").strip().replace("<eos>", "").replace(' #', '')

        return {
            "pred": text
        }
