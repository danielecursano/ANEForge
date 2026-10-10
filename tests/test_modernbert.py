"""ModernBERT graph helpers and checkpoint mapping."""
from types import SimpleNamespace

import numpy as np
import pytest

from _helpers import requires_ane
from aneforge.graph import input
from aneforge.models import Encoder, _modernbert_band, _modernbert_rope, _modernbert_rotate


def test_modernbert_rope_and_band():
  cos, sin = _modernbert_rope(5, 8, 10000.0)
  assert cos.shape == sin.shape == (1, 5, 8)
  assert np.array_equal(cos[0, 0], np.ones(8))
  assert np.array_equal(sin[0, 0], np.zeros(8))
  assert np.allclose(cos[0, :, :4], cos[0, :, 4:])
  assert np.array_equal(_modernbert_band(5, 1)[0, 2], [-1e4, 0, 0, 0, -1e4])
  assert _modernbert_rotate(input((2, 5, 8)), cos, sin).shape == (2, 5, 8)


def test_modernbert_embedded_checkpoint_mapping():
  cfg = SimpleNamespace(model_type="modernbert", hidden_size=8, num_attention_heads=2,
                        num_hidden_layers=1, norm_eps=1e-5, local_attention=4,
                        global_attn_every_n_layers=3, rope_parameters={"rope_theta": 12000})
  prefix = "encoder."
  weights = {
    prefix + "embeddings.tok_embeddings.weight": np.ones((10, 8)),
    prefix + "embeddings.norm.weight": np.ones(8),
    prefix + "final_norm.weight": np.ones(8),
  }
  for name, shape in (("attn.Wqkv.weight", (24, 8)), ("attn.Wo.weight", (8, 8)),
                      ("mlp.Wi.weight", (32, 8)), ("mlp.Wo.weight", (8, 16)),
                      ("mlp_norm.weight", (8,))):
    weights[prefix + "layers.0." + name] = np.ones(shape)
  enc = Encoder("unused", state_dict=weights, config=cfg, tokenizer=object())
  assert enc.modernbert and enc.D == 8 and enc.rel_bias is None
  assert enc.layers[0]["Wqkv"].shape == (24, 8)
  assert enc.rope_theta == {"full_attention": 12000.0, "sliding_attention": 12000.0}
  with pytest.raises(ValueError, match="shape"):
    enc.hidden_states(np.zeros((2, 4), np.int64), np.ones((2, 4), np.int64))


@requires_ane
def test_modernbert_base_matches_hf():
  import torch
  from transformers import AutoModel
  import aneforge as af

  name = "answerdotai/ModernBERT-base"
  enc = af.load(name)
  tokens = enc.tok("ModernBERT runs on the Neural Engine.", return_tensors="pt")
  with torch.no_grad():
    ref = AutoModel.from_pretrained(name).eval()(**tokens).last_hidden_state.float().numpy()
  got = enc.hidden_states(tokens["input_ids"].numpy(), tokens["attention_mask"].numpy())
  cosine = np.sum(got * ref, -1) / (np.linalg.norm(got, axis=-1) * np.linalg.norm(ref, axis=-1) + 1e-9)
  assert float(cosine.mean()) > 0.99
  enc.release()
