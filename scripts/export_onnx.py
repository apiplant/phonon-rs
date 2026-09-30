"""Export Phonon-2 (model.fermion) to the ONNX layout transcribe-rs's Parakeet engine loads.

    <out>/encoder-model.onnx        audio_signal [B,128,T], length [B] -> outputs [B,640,T'], encoded_lengths [B]
    <out>/decoder_joint-model.onnx  encoder_outputs [B,640,1], targets [B,1], target_length [B],
                                    input_states_1/2 [2,B,640] -> outputs [B,1,1,8198], prednet_lengths,
                                    output_states_1/2
    <out>/nemo128.onnx, vocab.txt   copied from istupakov/parakeet-tdt-0.6b-v3-onnx (weight-free preprocessor)

The joint's encoder projection (1024 -> 640) runs once per utterance inside the encoder graph rather than
once per decode step in the decoder graph; transcribe-rs does not look at the channel count.

With --fp16 the encoder is written as encoder-model.fp16.onnx (fp16 weights and activations, fp32 at the graph
edges), which transcribe-rs picks with Quantization::FP16 (`phonon --backend onnx --dtype f16`).

Usage: python export_onnx.py ../model_phonon2_c4c_int6/model.fermion out_dir [--fp16]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from reference_transformers import container_state_dict  # noqa: E402


class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.proj = model.encoder_projector

    def forward(self, audio_signal, length):
        feats = audio_signal.transpose(1, 2)  # [B,T,128]
        mask = torch.arange(feats.shape[1], device=feats.device)[None, :] < length[:, None]
        out = self.encoder(input_features=feats, attention_mask=mask)
        enc = self.proj(out.last_hidden_state)  # [B,T',640]
        lengths = out.attention_mask.sum(-1).to(torch.int64)
        return enc.transpose(1, 2), lengths


class DecoderJoint(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder
        self.joint = model.joint

    def forward(self, encoder_outputs, targets, target_length, input_states_1, input_states_2):
        emb = self.decoder.embedding(targets.to(torch.int64))  # [B,1,640]
        out, (h, c) = self.decoder.lstm(emb, (input_states_1, input_states_2))
        g = self.decoder.decoder_projector(out)  # [B,1,640]
        f = encoder_outputs.transpose(1, 2)  # [B,1,640]
        logits = self.joint(decoder_hidden_states=g[:, None, :, :], encoder_hidden_states=f[:, :, None, :])
        return logits, target_length, h, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("container")
    ap.add_argument("out")
    ap.add_argument("--base-dir", default="nvidia/parakeet-tdt-0.6b-v3")
    ap.add_argument("--onnx-repo", default="istupakov/parakeet-tdt-0.6b-v3-onnx")
    ap.add_argument("--fp16", action="store_true", help="write encoder-model.fp16.onnx instead of the fp32 encoder")
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from transformers import ParakeetForTDT, ParakeetTDTConfig

    cfg = ParakeetTDTConfig.from_pretrained(a.base_dir)
    model = ParakeetForTDT(cfg)
    sd, _ = container_state_dict(a.container)
    model.load_state_dict(sd, strict=True)
    model.eval()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    enc = Encoder(model)
    if a.fp16:
        enc = enc.half()
    dtype = torch.float16 if a.fp16 else torch.float32
    x = torch.randn(1, 128, 400, dtype=dtype)
    ln = torch.tensor([400], dtype=torch.int64)
    tmp = out / "_export_tmp"
    tmp.mkdir(exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            enc, (x, ln), str(tmp / "encoder.onnx"),
            input_names=["audio_signal", "length"], output_names=["outputs", "encoded_lengths"],
            dynamic_axes={"audio_signal": {0: "B", 2: "T"}, "length": {0: "B"},
                          "outputs": {0: "B", 2: "T_out"}, "encoded_lengths": {0: "B"}},
            opset_version=17, dynamo=False, external_data=True,
        )
    # the legacy exporter writes one external file per initializer; repack them into one .data file
    import onnx
    from onnx import TensorProto, helper
    m = onnx.load(str(tmp / "encoder.onnx"))
    g = m.graph
    if a.fp16:
        # transcribe-rs feeds and reads fp32; wrap the fp16 graph with casts at the edges
        g.input[0].type.tensor_type.elem_type = TensorProto.FLOAT
        for n in g.node:
            n.input[:] = ["audio_signal_f16" if s == "audio_signal" else s for s in n.input]
            n.output[:] = ["outputs_f16" if s == "outputs" else s for s in n.output]
        g.node.insert(0, helper.make_node("Cast", ["audio_signal"], ["audio_signal_f16"], to=TensorProto.FLOAT16))
        g.node.append(helper.make_node("Cast", ["outputs_f16"], ["outputs"], to=TensorProto.FLOAT))
        g.output[0].type.tensor_type.elem_type = TensorProto.FLOAT
    shutil.rmtree(tmp)
    name = "encoder-model.fp16.onnx" if a.fp16 else "encoder-model.onnx"
    for f in out.glob(name + "*"):
        f.unlink()
    onnx.save(m, str(out / name), save_as_external_data=True, all_tensors_to_one_file=True,
              location=name + ".data", size_threshold=1024)

    dj = DecoderJoint(model)
    args = (torch.randn(1, 640, 1), torch.zeros(1, 1, dtype=torch.int32), torch.ones(1, dtype=torch.int32),
            torch.zeros(2, 1, 640), torch.zeros(2, 1, 640))
    with torch.no_grad():
        torch.onnx.export(
            dj, args, str(out / "decoder_joint-model.onnx"),
            input_names=["encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"],
            output_names=["outputs", "prednet_lengths", "output_states_1", "output_states_2"],
            dynamic_axes={"encoder_outputs": {0: "B"}, "targets": {0: "B"}, "target_length": {0: "B"},
                          "input_states_1": {1: "B"}, "input_states_2": {1: "B"},
                          "outputs": {0: "B"}, "prednet_lengths": {0: "B"},
                          "output_states_1": {1: "B"}, "output_states_2": {1: "B"}},
            opset_version=17, dynamo=False,
        )

    for f in ("nemo128.onnx", "vocab.txt", "config.json"):
        shutil.copyfile(hf_hub_download(a.onnx_repo, f), out / f)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
