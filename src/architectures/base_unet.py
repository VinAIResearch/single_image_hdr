import torch
import torch.nn as nn

from .modules import UNetDecSPADEBlock, act


class BaseUNet(nn.Module):
    def __init__(
        self,
        enc_blk,
        dec_blk,
        downsampler,
        upsampler,
        last_conv,
        in_channels=3,
        out_channels=3,
        nfeats=32,
        maxfeats=None,
        nlayers=5,
        global_res="concat",
    ):
        super().__init__()

        if maxfeats is None:
            maxfeats = (2 ** (nlayers - 1)) * nfeats

        self.nlayers = nlayers
        self.global_res = global_res

        # Encoder
        feats = {"enc1": (in_channels, 1 * nfeats)}
        for i in range(1, nlayers + 1):
            ins, outs = feats[f"enc{i}"]
            setattr(self, f"enc{i}", enc_blk(ins, outs))
            if i > 1:  # Skip the first encoding layer
                setattr(self, f"down{i}", downsampler())

            next_ins = outs
            next_outs = min(2 * outs, maxfeats)
            feats[f"enc{i + 1}"] = (next_ins, next_outs)

        # Decoder
        ins, outs = feats[f"enc{nlayers}"]
        for i in range(nlayers, 1, -1):
            setattr(self, f"up{i}", upsampler(outs))
            prev_ins, prev_outs = feats[f"enc{i - 1}"]
            ins = outs + prev_outs
            outs = prev_ins if prev_ins != in_channels else outs
            setattr(self, f"dec{i}", dec_blk(ins, outs))

        ins_dec1 = in_channels + nfeats if global_res == "concat" else nfeats
        self.dec1 = nn.Sequential(
            last_conv["conv"](in_channels=ins_dec1, out_channels=out_channels),
            act(last_conv["act_type"]),
        )

    def forward(self, x, **kwargs):
        # Encoder
        mask = kwargs.get("mask", None)
        encs = [x * mask if mask is not None else x]
        encs.append(self.enc1(encs[0]))
        for i in range(2, self.nlayers + 1):
            downsampled = getattr(self, f"down{i}")(encs[i - 1])
            encs.append(getattr(self, f"enc{i}")(downsampled))

        # Decoder
        dec = encs[-1]
        for i in range(self.nlayers, 1, -1):
            upsampled = getattr(self, f"up{i}")(dec)
            dec = torch.cat([encs[i - 1], upsampled], dim=1)
            dec_blk = getattr(self, f"dec{i}")
            dec = (
                dec_blk(dec)
                if not isinstance(dec_blk, UNetDecSPADEBlock)
                else dec_blk(dec, mask)
            )

        if self.global_res == "concat":
            dec = torch.cat([encs[0], dec], dim=1)

        x_hat = self.dec1(dec)

        return x_hat, encs[0]
