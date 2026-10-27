"""Patch rules: where a series is cut into patches.

The forecaster turns every patch into one Transformer token, so these rules decide how much the sequence is compressed and
where the Transformer "looks". Both rules return the start index of every patch (always beginning with 0), use only values
up to the index they start at (so the model stays causal), and run on the CPU in dataloader workers.
"""
