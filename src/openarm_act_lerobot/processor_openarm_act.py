"""Processors for policy type "openarm_act": ACT's own, unchanged.

Images pass through them as they are (VISUAL normalisation is IDENTITY);
the model resizes and normalises them, and fuses the depth camera into the
RGB one (modeling_openarm_act).
"""

from lerobot.policies.act.processor_act import make_act_pre_post_processors


def make_openarm_act_pre_post_processors(config, dataset_stats=None):
    return make_act_pre_post_processors(config, dataset_stats)
