import sys
from collections import defaultdict

import torch


def remove_prefix(prefix, state_dict):
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def match_state_dict(target_keys, state_dict, return_prefixes=False, strict=True):
    prefix_candidates = defaultdict(lambda: 0)

    target_list = [key[::-1] for key in target_keys]
    target_list.sort()

    state_list = [key[::-1] for key in state_dict]
    state_list.sort()

    target_i = 0
    state_i = 0
    while target_i < len(target_list) and state_i < len(state_list):
        state_key = state_list[state_i]
        targ_key = target_list[target_i]
        if state_key.startswith(targ_key):
            prefix = state_key[len(targ_key) :][::-1]
            prefix_candidates[prefix] += 1

            state_i += 1
        else:
            target_i += 1

    prefixes = [prefix for prefix, count in prefix_candidates.items() if count == len(target_keys)]

    if return_prefixes:
        return prefixes

    try:
        if strict:
            assert len(prefixes) == 1

            prefix = prefixes[0]
        else:
            assert len(prefix_candidates) > 0

            prefix = max(prefix_candidates.items(), key=lambda x: x[1])[0]

        assert len(prefix) == 0 or prefix.endswith(".")
    except AssertionError:
        raise ValueError("Incompatible state_dicts: {} and {}".format(target_keys, state_dict.keys()))

    return remove_prefix(prefix, state_dict)


def load_by_all_means(path, warn=True):
    # Loading state_dicts and lightning modules

    try:
        data = torch.load(path, map_location="cpu")
    except ModuleNotFoundError:
        # maybe it is apex issue?

        import pickle

        class UpicklePatch(pickle.Unpickler):
            def find_class(self, module, name):
                if name == "FusedAdamFix" or name == "FusedAdam":
                    module = "torch.optim"
                    name = "AdamW"

                return super().find_class(module, name)

        class PicklePatch:
            Unpickler = UpicklePatch

        data = torch.load(path, pickle_module=PicklePatch, map_location="cpu")
        if warn:
            print("Warning: Supressing apex import error", file=sys.stderr)

    if "state_dict" in data and "epoch" in data:
        # Yep, we get lit_checkpoint and not state_dict

        # just in case we need them
        data.get("hparams", data.get("hyper_parameters"))

        data = data["state_dict"]

        if warn:
            print("Warning: Loading lit_checkpoint, not state_dict", file=sys.stderr)

    return data


# load state_dicts, lit_modules, fixing all known possible errors
def load_state_dict_by_all_means(model, state_dict, warn=True):
    data = state_dict

    try:
        model.load_state_dict(data)
        return
    except RuntimeError:
        try:
            fixed_data = match_state_dict(model.state_dict().keys(), data, strict=False)
        except ValueError:
            raise RuntimeError("Can't match {} with model, sorry".format(state_dict))

        try:
            model.load_state_dict(fixed_data)
            return
        except RuntimeError:
            # fix for 'embeddings.position_ids' missing in new versions of torch
            res = model.load_state_dict(fixed_data, strict=False)
            # assert len(res.unexpected_keys) == 0

            if warn:
                # print("Ignoring missing_keys: {}".format(res.missing_keys), file=sys.stderr)
                print("Warning: ignoring {}".format(res), file=sys.stderr)

            return

    raise RuntimeError("Can't load {} for model, sorry".format(state_dict))
