import scipy.io as scio
import numpy as np
import pandas as pd
from pathlib import Path


HUMAN_GENERALIZATION_TASKS = {
    "Animate vs. Inanimate": "animate_vs_inanimate_gen",
    "Natural vs. Artificial": "natural_vs_artificial_gen",
    "Mammal vs. Non-mammal": "mammal_vs_reptile_gen",
    "Tools vs. Electronics": "electronics_vs_tools_gen",
    "Human vs. Monkey": "human_vs_monkey_gen",
    "Monkey vs. Mammal": "monkey_vs_mammal_gen",
    "Animate vs. Inanimate(THINGS)": "Large_animate_vs_inanimate",
    "Natural vs. Artificial(THINGS)": "Large_natural_vs_artificial",
    "Mammal vs. Non-mammal(THINGS)": "Large_mammal_vs_nonmammal",
    "Tools vs. Electronics(THINGS)": "electronics_vs_tools_THINGS_gen",
    "Outdoor vs. Indoor": "outdoor_vs_indoor_gen",
    "Big vs. Small(THINGS)": "big_vs_small_gen",
    "Big vs. Small(Konkle)": "big_vs_small_Konkle_gen",
    "Fire related vs. Water related(THINGS)": "fire related_vs_water related_gen",
    "Eastern vs. Western": "eastern_vs_western_gen",
}


def session_table(species, root="data_behavior"):
    """Read curated sessions in the preserved worksheet row order."""
    if species not in {"Human", "Monkey"}:
        raise ValueError(f"Unknown species: {species}")
    frame = pd.read_excel(Path(root) / f"{species}_sessions.xlsx",
                          sheet_name="Sessions", dtype={"subject": str})
    return frame.reset_index(drop=True)


def human_generalization_sessions():
    """Return shared generalization sessions with their manuscript task labels."""
    frame = session_table("Human")
    labels = {task: label for label, task in HUMAN_GENERALIZATION_TASKS.items()}
    frame = frame[frame["task"].isin(labels)].copy()
    frame["task_label"] = frame["task"].map(labels)
    return frame


def _load_sessions(frame, root, extend, verbose, **filters):
    data, files = [], []
    for row in frame.itertuples(index=False):
        path = (Path(root) / row.task / row.filename).as_posix()
        trials = loadTrials(path, **filters)
        if verbose:
            print(path, f"Trials Num:{len(trials)}")
        data.extend(trials) if extend else data.append(trials)
        files.append(path)
    return data, files


def loadTrialsM(task, monkey, ismain=True, islearn=True, extend=True,
                verbose=False, root="data_behavior/Monkey", response=True,
                two_target=True, no_correction_loop=True):
    """Load shared monkey sessions. All indexed sessions are main experiments.

    ismain is retained for notebook compatibility. islearn=True selects only
    sessions marked learned=1; False includes all shared training stages.
    Texform sessions have no learned annotation; use islearn=False for them.
    """
    frame = session_table("Monkey", Path(root).parent)
    frame = frame[(frame["task"] == task) & (frame["subject"] == monkey)]
    if islearn:
        frame = frame[frame["learned"] == 1]
    return _load_sessions(frame, root, extend, verbose, response=response,
                          two_target=two_target, no_correction_loop=no_correction_loop)


def loadTrialsH(task, human, isvalid=True, ismain=True, extend=True,
                verbose=False, root="data_behavior/Human", response=True,
                two_target=True, no_correction_loop=True):
    """Load shared human sessions; every indexed session is valid and included.

    isvalid and ismain remain for notebook compatibility. The shared index
    intentionally excludes invalid and non-main source sessions.
    """
    frame = session_table("Human", Path(root).parent)
    frame = frame[(frame["task"] == task) & (frame["subject"] == str(human))]
    return _load_sessions(frame, root, extend, verbose, response=response,
                          two_target=two_target, no_correction_loop=no_correction_loop)


def _plain_field(v):
    """Trial-struct field -> plain value (float / str / object-array / ndarray).

    The shared session files store trial_data as an Nx1 cell array of
    trial structs with plain fields (rig source container format).
    scipy represents the struct fields of a cell element
    as one more object-array layer; unwrap it here.
    """
    v = np.asarray(v)
    if v.dtype.kind == "O":
        vals = [_plain_field(x) for x in v.ravel()]   # cells may nest
        if len(vals) == 1:
            return vals[0]
        return np.array(vals, dtype=object)   # e.g. the two category names
    if v.dtype.kind in "US":
        return str(v.flatten()[0]) if v.size == 1 else np.array(
            [str(x) for x in v.ravel()], dtype=object)
    return v.item() if v.size == 1 else v.squeeze()


def loadTrials(file,response=True,two_target=True,no_correction_loop=True):
    """Load one shared session file.

    The files already contain only responded (CORRECT/WRONG),
    non-correction-loop trials. The filters below are kept for signature
    compatibility and are no-ops on shared files, except two_target:
    hide_wrg_target trials ARE stored (Figure1_2 marks them NaN), so
    two_target=True (default) still drops them at load time.

    trial_type uses one vocabulary for every task and species:
    'main' = trained-image trial, 'gen' = generalization (novel-image)
    trial.  Monkey LargeTest 'anchor'/'stim' and the human block-final
    'last' marker were folded into this scheme; the legacy per-trial
    'is_gen' and 'valid' fields no longer exist.
    """
    td = scio.loadmat(file)["trial_data"]
    if td.size == 0:
        print("No trials")
        return []
    trials = []
    for rec in td.reshape(-1):
        t = {}
        for name in rec.dtype.names:
            v = _plain_field(rec[name])
            if isinstance(v, np.ndarray) and v.size == 0:
                continue          # field absent for this trial
            t[name] = v
        if response and t.get("result") not in ("WRONG", "CORRECT"):
            continue
        if two_target:
            if "hide_wrg_target" in t and t["hide_wrg_target"] != 0:
                continue
            if "hide_wrgtarg" in t and t["hide_wrgtarg"] != 0:
                continue
        if no_correction_loop and t.get("correction_loop_flg", 0) != 0:
            continue
        trials.append(t)
    return trials

def unpack_basic(item):
    if np.issubdtype(item.dtype,np.number):
        item = item.squeeze()
        if item.shape == ():
            item = item.item()
    elif np.issubdtype(item.dtype,np.unicode_):
        if item.shape != (0,):
            item = item[0]
    return item

def unpack_cell(cell):
    '''
    Only support 1-D and 2-D cell
    '''
    if len(cell.shape) == 1:
        s = cell.shape
        for i in range(s[0]):
            if isinstance(cell[i],scio.matlab._mio5_params.MatlabOpaque):
                # I donot process container.Map now
                continue
            elif np.issubdtype(cell[i].dtype,np.object_):
                cell[i] = unpack_cell(cell[i])
            elif np.issubdtype(cell[i].dtype,np.void):
                cell[i] = unpack_struct(cell[i])
            else:
                cell[i] = unpack_basic(cell[i])  
    elif len(cell.shape) == 2:
        s1,s2 = cell.shape
        for i in range(s1):
            for j in range(s2):
                if isinstance(cell[i,j],scio.matlab._mio5_params.MatlabOpaque):
                    # I donot process container.Map now
                    continue
                elif np.issubdtype(cell[i,j].dtype,np.object_):
                    cell[i,j] = unpack_cell(cell[i,j])
                elif np.issubdtype(cell[i,j].dtype,np.void):
                    cell[i,j] = unpack_struct(cell[i,j])
                else:
                    cell[i,j] = unpack_basic(cell[i,j])
    elif len(cell.shape) >= 3:
        raise Exception("Only support 1-D and 2-D cell")

    cell = cell.squeeze()
    if cell.shape == ():
        cell = cell.item()
    return cell

def unpack_struct(struct):
    new_struct = {}
    for name in struct.dtype.names:
        item = struct[name]
        if np.issubdtype(item.dtype,np.object_):
            new_struct[name] = unpack_cell(item)
        elif np.issubdtype(item.dtype,np.void):
            new_struct[name] = unpack_struct(item)
        else:
            new_struct[name] = unpack_basic(item)
    return new_struct