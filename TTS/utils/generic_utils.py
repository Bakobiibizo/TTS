import datetime
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def get_git_branch():
    try:
        out = subprocess.check_output(["git", "branch"]).decode("utf8")
        current = next(line for line in out.split("\n")
                       if line.startswith("*"))
        current.replace("* ", "")
    except subprocess.CalledProcessError:
        current = "inside_docker"
    return current


def get_commit_hash():
    """https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script"""
    # try:
    #     subprocess.check_output(['git', 'diff-index', '--quiet',
    #                              'HEAD'])  # Verify client is clean
    # except:
    #     raise RuntimeError(
    #         " !! Commit before training to get the commit hash.")
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    # Not copying .git folder into docker container
    except subprocess.CalledProcessError:
        commit = "0000000"
    print(f' > Git Hash: {commit}')
    return commit


def create_experiment_folder(root_path, model_name, debug):
    """ Create a folder with the current date and time """
    date_str = datetime.datetime.now().strftime("%B-%d-%Y_%I+%M%p")
    commit_hash = 'debug' if debug else get_commit_hash()
    output_folder = os.path.join(
        root_path, f'{model_name}-{date_str}-{commit_hash}'
    )
    os.makedirs(output_folder, exist_ok=True)
    print(f" > Experiment folder: {output_folder}")
    return output_folder


def remove_experiment_folder(experiment_path):
    """Check folder if there is a checkpoint, otherwise remove the folder"""

    if checkpoint_files := glob.glob(f"{experiment_path}/*.pth.tar"):
        print(f" ! Run is kept in {experiment_path}")
    elif os.path.exists(experiment_path):
        shutil.rmtree(experiment_path, ignore_errors=True)
        print(f" ! Run is removed from {experiment_path}")


def count_parameters(model):
    r"""Count number of trainable parameters in a network"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_user_data_dir(appname):
    if sys.platform == "win32":
        import winreg  # pylint: disable=import-outside-toplevel
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
        )
        dir_, _ = winreg.QueryValueEx(key, "Local AppData")
        ans = Path(dir_).resolve(strict=False)
    elif sys.platform == 'darwin':
        ans = Path('~/Library/Application Support/').expanduser()
    else:
        ans = Path.home().joinpath('.local/share')
    return ans.joinpath(appname)


def set_init_dict(model_dict, checkpoint_state, c):
    # Partial initialization: if there is a mismatch with new and old layer, it is skipped.
    for k, v in checkpoint_state.items():
        if k not in model_dict:
            print(f" | > Layer missing in the model definition: {k}")
    # 1. filter out unnecessary keys
    pretrained_dict = {
        k: v
        for k, v in checkpoint_state.items() if k in model_dict
    }
    # 2. filter out different size layers
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if v.numel() == model_dict[k].numel()
    }
    # 3. skip reinit layers
    if c.reinit_layers is not None:
        for reinit_layer_name in c.reinit_layers:
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if reinit_layer_name not in k
            }
    # 4. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    print(f" | > {len(pretrained_dict)} / {len(model_dict)} layers are restored.")
    return model_dict


class KeepAverage():
    def __init__(self):
        self.avg_values = {}
        self.iters = {}

    def __getitem__(self, key):
        return self.avg_values[key]

    def items(self):
        return self.avg_values.items()

    def add_value(self, name, init_val=0, init_iter=0):
        self.avg_values[name] = init_val
        self.iters[name] = init_iter

    def update_value(self, name, value, weighted_avg=False):
        if name not in self.avg_values:
            # add value if not exist before
            self.add_value(name, init_val=value)
        elif weighted_avg:
            self.avg_values[name] = 0.99 * self.avg_values[name] + 0.01 * value
            self.iters[name] += 1
        else:
            self.avg_values[name] = self.avg_values[name] * \
                    self.iters[name] + value
            self.iters[name] += 1
            self.avg_values[name] /= self.iters[name]

    def add_values(self, name_dict):
        for key, value in name_dict.items():
            self.add_value(key, init_val=value)

    def update_values(self, value_dict):
        for key, value in value_dict.items():
            self.update_value(key, value)


def check_argument(name, c, enum_list=None, max_val=None, min_val=None, restricted=False, val_type=None, alternative=None):
    if alternative in c.keys() and c[alternative] is not None:
        return
    if restricted:
        assert name in c.keys(), f' [!] {name} not defined in config.json'
    if name in c.keys():
        if max_val:
            assert c[name] <= max_val, f' [!] {name} is larger than max value {max_val}'
        if min_val:
            assert c[name] >= min_val, f' [!] {name} is smaller than min value {min_val}'
        if enum_list:
            assert c[name].lower() in enum_list, f' [!] {name} is not a valid value'
        if isinstance(val_type, list):
            is_valid = any(isinstance(c[name], typ) for typ in val_type)
            assert is_valid or c[name] is None, f' [!] {name} has wrong type - {type(c[name])} vs {val_type}'
        elif val_type:
            assert isinstance(c[name], val_type) or c[name] is None, f' [!] {name} has wrong type - {type(c[name])} vs {val_type}'
