import enum
import os
import json
import uuid
import logging
from pathlib import Path
from typing import Set
import torch as th

CHECKPOPINTS_METADATA_FILE_NAME = 'checkpoints.metadata'

_logger = logging.getLogger("checkpoint_manager")


class _CheckpointMetadataDictKeys(str, enum.Enum):
    NEXT_ID = 'next_id'
    PRNT_ID = 'parent_id'
    ID_2_PRNT = 'id_2_prnt'
    ID_2_PATH = 'id_2_path'
    ID_2_META = 'id_2_meta'


class _CheckpointDictKeys(str, enum.Enum):
    MODEL = 'model'
    OPTIM = 'optimizer'
    LRATE = 'learning_rate'
    BSIZE = 'batch_size'
    TDATA = 'train_dataset'
    EDATA = 'eval_dataset'
    ENAME = 'experiment_name'


class CheckpointManager(object):
    def __init__(self, root_directory: str = 'root_experiment') -> None:
        self.root_directory = Path(root_directory)

        self.next_id = -1
        self.prnt_id = -1
        self.id_to_path = {}
        self.id_to_prnt = {}
        self.id_to_meta = {}

        Path(root_directory).mkdir(parents=True, exist_ok=True)
        self._load_metadata()

    def __repr__(self) -> str:
        return f'CheckpointManager(root_directory={self.root_directory})\n' + \
            f'next_id={self.next_id}\n' + \
            f'prnt_id={self.prnt_id}\n' + \
            f'id_to_prnt={self.id_to_prnt}\n' + \
            f'id_to_path={self.id_to_path}\n' + \
            f'id_to_meta={self.id_to_meta}\n'

    def get_ids(self) -> Set[int]:
        return set(self.id_to_path.keys())

    def get_path_for_id(self, id: int) -> Path:
        return self.id_to_path[id]

    def _generate_checkpoint_id(self):
        self.next_id += 1
        return self.next_id

    def attach_metadata(self, checkpoint_id: int, metadata: dict):
        if checkpoint_id in self.id_to_path:
            raise ValueError(f"Checkpoint {checkpoint_id} does not exist.")

        self.id_to_meta[checkpoint_id] = dict(metadata)

    def get_metadata(self, checkpoint_id: int) -> dict:
        if checkpoint_id in self.id_to_path:
            raise ValueError(f"Checkpoint {checkpoint_id} does not exist.")

        return self.id_to_meta[checkpoint_id]

    def get_latest_experiment(self):
        return self.next_id

    def _dump_metadata(self):
        state_dict = {
            _CheckpointMetadataDictKeys.NEXT_ID: self.next_id,
            _CheckpointMetadataDictKeys.PRNT_ID: self.prnt_id,
            _CheckpointMetadataDictKeys.ID_2_PATH: self.id_to_path,
            _CheckpointMetadataDictKeys.ID_2_PRNT: self.id_to_prnt,
            _CheckpointMetadataDictKeys.ID_2_META: self.id_to_meta
        }

        file_path = self.root_directory.joinpath(
            CHECKPOPINTS_METADATA_FILE_NAME)
        with open(file_path, 'w') as ckpt_metadata_file:
            ckpt_metadata_file.write(json.dumps(state_dict))

    def _load_metadata(self):
        state_dict = None
        file_path = self.root_directory.joinpath(
            CHECKPOPINTS_METADATA_FILE_NAME)

        if not file_path.exists():
            print("Checkpoint manager load: ", file_path, " not found")
            return

        with open(file_path, 'r') as ckpt_metadata_file:
            state_dict = json.loads(ckpt_metadata_file.read())

        if state_dict is None:
            return

        self.next_id = state_dict[_CheckpointMetadataDictKeys.NEXT_ID]
        self.prnt_id = state_dict[_CheckpointMetadataDictKeys.PRNT_ID]
        self.id_to_path = state_dict[_CheckpointMetadataDictKeys.ID_2_PATH]
        self.id_to_prnt = state_dict[_CheckpointMetadataDictKeys.ID_2_PRNT]

    def dump(self, experiment: "Experiment"):
        current_ckpt_id = self._generate_checkpoint_id()
        _logger.info(
            "Dumping experiment: %d", current_ckpt_id)
        self.id_to_prnt[current_ckpt_id] = self.prnt_id
        self.prnt_id = current_ckpt_id
        file_name = "ckpt_" + str(current_ckpt_id) + "_" + str(uuid.uuid4())
        ckpt_save_path = self.root_directory.joinpath(Path(file_name))

        th.save({
            _CheckpointDictKeys.ENAME: experiment.name,
            _CheckpointDictKeys.BSIZE: experiment.batch_size,
            _CheckpointDictKeys.LRATE: experiment.learning_rate,
            _CheckpointDictKeys.MODEL: experiment.model.state_dict(),
            _CheckpointDictKeys.OPTIM: experiment.optimizer.state_dict(),
            _CheckpointDictKeys.EDATA:
                experiment.eval_loader.dataset.state_dict(),
            _CheckpointDictKeys.TDATA:
                experiment.train_loader.dataset.state_dict(),
        }, ckpt_save_path)

        self.id_to_path[current_ckpt_id] = str(ckpt_save_path)
        self._dump_metadata()
        return current_ckpt_id

    def load(self, checkpoint_id: int | str, experiment: "Experiment"):
        _logger.info(f"Loading checkpoint: {checkpoint_id}")

        if checkpoint_id not in self.id_to_path:
            checkpoint_id = str(checkpoint_id)

        if checkpoint_id not in self.id_to_path:
            _logger.warning(f"Checkpoint {checkpoint_id} not found")
            return

        if not os.path.exists(self.id_to_path[checkpoint_id]):
            _logger.warning(f"Checkpoint {checkpoint_id} file not found")
            return

        self.prnt_id = checkpoint_id
        try:
            ckpt_dict = th.load(self.id_to_path[checkpoint_id])
        except Exception as e:
            _logger.error(
                f"Could not load checkpoint {checkpoint_id} due to {str(e)}")   

        experiment.name = ckpt_dict[_CheckpointDictKeys.ENAME]
        experiment.batch_size = ckpt_dict[_CheckpointDictKeys.BSIZE]
        experiment.learning_rate = ckpt_dict[_CheckpointDictKeys.LRATE]
        experiment.model.load_state_dict(
            ckpt_dict[_CheckpointDictKeys.MODEL], strict=False)
        try:
            experiment.optimizer.load_state_dict(
                ckpt_dict[_CheckpointDictKeys.OPTIM])
        except ValueError:
            print(
                f"Checkpoint {checkpoint_id} does not contain the same "
                f"version of the optimizer the as the current experiment.")

        experiment.eval_loader.dataset.load_state_dict(
            ckpt_dict[_CheckpointDictKeys.EDATA])
        experiment.train_loader.dataset.load_state_dict(
            ckpt_dict[_CheckpointDictKeys.TDATA])

        experiment.reset_data_iterators()
