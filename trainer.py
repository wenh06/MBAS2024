import argparse
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader, Dataset
from torch_ecg.cfg import CFG
from torch_ecg.components import BaseTrainer
from torch_ecg.utils.misc import str2bool
from tqdm.auto import tqdm

from cfg import ModelCfg, TrainCfg
from dataset import MBAS2024Dataset
from models import MultiHead_MBAS2024
from utils.scoring_metrics import compute_challenge_metrics

__all__ = [
    "MBAS2024Trainer",
]


class MBAS2024Trainer(BaseTrainer):
    """Trainer for the MBAS2024 model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained
    model_config : dict
        The configuration of the model,
        used to keep a record in the checkpoints
    train_config : dict
        The configuration of the training,
        including configurations for the data loader, for the optimization, etc.
        will also be recorded in the checkpoints.
        `train_config` should at least contain the following keys:

            - "stage": obj:`int`, the stage of the pipeline, 0 for raw localization, 1 for fine segmentation
            - "loss": obj:`str`,
            - "n_epochs": obj:`int`,
            - "batch_size": obj:`int`,
            - "learning_rate": obj:`float`,
            - "lr_scheduler": obj:`str`,
            - "lr_step_size": obj:`int`, optional, depending on the scheduler
            - "lr_gamma": obj:`float`, optional, depending on the scheduler
            - "max_lr": obj:`float`, optional, depending on the scheduler
            - "optimizer": obj:`str`,
            - "decay": obj:`float`, optional, depending on the optimizer
            - "momentum": obj:`float`, optional, depending on the optimizer

    device : torch.device, optional
        The device to be used for training

    """

    __DEBUG__ = True
    __name__ = "MBAS2024Trainer"

    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> None:
        assert train_config.stage in [0, 1], "stage must be 0 or 1"
        if train_config.stage == 0:
            train_config.classes = train_config.stage0_classes
        else:
            train_config.classes = train_config.stage1_classes
        # check if the "apply_mclahe" is consistent in the model and the training configuration
        assert (
            model_config.apply_mclahe == train_config.apply_mclahe
        ), "apply_mclahe should be consistent in the model and the training configuration"
        assert (
            model_config.use_tio_transforms == train_config.use_tio_transforms
        ), "use_tio_transforms should be consistent in the model and the training configuration"
        super().__init__(
            model=model,
            dataset_cls=MBAS2024Dataset,
            model_config=model_config,
            train_config=train_config,
            device=device,
            **kwargs,
        )

    def _setup_dataloaders(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ) -> None:
        """
        setup the dataloaders for training and validation

        Parameters
        ----------
        train_dataset: Dataset, optional,
            the training dataset
        val_dataset: Dataset, optional,
            the validation dataset

        """
        if train_dataset is None:
            train_dataset = self.dataset_cls(
                config=self.train_config,
                stage=self.train_config.stage,
            )

        if self.train_config.debug:
            val_train_dataset = train_dataset
        else:
            val_train_dataset = None

        # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4
        if self.device == torch.device("cpu"):
            num_workers = 1
        else:
            num_workers = 4

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        if self.train_config.debug:
            self.val_train_loader = DataLoader(
                dataset=val_train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
            )
        else:
            self.val_train_loader = None

    def train_one_epoch(self, pbar: tqdm) -> None:
        """Train one epoch, and update the progress bar

        Parameters
        ----------
        pbar : tqdm
            the progress bar for training

        """
        for epoch_step, input_tensors in enumerate(self.train_loader):
            self.global_step += 1
            n_samples = input_tensors["image"].shape[self.batch_dim]

            out_tensors = self.run_one_step(input_tensors)

            # NOTE: loss is computed in the model, and kept in the out_tensors
            loss = out_tensors["total_loss"]
            # if trained on multiple GPUs (N), then loss has shape (N,)
            # in order to run loss.backward(), the loss should be averaged
            # over all GPUs
            loss = loss.mean()

            if self.train_config.flooding_level > 0:
                flood = (loss - self.train_config.flooding_level).abs() + self.train_config.flooding_level
                self.epoch_loss += loss.item()
                self.optimizer.zero_grad()
                flood.backward()
            else:
                self.epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
            self.optimizer.step()
            self._update_lr()

            if self.global_step % self.train_config.log_step == 0:
                train_step_metrics = {"loss": loss.item()}
                if self.scheduler:
                    train_step_metrics.update({"lr": self.scheduler.get_last_lr()[0]})
                    pbar.set_postfix(
                        **{
                            "loss (batch)": loss.item(),
                            "lr": self.scheduler.get_last_lr()[0],
                        }
                    )
                else:
                    pbar.set_postfix(
                        **{
                            "loss (batch)": loss.item(),
                        }
                    )
                if self.train_config.flooding_level > 0:
                    train_step_metrics.update({"flood": flood.item()})
                self.log_manager.log_metrics(
                    metrics=train_step_metrics,
                    step=self.global_step,
                    epoch=self.epoch,
                    part="train",
                )
            pbar.update(n_samples)

    def run_one_step(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run one step (batch) of training

        Parameters
        ----------
        input_tensors : dict
            the tensors to be processed for training one step (batch), with the following items:
                - "image" (required): the input waveforms
                - "dx" (optional): the Dx classification labels
                - "digitization" (optional): the signal reconstruction labels
                - "mask" (optional): the mask for the signal reconstruction

        Returns
        -------
        out_tensors : dict
            with the following items (some are optional):
            - "seg_logits": the segmentation logits, of shape ``(B, H, W, D, n_classes)``.
            - "seg_mask": the segmentation mask, of shape ``(B, H, W, D)``.
            - "seg_loss": the segmentation loss
            - "total_loss": the total loss for the training step

        """
        image = self._model.get_input_tensors(input_tensors.pop("image"))
        out_tensors = self.model(image, input_tensors)
        return out_tensors

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on the given data loader"""

        self.model.eval()

        all_outputs = []
        all_labels = []

        with tqdm(
            total=len(data_loader.dataset),
            desc="Evaluation",
            unit="image",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=False,
        ) as pbar:
            for input_tensors in data_loader:
                # input_tensors is assumed to be a dict of tensors, with the following items:
                # "image" (required): the input image list
                # "dx" (optional): the Dx classification labels
                # "digitization" (optional): the signal reconstruction labels
                # "mask" (optional): the mask for the signal reconstruction
                # image = self._model.get_input_tensors(input_tensors.pop("image"))
                image = input_tensors.pop("image")
                labels = {k: v.numpy() for k, v in input_tensors.items() if v is not None}

                all_labels.append(labels)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                all_outputs.append(self._model.inference(image))  # of type MBAS2024Outputs
                pbar.update(len(image))

        eval_res = compute_challenge_metrics(
            labels=all_labels,
            outputs=all_outputs,
            ignore_index=[0],  # ignore background
            average="samples-flatten",  # to format "class-metric": value
            use_official_metric=True,
            progress=True,
        )

        # in case possible memeory leakage?
        del all_labels
        del all_outputs

        self.model.train()

        return eval_res

    @property
    def batch_dim(self) -> int:
        """
        batch dimension, usually 0,
        but can be 1 for some models, e.g. RR_LSTM
        """
        return 0

    @property
    def extra_required_train_config_fields(self) -> List[str]:
        return ["stage"]

    @property
    def save_prefix(self) -> str:
        prefix = f"""{self.model_config.seg_model_name}-Stage{self.train_config.stage}"""
        return prefix + "_"

    def extra_log_suffix(self) -> str:
        suffix = f"""{self.model_config.seg_model_name}-Stage{self.train_config.stage}"""
        suffix = f"{suffix}-{super().extra_log_suffix()}"
        return suffix

    def _setup_criterion(self) -> None:
        # since criterion is defined in the model,
        # override this method to do nothing
        pass


def get_args(**kwargs: Any):
    """NOT checked,"""
    cfg = deepcopy(kwargs)
    parser = argparse.ArgumentParser(
        description="Train the Model on CINC2024 database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=int,
        help="the stage of the pipeline",
        dest="stage",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=4,
        help="the batch size for training",
        dest="batch_size",
    )
    parser.add_argument(
        "--keep-checkpoint-max",
        type=int,
        default=10,
        help="maximum number of checkpoints to keep. If set 0, all checkpoints will be kept",
        dest="keep_checkpoint_max",
    )
    # parser.add_argument(
    #     "--optimizer", type=str, default="adam",
    #     help="training optimizer",
    #     dest="train_optimizer")
    parser.add_argument(
        "--debug",
        type=str2bool,
        default=False,
        help="train with more debugging information",
        dest="debug",
    )

    args = vars(parser.parse_args())

    cfg.update(args)

    return CFG(cfg)


if __name__ == "__main__":
    # WARNING: most training were done in notebook,
    # NOT in cli
    train_config = get_args(**TrainCfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: adjust for MBAS2024
    model_config = deepcopy(ModelCfg)
    # adjust the model configuration if necessary
    model = MultiHead_MBAS2024(stage=train_config.stage, config=model_config)

    if torch.cuda.device_count() > 1:
        model = DP(model)
        # model = DDP(model)
    model = model.to(device=device)

    trainer = MBAS2024Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=device,
    )

    try:
        best_model_state_dict = trainer.train()
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
