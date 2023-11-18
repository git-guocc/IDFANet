from typing import Any, Dict, Optional, Tuple
import logging
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class ShapeNetDataModule(LightningDataModule):

    def __init__(
            self,
            dataset,
            data_dir: str = "data/",
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
    ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset = dataset

        # data transformations
        self.transforms = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already

        if stage == "fit" and not self.data_train and not self.data_val:
            self.data_train = self.dataset(
                data_path=self.hparams.data_dir, split='train', transform=self.transforms
            )
            logging.info(f"Train dataset loaded, size: {len(self.data_train)}")
            self.data_val = self.dataset(
                data_path=self.hparams.data_dir, split='val', transform=self.transforms
            )
            logging.info(f"Validate dataset loaded, size: {len(self.data_val)}")
        # Assign test dataset for use in dataloader(s)
        if stage == "test" and not self.data_test:
            self.data_test = self.dataset(
                self.hparams.data_dir, train=False, transform=self.transforms
            )
            logging.info(f"Test dataset loaded, size: {len(self.data_test)}")

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    # def teardown(self, stage: Optional[str] = None) -> None:
    #     """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
    #     `trainer.test()`, and `trainer.predict()`.
    #
    #     :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
    #         Defaults to ``None``.
    #     """
    #     pass
    #
    # def state_dict(self) -> Dict[Any, Any]:
    #     """Called when saving a checkpoint. Implement to generate and save the datamodule state.
    #
    #     :return: A dictionary containing the datamodule state that you want to save.
    #     """
    #     return {}
    #
    # def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
    #     """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
    #     `state_dict()`.
    #
    #     :param state_dict: The datamodule state returned by `self.state_dict()`.
    #     """
    #     pass


if __name__ == "__main__":

    _ = ShapeNetDataModule()
