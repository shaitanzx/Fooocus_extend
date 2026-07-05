from abc import ABC, abstractmethod


class BaseModelProcessor(ABC):
    """
    Base abstract class for model-based processors that provides common
    functionality for managing PyTorch models.

    Args:
        model (`nn.Module`): The torch model to use.
    """

    def __init__(self, model):
        self.model = model

    def to(self, device):
        """
        Moves the underlying model to the specified device
        (e.g., CPU or GPU).

        Args:
            device (`torch.device`): The target device.

        Returns:
            `BaseModelProcessor`: The processor object itself (for method chaining).
        """
        self.model = self.model.to(device)
        return self

    @abstractmethod
    def from_pretrained(self):
        """
        This abstract method defines how the processor loads pre-trained
        weights or configurations specific to the model it supports. Subclasses
        must implement this method to handle model-specific loading logic.

        This method might download pre-trained weights from a repository or
        load them from a local file depending on the model's requirements.
        """
        pass
