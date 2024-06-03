import torch


class LossUsingCrossEntropyGradientNorm:
    def __init__(self):
        self._softmax = torch.nn.Softmax()

    def _cross_entropy_loss_gradient_norm(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = self._softmax(input).detach()
        targets = torch.zeros_like(probabilities)
        for i in range(probabilities.shape[0]):
            targets[i, target[i].long()] = 1
        loss_gradient_norm = (targets - probabilities.float()).norm(dim=1)
        return loss_gradient_norm


class DropInstancesWithTopValues:
    def __init__(self, dropped_fraction: float, loss_func, drop_largest=True):
        self.dropped_fraction = dropped_fraction
        self.drop_largest = drop_largest
        self.loss_func = loss_func
        self.selected_indices = None

    def mean_loss_of_kept_instances(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        value_to_filter_on: torch.Tensor,
    ) -> torch.Tensor:
        number_of_kept_examples = input.shape[0] - int(input.shape[0] * self.dropped_fraction)
        _, self.selected_indices = torch.topk(
            value_to_filter_on, number_of_kept_examples, axis=0, largest=not self.drop_largest
        )
        selected_instances_loss = self.loss_func(input[self.selected_indices], target[self.selected_indices])
        return selected_instances_loss.mean()
