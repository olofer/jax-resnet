import numpy as np
from dataset_batcher import IterableDataset
import time


def dataset_loss(modelparams, dataset: IterableDataset, lossfcn: callable):
    nb = len(dataset)
    losses = np.tile(np.nan, (nb,))
    samples = np.tile(np.nan, (nb,))
    for b, XYb in enumerate(dataset):
        assert len(XYb) >= 2, "expected X, y, .."
        assert XYb[0].shape[0] == XYb[1].shape[0]
        samples[b] = XYb[0].shape[0]
        if len(XYb) == 2:
            losses[b] = lossfcn(modelparams, XYb[0], XYb[1])
        else:
            assert len(XYb) == 3, "expected X, y, w"
            assert XYb[0].shape[0] == XYb[2].shape[0]
            losses[b] = lossfcn(modelparams, XYb[0], XYb[1], XYb[2])

    return np.sum(losses * samples) / np.sum(samples)


def update_one_epoch(
    modelparams, dataset: IterableDataset, trainparams: dict, updatefcn: callable
):
    for b, XYb in enumerate(dataset):
        assert len(XYb) == 2 or len(XYb) == 3, "expected (X,y) or (X,y,w)"
        assert XYb[0].shape[0] == XYb[1].shape[0]

        if trainparams["bprint"] > 0 and b % trainparams["bprint"] == 0:
            print(
                "batch %i -- features are %i-by-%i"
                % (b, XYb[0].shape[0], XYb[0].shape[1])
            )

        if len(XYb) == 2:
            modelparams = updatefcn(
                modelparams,
                XYb[0],
                XYb[1],
                trainparams["step_size"],
                trainparams["weight_decay"] / len(dataset),
            )
        else:
            modelparams = updatefcn(
                modelparams,
                XYb[0],
                XYb[1],
                XYb[2],
                trainparams["step_size"],
                trainparams["weight_decay"] / len(dataset),
            )

    return modelparams


def update_many_epochs(
    modelparams,
    train_data: IterableDataset,
    trainparams: dict,
    updatefcn: callable,
    lossfcn: callable = None,
    test_data: IterableDataset = None,
):
    train_losses = []
    test_losses = []

    for epoch in range(trainparams["epochs"]):
        if trainparams["shuffle"]:
            train_data.shuffle()

        start_time = time.time()

        modelparams = update_one_epoch(modelparams, train_data, trainparams, updatefcn)

        epoch_time = time.time() - start_time

        train_loss, test_loss = np.nan, np.nan

        if not lossfcn is None:
            train_loss = dataset_loss(modelparams, train_data, lossfcn)
            if not test_data is None:
                test_loss = dataset_loss(modelparams, test_data, lossfcn)

        train_losses.append(float(train_loss))
        test_losses.append(float(test_loss))

        if epoch == 0 or epoch % trainparams["eprint"] == 0:
            print("epoch {} in {:0.2f} sec".format(epoch, epoch_time))
            if not lossfcn is None:
                print("training/testing losses {} / {}".format(train_loss, test_loss))

    return modelparams, {"train": train_losses, "test": test_losses}
