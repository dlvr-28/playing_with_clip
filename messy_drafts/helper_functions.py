import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

def fit_retrieval(epochs, model, loss_func, opt, train_dl, val_dl):
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_dl, desc=f"Epoch {epoch}", leave=False):
            image, text = batch["image"], batch["alt_text"]
            image = image.to(model.device)
            #text = text.to(model.device)
            logits = model.logits(image, text)
            loss = loss_func(logits)

            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        losses, nums = [], []
        with torch.no_grad():
            for batch in val_dl:
                image, text = batch["image"], batch["alt_text"]
                image = image.to(model.device)
                #text = text.to(model.device)
                logits = model.logits(image, text)
                loss = loss_func(logits)
                losses.append(loss.item())
                nums.append(image.size(0))

        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(f"Epoch {epoch}, val_loss {val_loss}")

def get_data(train_ds, valid_ds, bs, n_shot=None):
    if n_shot is not None:
        indices = np.random.choice(len(train_ds), size=n_shot, replace=False)
        train_ds = torch.utils.data.Subset(train_ds, indices)
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

def clip_contrastive_loss_from_logits(logits_per_image: torch.Tensor):
    """
    logits_per_image: [B, B] similarity (already scaled) where row i compares image i vs all texts
    Returns the symmetric CLIP loss using cross-entropy on both directions.
    """
    assert logits_per_image.ndim == 2 and logits_per_image.size(0) == logits_per_image.size(1), \
        "logits_per_image must be square [B, B]"

    device = logits_per_image.device
    B = logits_per_image.size(0)
    targets = torch.arange(B, device=device)

    loss_i = F.cross_entropy(logits_per_image, targets)          # image -> text
    loss_t = F.cross_entropy(logits_per_image.t(), targets)      # text  -> image
    return 0.5 * (loss_i + loss_t)