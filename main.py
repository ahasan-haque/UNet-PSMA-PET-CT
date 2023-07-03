import torch
import glob
from monai.config import print_config
from monai.data import (
    ThreadDataLoader,
    Dataset,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import AsDiscrete

from   ipywidgets import interactive, fixed
import matplotlib.pyplot as plt
import numpy as np
import os
from   pathlib import Path
from   tqdm.notebook import tqdm


import os
import tempfile
os.environ['MONAI_DATA_DIRECTORY'] = 'monai_data_directory'

directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

num_samples = 4

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
)



def get_train_transforms(num_samples, device):
    train_transforms = Compose(
        [
            # https://docs.monai.io/en/stable/transforms.html#loadimage
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-4177.8423,
                a_max=4362.1372,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            # Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Spacingd(
            #     keys=["image", "label"],
            #     pixdim=(1.5, 1.5, 2.0),
            #     mode=("bilinear", "nearest"),
            # ),
            # EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=0.001,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            # RandFlipd(
            #     keys=["image", "label"],
            #     spatial_axis=[0],
            #     prob=0.10,
            # ),
            # RandFlipd(
            #     keys=["image", "label"],
            #     spatial_axis=[1],
            #     prob=0.10,
            # ),
            # RandFlipd(
            #     keys=["image", "label"],
            #     spatial_axis=[2],
            #     prob=0.10,
            # ),
            # RandRotate90d(
            #     keys=["image", "label"],
            #     prob=0.10,
            #     max_k=3,
            # ),
            # RandShiftIntensityd(
            #     keys=["image"],
            #     offsets=0.10,
            #     prob=0.50,
            # ),
        ]
    )

    return train_transforms



def get_val_transforms(device):
    val_transforms = Compose(
        [
            # https://docs.monai.io/en/stable/transforms.html#loadimage
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"], a_min=-4177.8423, a_max=4362.1372, b_min=0.0, b_max=1.0, clip=True
            ),
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            # Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Spacingd(
            #     keys=["image", "label"],
            #     pixdim=(1.5, 1.5, 2.0),
            #     mode=("bilinear", "nearest"),
            # ),
            # EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
        ]
    )

    return val_transforms


train_transforms = get_train_transforms(num_samples=num_samples, device=device)
val_transforms = get_val_transforms(device=device)

non_test_image_dir = 'large-dataset-mar2023/imagesTr'
non_test_label_dir = 'large-dataset-mar2023/labelsTr'

test_image_dir = 'large-dataset-mar2023/imagesTs'
test_label_dir = 'large-dataset-mar2023/labelsTs'

train_image_paths = sorted(glob.glob(os.path.join(non_test_image_dir, '*0000.nii.gz')))[:300]
train_label_paths = sorted(glob.glob(os.path.join(non_test_label_dir, '*.nii.gz')))[:300]

val_image_paths = sorted(glob.glob(os.path.join(non_test_image_dir, '*0000.nii.gz')))[300:]
val_label_paths = sorted(glob.glob(os.path.join(non_test_label_dir, '*.nii.gz')))[300:]

test_image_paths = sorted(glob.glob(os.path.join(test_image_dir, '*0000.nii.gz')))
test_label_paths = sorted(glob.glob(os.path.join(test_label_dir, '*.nii.gz')))


input_data_train = [
    {
        "image": image_path,
        "label": label_path
    }
for image_path, label_path in zip(train_image_paths, train_label_paths)]

input_data_valid = [
    {
        "image": image_path,
        "label": label_path
    }
for image_path, label_path in zip(val_image_paths, val_label_paths)]


input_data_test = [
    {
        "image": image_path,
        "label": label_path
    }
for image_path, label_path in zip(test_image_paths, test_label_paths)]

print("data path loaded")
train_ds = CacheDataset(
    data        = input_data_train,
    transform   = train_transforms,
    cache_num   = 8,
    cache_rate  = 1.0,
    num_workers = 8 # 8
)
print("after train_ds")
val_ds = CacheDataset(
    data       = input_data_valid,
    transform  = val_transforms,
    cache_num  = 8,
    cache_rate = 1.0,
    num_workers= 4 # 4
)
print("after val_ds")

train_loader = ThreadDataLoader(
    train_ds,
    num_workers=0,
    batch_size=1,
    shuffle=True
)
print("after train_loader")

val_loader = ThreadDataLoader(
    val_ds,
    num_workers=0,
    batch_size=1
)
print("after val_loader")

print("all the loader loaded")
#set_track_meta(False)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm='batch'
).to(device)

print("model loaded")
'''
current_weights    = model.state_dict()
pretrained_weights = torch.load("pretrained_models/pretrained_unet.pt")

print("pretrained model loaded")
# 1. select layers which shall have pretrained weights
selected_keys = list(current_weights.keys())[:-10]

# 2. remove unwanted layers from pretrained weights
selected_pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in selected_keys}

# 3. overwrite selected layers in the existing set of weights
current_weights.update(selected_pretrained_weights)

# 4. load the overwritten weights
model.load_state_dict(current_weights)

print("Using pretrained UNet weights !")
'''
torch.backends.cudnn.benchmark = True

loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True)
optimizer     = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Scale network loss by a factor to prevent underflow, i.e. float16 gradient values flushing to zero
scaler = torch.cuda.amp.GradScaler()

label_to_onehot   = AsDiscrete(to_onehot=2)
pred_to_onehot    = AsDiscrete(argmax=True, to_onehot=2)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)


def validation(epoch_iterator_val, label_to_onehot, pred_to_onehot, valid_metric_buffer):
    # Set the module in evaluation mode
    model.eval()

    # Disable gradient calculation, reduces memory consumption
    with torch.no_grad():

        # Loop over validation batches
        for step, batch in enumerate(epoch_iterator_val):

            # Get images and labels
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())

            # Sliding window inference
            # The model is trained for input_size=(96,96,96). However, real medical images are typically larger
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(
                    inputs        = val_inputs,
                    roi_size      = (96, 96, 96),
                    sw_batch_size = 4,
                    predictor     = model)

            # Decollate label batch: BxCxWxHxD -> [CxWxHxD, CxWxHxD, ...] (length of list = B)
            # In other words: turn a batch of samples (i.e. a single vector) into a list of samples (i.e. multiple vectors)
            val_labels_list = decollate_batch(val_labels)

            # Create one-hot-encoded label: 1xWxHxD -> CxWxHxD (here C = 14)
            val_labels_convert = [label_to_onehot(val_label_tensor) for val_label_tensor in val_labels_list]

            # Decollate prediction batch
            val_outputs_list = decollate_batch(val_outputs)

            # Create one-hot-encoded predictions: CxWxHxD -> CxWxHxD (only one channel C will contain 1.0 values, all other channels are )
            val_output_convert = [pred_to_onehot(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            # Evaluate predictions of a single batch using dice metric
            # Results will be buffered in a DiceMetric instance, i.e. in <dice_metric>
            valid_metric_buffer(y_pred=val_output_convert, y=val_labels_convert)
            # Update progress bar
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)"
                % (global_step, 10.0)
            )

        # Aggregate all buffered results
        mean_dice_val = valid_metric_buffer.aggregate().item()

        # Reset metric buffer
        dice_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, eval_num, dice_val_best, label_to_onehot, pred_to_onehot, valid_metric_buffer, global_step_best, epoch_loss_values, metric_values):
    # Set the module in training mode. This has any effect only on certain modules, e.g. Dropout, BatchNorm, etc
    model.train()

    # Start training at step = 0, with loss = 0
    epoch_loss = 0
    step       = 0

    # Create progress bar for training dataloader
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True, position=0, leave=True
    )

    # Loop over training batches
    for step, batch in enumerate(epoch_iterator):
        step += 1

        # Get images and labels
        x, y = (batch["image"].cuda(), batch["label"].cuda())

        # Forward pass and calculate loss
        with torch.cuda.amp.autocast():
            logit_map = model(x)
            loss      = loss_function(logit_map, y)

        # Backpropagation
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)

        # Update optimizer
        scaler.step(optimizer)
        scaler.update()

        # Reset gradients
        optimizer.zero_grad()

        # Update progress bar
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)"
            % (global_step, max_iterations, loss)
        )

        # Check it is time to perform a validation
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:

            # Create progress bar for validation dataloader
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps)", dynamic_ncols=True, position=1, leave=True
            )

            # Get here: dice score), i.e. perform validation
            dice_val = validation(epoch_iterator_val, label_to_onehot, pred_to_onehot, valid_metric_buffer)

            # Calculate epoch loss
            epoch_loss /= step

            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)

            # Always save the weights which yield the best validation loss
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(), os.path.join(root_dir, "best_metric_unet.pth")
                )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )

        global_step += 1

    return global_step, dice_val_best, global_step_best, epoch_loss_values, metric_values

max_iterations    = 200
eval_num          = 20

global_step       = 0
dice_val_best     = 0.0
global_step_best  = 0

# Training


while global_step < max_iterations:
    global_step, dice_val_best, global_step_best, epoch_loss_values, metric_values = train(
        global_step         = global_step,
        train_loader        = train_loader,
        eval_num            = eval_num,
        label_to_onehot     = label_to_onehot,
        pred_to_onehot      = pred_to_onehot,
        valid_metric_buffer = dice_metric,
        dice_val_best       = dice_val_best,
        global_step_best    = global_step_best,
        epoch_loss_values   = [],
        metric_values       = []
    )

print(
    f"train completed, best_metric: {dice_val_best:.4f} "
    f"at iteration: {global_step_best}"
)
