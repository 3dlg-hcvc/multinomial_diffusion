import torch
from diffusion_utils.loss import elbo_bpd, floor_loss
from diffusion_utils.utils import add_parent_path
import wandb
import numpy as np
import json
import os

add_parent_path(level=2)
from diffusion_utils.experiment import DiffusionExperiment
from diffusion_utils.experiment import add_exp_args as add_exp_args_parent


def add_exp_args(parser):
    add_exp_args_parent(parser)
    parser.add_argument("--clip_value", type=float, default=None)
    parser.add_argument("--clip_norm", type=float, default=None)
    parser.add_argument("--floor_loss", type=eval, default=False)


def load_json(json_path):
    """Load a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


class Experiment(DiffusionExperiment):

    def train_fn(self, epoch):
        text_condition = self.args.text_condition
        self.model.train()
        loss_sum = 0.0
        loss_count = 0
        for data in self.train_loader:
            if self.args.text_condition:
                x, floor_plan, room_type, text_condition = data
                text_condition = text_condition.to(self.args.device)
            else:
                x, floor_plan, room_type = data
            floor_plan = (
                floor_plan.to(self.args.device) if torch.sum(floor_plan) != 0 else None
            )
            room_type = (
                room_type.to(self.args.device) if torch.sum(room_type) >= 0 else None
            )
            self.optimizer.zero_grad()
            if self.args.text_condition:
                # breakpoint()
                loss_elbo = elbo_bpd(
                    self.model, x.to(self.args.device), floor_plan, room_type, text_condition
                )
            else:
                loss_elbo = elbo_bpd(
                    self.model, x.to(self.args.device), floor_plan, room_type
                )
            if self.args.floor_loss:
                loss_floor = floor_loss(
                    self.model, x.to(self.args.device), floor_plan, room_type
                )
                loss = loss_elbo + loss_floor


                loss_floor_sum = loss_floor.detach().cpu().item() * len(x)
            else:
                loss = loss_elbo
            loss.backward()
            if self.args.clip_value:
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(), self.args.clip_value
                )
            if self.args.clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.clip_norm
                )
            self.optimizer.step()
            if self.scheduler_iter:
                self.scheduler_iter.step()
            loss_sum += loss.detach().cpu().item() * len(x)
            loss_count += len(x)
            if self.args.floor_loss:
                print(
                    "Training. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}, Floor_loss: {:.3f}".format(
                        epoch + 1,
                        self.args.epochs,
                        loss_count,
                        len(self.train_loader.dataset),
                        loss_sum / loss_count,
                        loss_floor_sum / loss_count,
                    ),
                    end="\r",
                )
            else:
                print(
                    "Training. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}".format(
                        epoch + 1,
                        self.args.epochs,
                        loss_count,
                        len(self.train_loader.dataset),
                        loss_sum / loss_count,
                    ),
                    end="\r",
                )
        print("")
        if self.scheduler_epoch:
            self.scheduler_epoch.step()
        return (
            {"bpd": loss_sum / loss_count, "floor_loss": loss_floor_sum / loss_count}
            if self.args.floor_loss
            else {"bpd": loss_sum / loss_count}
        )

    def eval_fn(self, epoch):
        self.model.eval()

        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            for data in self.train_loader:
                if self.args.text_condition:
                    x, floor_plan, room_type, text_condition = data
                    text_condition = text_condition.to(self.args.device)
                else:
                    x, floor_plan, room_type = data
                floor_plan = (
                    floor_plan.to(self.args.device)
                    if torch.sum(floor_plan) != 0
                    else None
                )
                room_type = (
                    room_type.to(self.args.device)
                    if torch.sum(room_type) >= 0
                    else None
                )
                # loss = elbo_bpd(self.model, x.to(self.args.device), floor_plan, room_type)
                if self.args.text_condition:
                    # breakpoint()
                    loss_elbo = elbo_bpd(
                        self.model, x.to(self.args.device), floor_plan, room_type, text_condition
                    )
                else:
                    loss_elbo = elbo_bpd(
                        self.model, x.to(self.args.device), floor_plan, room_type
                    )
                if self.args.floor_loss:
                    loss_floor = floor_loss(
                        self.model, x.to(self.args.device), floor_plan, room_type
                    )
                    loss = loss_elbo + loss_floor

                    loss_floor_sum = loss_floor.detach().cpu().item() * len(x)
                else:
                    loss = loss_elbo
                loss_sum += loss.detach().cpu().item() * len(x)
                loss_count += len(x)
                if self.args.floor_loss:
                    # print('Train evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.train_loader.dataset), loss_sum/loss_count), end='\r')
                    print(
                        "Train evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}, Floor_loss: {:.3f}".format(
                            epoch + 1,
                            self.args.epochs,
                            loss_count,
                            len(self.train_loader.dataset),
                            loss_sum / loss_count,
                            loss_floor_sum / loss_count,
                        ),
                        end="\r",
                    )
                else:
                    print(
                        "Train evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}".format(
                            epoch + 1,
                            self.args.epochs,
                            loss_count,
                            len(self.train_loader.dataset),
                            loss_sum / loss_count,
                        ),
                        end="\r",
                    )
            print("")

        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            for data in self.train_loader:
                if self.args.text_condition:
                    x, floor_plan, room_type, text_condition = data
                    text_condition = text_condition.to(self.args.device)
                else:
                    x, floor_plan, room_type = data
                floor_plan = (
                    floor_plan.to(self.args.device)
                    if torch.sum(floor_plan) != 0
                    else None
                )
                room_type = (
                    room_type.to(self.args.device)
                    if torch.sum(room_type) >= 0
                    else None
                )
                # loss = elbo_bpd(self.model, x.to(self.args.device), floor_plan, room_type)
                if self.args.text_condition:
                    loss_elbo = elbo_bpd(
                        self.model, x.to(self.args.device), floor_plan, room_type, text_condition
                    )
                else:
                    loss_elbo = elbo_bpd(
                        self.model, x.to(self.args.device), floor_plan, room_type
                    )
                if self.args.floor_loss:
                    loss_floor = floor_loss(
                        self.model, x.to(self.args.device), floor_plan, room_type
                    )
                    loss = loss_elbo + loss_floor

                    loss_floor_sum = loss_floor.detach().cpu().item() * len(x)
                else:
                    loss = loss_elbo
                loss_sum += loss.detach().cpu().item() * len(x)
                loss_count += len(x)
                if self.args.floor_loss:
                    print(
                        "     Evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}, Floor_loss: {:.3f}".format(
                            epoch + 1,
                            self.args.epochs,
                            loss_count,
                            len(self.eval_loader.dataset),
                            loss_sum / loss_count,
                            loss_floor_sum / loss_count,
                        ),
                        end="\r",
                    )
                else:
                    print(
                        "     Evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}".format(
                            epoch + 1,
                            self.args.epochs,
                            loss_count,
                            len(self.eval_loader.dataset),
                            loss_sum / loss_count,
                        ),
                        end="\r",
                    )
            print("")
        return (
            {"bpd": loss_sum / loss_count, "floor_loss": loss_floor_sum / loss_count}
            if self.args.floor_loss
            else {"bpd": loss_sum / loss_count}
        )

    def instance_map_to_color(self, batch_instance_maps, color_map):
        batch_size, _, height, width = batch_instance_maps.shape
        assert batch_instance_maps.shape[1] == 1, "Expected a single channel for instance maps."

        # Remove the channels dimension
        batch_instance_maps = np.squeeze(batch_instance_maps, axis=1)

        # Create an empty RGB image batch with the same size as the batched instance ID maps
        batch_colored_maps = np.zeros((batch_size, height, width, 3), dtype=np.uint8)

        # Loop through each instance ID map in the batch
        for i in range(batch_size):
            instance_map = batch_instance_maps[i]

            # Fill the RGB image with colors from the mapping
            for instance_id, color in color_map.items():
                mask = (instance_map == instance_id)
                batch_colored_maps[i, mask, 0] = color[0]
                batch_colored_maps[i, mask, 1] = color[1]
                batch_colored_maps[i, mask, 2] = color[2]

        return batch_colored_maps

    def samples_process(self, batch, colormap):
        if len(batch.size()) == 3:
            batch = batch.unsqueeze(1)

        raw_layout = batch[0]
        batch = self.instance_map_to_color(batch.cpu().numpy(), colormap)
        batch_transposed = batch.transpose(0, 3, 1, 2)
        batch_tensor = torch.tensor(batch_transposed).to(torch.uint8)

        batch_tensor = batch_tensor.permute(0, 2, 3, 1)
        return batch_tensor, raw_layout

    def assign_color_rgb(self, color_palette_path, idx_to_label_path):
        """
        Assigns RGB colors to room types based on a color palette and label mapping.
        """
        color_palette = load_json(color_palette_path)
        idx_to_label = load_json(idx_to_label_path)

        # If w_arch is True, use original mapping without adjustments
        if hasattr(self.args, 'w_arch') and self.args.w_arch:
            colors = {}
            for idx, label in idx_to_label.items():
                if label in color_palette:
                    colors[int(idx)] = tuple(color_palette[label])
            return colors
        elif hasattr(self.args, 'wo_room') and self.args.wo_room:
            # Otherwise, adjust IDs by removing door and window
            door_id = None
            window_id = None
            for idx, label in idx_to_label.items():
                if label.lower() == 'door':
                    door_id = int(idx)
                elif label.lower() == 'window':
                    window_id = int(idx)

            colors = {}
            for idx, label in idx_to_label.items():
                idx = int(idx)
                if label.lower() in ['door', 'window']:
                    continue
                
                if label in color_palette:
                    # If ID is after door, decrease by 1
                    adjusted_idx = idx
                    if door_id is not None and idx > door_id:
                        adjusted_idx -= 1
                    # If ID is after window, decrease by 1 again    
                    if window_id is not None and idx > window_id:
                        adjusted_idx -= 1
                    colors[adjusted_idx] = tuple(color_palette[label])

            return colors
        else:
            colors = {}
            for idx, label in idx_to_label.items():
                if label in color_palette:
                    colors[int(idx)] = tuple(color_palette[label])
            return colors

    def log_samples_fn(self, epoch):
        """Generate and log samples to wandb."""
        self.model.eval()
        with torch.no_grad():
            # Get a batch of validation data
            data = next(iter(self.eval_loader))
            if self.args.text_condition:
                x, floor_plan, room_type, text_condition = data
                text_condition = text_condition.to(self.args.device)
            else:
                x, floor_plan, room_type = data
                text_condition = None
            
            # Prepare conditions - select only num_samples from the batch
            if floor_plan is not None and torch.sum(floor_plan) != 0:
                floor_plan = floor_plan[:self.args.num_samples].to(self.args.device)
            else:
                floor_plan = None
                
            room_type = room_type[:self.args.num_samples].to(self.args.device) if room_type is not None and torch.sum(room_type) >= 0 else None
            if text_condition is not None:
                text_condition = text_condition[:self.args.num_samples]
            
            # Generate samples using sample_chain
            samples_chain = self.model.sample_chain(
                self.args.num_samples,
                floor_plan=floor_plan,
                room_type=room_type,
                text_condition=text_condition
            )
            
            # Process samples for visualization
            samples_chain = samples_chain.permute(1, 0, 2, 3, 4)
            
            # Load the color mapping
            color_palette_path = '../preprocess/metadata/color_palette.json'
            idx_to_label_path = '../preprocess/metadata/unified_idx_to_generic_label.json'
            colormap = self.assign_color_rgb(color_palette_path, idx_to_label_path)
            
            # Create a wandb table for the samples
            columns = ["Generated Layout", "Floor Plan", "Room Type"]
            
            data = []
            # breakpoint()
            for i in range(self.args.num_samples):
                # Get final generated sample (last timestep) and convert to RGB
                final_sample = samples_chain[i][-1].unsqueeze(0)  # Add batch dimension
                colored_samples, raw_layout = self.samples_process(final_sample, colormap)
                
                # Process floor plan for visualization if available
                floor_plan_img = None
                if floor_plan is not None:
                    fp = floor_plan[i].cpu().squeeze().numpy()
                    if self.args.w_arch:
                        # Create RGB image for architectural elements
                        colored_floor_plan = np.zeros((fp.shape[0], fp.shape[1], 3), dtype=np.uint8)
                        
                        # Assign colors to different architectural elements (in BGR format for cv2, but RGB for wandb)
                        colored_floor_plan[fp == 0] = [255, 255, 255]  # Background -> White
                        colored_floor_plan[fp == 1] = [211, 211, 211]  # Floor -> Gray
                        colored_floor_plan[fp == 2] = [153, 0, 0]      # Door -> Dark Red (RGB)
                        colored_floor_plan[fp == 3] = [255, 153, 153]  # Window -> Light Red (RGB)
                        
                        floor_plan_img = colored_floor_plan
                    else:
                        # Simple binary floor plan
                        floor_plan_img = fp * 255
                
                # Get room type as a number
                room_type_value = None
                if room_type is not None:
                    room_type_value = int(room_type[i].cpu().item())
                
                # breakpoint()
                row = [
                    wandb.Image(colored_samples[0].cpu().numpy()),  # Generated sample as RGB
                    wandb.Image(floor_plan_img) if floor_plan_img is not None else None,  # Floor plan
                    room_type_value,  # Room type as a number
                ]
                data.append(row)
            
            # Log the table to wandb
            # breakpoint()
            wandb.log({
                "generated_samples": wandb.Table(data=data, columns=columns)
            }, step=epoch+1)  # Explicitly set the step parameter
