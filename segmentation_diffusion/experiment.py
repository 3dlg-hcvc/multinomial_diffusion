import torch
from diffusion_utils.loss import elbo_bpd, floor_loss
from diffusion_utils.utils import add_parent_path

add_parent_path(level=2)
from diffusion_utils.experiment import DiffusionExperiment
from diffusion_utils.experiment import add_exp_args as add_exp_args_parent


def add_exp_args(parser):
    add_exp_args_parent(parser)
    parser.add_argument("--clip_value", type=float, default=None)
    parser.add_argument("--clip_norm", type=float, default=None)
    parser.add_argument("--floor_loss", type=eval, default=False)


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
