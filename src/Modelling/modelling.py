"""Create and train a model."""

import torch
import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from transformers import CamembertForTokenClassification, AdamW
from tqdm import trange
from datetime import datetime

logger = logging.getLogger('main_logger')

class Model:
    """Create a model class for the finetuning and training of the model."""

    def __init__(self, train_dataloader, val_dataloader, conf):
        self.conf = conf
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = CamembertForTokenClassification.from_pretrained(
            self.conf["model"]["model_name"],
            num_labels=self.conf["model"]["num_labels"],
            output_attentions=self.conf["model"]["output_attentions"],
            output_hidden_states=self.conf["model"]["output_hidden_states"]
        )
        self.validation_f1score_values = []
        self.f1score_values = []
        self.loss_values = []
        self.validation_loss_values = []

    def finetuning(self):
        """Create optimization parameters.
        Args:
            model parameters: the parameters of the model chosen.
        Returns:
            Grouped Parameters to optimize.
        """
        if self.conf["model"]["full_fine_tuning"]:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': self.conf["model"]["weight_decay_rate_1"]},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': self.conf["model"]["weight_decay_rate_2"]}
            ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        return optimizer_grouped_parameters

    def set_optimizer(self):
        """Set the optimizer for the model.
        Args:
            None
        Returns:
            Optimizer.
        """
        optimizer = AdamW(
            self.finetuning(),
            lr=self.conf["model"]["learning_rate"],
            eps=self.conf["model"]["eps"]
        )
        return optimizer

    def set_scheduler(self, optimizer):
        """Set the scheduler to adapt the learning rate throughout the epochs.
        Args:
            train_dataloader: data loader.
            optimizer: optimizer chosen for the model.
        Returns:
            Scheduler.
        """
        epochs = self.conf["model"]["epochs"]
        max_grad_norm = self.conf["model"]["max_grad_norm"]

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(self.train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.conf["model"]["num_warmup_steps"],
            num_training_steps=total_steps
        )
        return scheduler

    def train(self):
        """Train the model and print accuracy/f1 score of train and val sets.
        Args:
            optimizer: optimizer defined in the class.
            scheduler: scheduler defined in the class.
            train_dataloader: training set dataloader.
            val_dataloader: validation set dataloader.
        Returns:
            Trained model and outputs accuracy/f1 score.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        torch.cuda.get_device_name(0)
        self.model.cuda()
        ## Store the average loss after each epoch so we can plot them.
        tag_values = self.conf["model"]["tag_values"]

        # Initialize the optimizer and scheduler
        optimizer = self.set_optimizer()
        scheduler = self.set_scheduler(optimizer)

        for ep in trange(self.conf["model"]["epochs"], desc="Epoch"):
            # ========================================
            #               Training
            # ========================================
            # Perform one full pass over the training set.

            # Put the model into training mode.
            self.model.train()
            # Reset the total loss for this epoch.
            total_loss, train_f1_score = 0, 0
            predictions_train, true_labels_train = [], []
            # Training loop
            for step, batch in enumerate(self.train_dataloader):
                # add batch to gpu
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                # Always clear any previously calculated gradients before performing a backward pass.
                self.model.zero_grad()
                # forward pass
                # This will return the loss (rather than the model output)
                # because we have provided the `labels`.
                outputs = self.model(b_input_ids, token_type_ids=None,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)
                # get the loss
                loss = outputs[0]
                logits = outputs[1].detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                predictions_train.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels_train.extend(label_ids)
                # Perform a backward pass to calculate the gradients.
                loss.backward()
                # track train loss
                total_loss += loss.item()
                # Clip the norm of the gradient
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                               max_norm=self.conf["model"]["max_grad_norm"])
                # update parameters
                optimizer.step()
                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(self.train_dataloader)
            print("Average train loss: {}".format(avg_train_loss))

            pred_tags = [tag_values[p_i] for p, l in zip(predictions_train, true_labels_train)
                                         for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
            train_tags = [tag_values[l_i] for l in true_labels_train
                                          for l_i in l if tag_values[l_i] != "PAD"]

            train_f1_score = f1_score(pred_tags, train_tags, average=self.conf["model"]["f1_metric"])
            print("F1-Score: {}".format(train_f1_score))

            # Store the loss value for plotting the learning curve.
            self.loss_values.append(avg_train_loss)
            self.f1score_values.append(train_f1_score)

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            # Put the model into evaluation mode
            self.model.eval()
            # Reset the validation loss for this epoch.
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            predictions, true_labels = [], []
            for batch in self.val_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                # Telling the model not to compute or store gradients,
                # saving memory and speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    # This will return the logits rather than the loss because we have not provided labels.
                    outputs = self.model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask, labels=b_labels)
                # Move logits and labels to CPU
                logits = outputs[1].detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences.
                eval_loss += outputs[0].mean().item()
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.extend(label_ids)
            eval_loss = eval_loss / len(self.val_dataloader)
            self.validation_loss_values.append(eval_loss)
            print("Validation loss: {}".format(eval_loss))

            pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                         for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
            valid_tags = [tag_values[l_i] for l in true_labels
                                          for l_i in l if tag_values[l_i] != "PAD"]
            validation_f1_score = f1_score(pred_tags, valid_tags, average=self.conf["model"]["f1_metric"])
            print("Validation F1-Score: {}".format(validation_f1_score))
            self.validation_f1score_values.append(validation_f1_score)
        return None

    def save_model(self):
        """Save the model.
        Args:
            Trained model.
        Returns:
            None.
        """
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y-%H:%M")
        filename = self.conf["paths"]["Outputs_path"]+self.conf["paths"]["folder_models"] + self.conf["model"]["model_name"] + dt_string + '.sav'
        pickle.dump(self.model, open(filename, 'wb'))
        logger.info('Modele sauvergardé: ' + filename)
        return None

    def evaluation(self):
        """Evaluate the model.
        Args:
            Trained model.
        Returns:
            Saves graphs for loss and F1 on train and validation set.
        """
        # Use plot styling from seaborn.
        sns.set(style='darkgrid')
        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12, 6)
        # Plot the learning curve.
        plt.plot(self.loss_values, 'b-o', label="training loss")
        plt.plot(self.validation_loss_values, 'r-o', label="validation loss")
        # Label the plot.
        plt.title("Learning curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.conf["paths"]["Outputs_path"] + self.conf["paths"]["folder_evaluations"] + self.conf["paths"]["evaluation_loss_file"])
        plt.show()
        plt.clf()
        
        # Use plot styling from seaborn.
        sns.set(style='darkgrid')
        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12, 6)
        # Plot the learning curve.
        plt.plot(self.f1score_values, 'b-o', label="F1 score")
        plt.plot(self.validation_f1score_values, 'r-o', label="validation F1 score")
        # Label the plot.
        plt.title("Learning curve")
        plt.xlabel("Epoch")
        plt.ylabel("F1 score")
        plt.legend()
        plt.savefig(self.conf["paths"]["Outputs_path"] + self.conf["paths"]["folder_evaluations"] + self.conf["paths"]["evaluation_f1_file"])
        plt.show()
        plt.clf