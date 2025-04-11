import torch
import numpy as np
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, TensorDataset
import torch.nn as nn
import torch.optim as optim

class MembershipInferenceAttack:
    """Enhanced implementation of a Membership Inference Attack (MIA)."""
    
    def __init__(self, target_model, shadow_model=None):
        """
        Initialize a membership inference attack.
        
        Args:
            target_model: The target model to attack
            shadow_model: Optional shadow model for training the attack model
        """
        self.target_model = target_model
        self.shadow_model = shadow_model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.attack_model = None
    
    def compute_confidence_vector(self, model, inputs, targets):
        """
        Compute a vector of features used for membership inference.
        
        Args:
            model: The model to evaluate
            inputs: Input data
            targets: Target labels
            
        Returns:
            Confidence vector features
        """
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
            
            # Extract features that might indicate membership
            # 1. Loss value
            # 2. Confidence (probability) of correct class
            # 3. Confidence (probability) of predicted class
            # 4. Entropy of output distribution
            
            correct_class_probs = probabilities[torch.arange(probabilities.size(0)), targets]
            predicted_class = torch.argmax(probabilities, dim=1)
            predicted_class_probs = probabilities[torch.arange(probabilities.size(0)), predicted_class]
            
            # Compute prediction entropy (higher entropy = more uncertainty)
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
            
            # Return as numpy array for scikit-learn compatibility
            features = torch.stack([
                loss,                 # Loss value
                correct_class_probs,  # Confidence for correct class
                predicted_class_probs,# Confidence for predicted class
                entropy,              # Prediction entropy
                (predicted_class == targets).float()  # Correctness of prediction
            ], dim=1).cpu().numpy()
            
            return features
    
    def compute_loss_values(self, model, dataloader):
        """
        Compute per-sample loss values and confidence vectors for the given model and data.
        
        Args:
            model: The model to evaluate
            dataloader: DataLoader containing samples
            
        Returns:
            List of feature vectors for each sample
        """
        model.eval()
        model.to(self.device)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        features_list = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                features = self.compute_confidence_vector(model, inputs, targets)
                features_list.append(features)
                    
        return np.concatenate(features_list, axis=0) if features_list else np.array([])
    
    def train_attack_model(self, member_features, non_member_features):
        """
        Train a meta-classifier to distinguish between members and non-members.
        
        Args:
            member_features: Features from training data (members)
            non_member_features: Features from test data (non-members)
            
        Returns:
            Trained attack model
        """
        # Combine features and create labels
        X = np.vstack([member_features, non_member_features])
        y = np.concatenate([np.ones(len(member_features)), np.zeros(len(non_member_features))])
        
        # Create a simple meta-classifier (you could use any model here)
        class AttackModel(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                return self.model(x)
        
        # Train on 80% of the data, validate on 20%
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert to torch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        
        # Create datasets and data loaders
        train_dataset = TensorDataset(X_train, y_train.unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Create attack model
        attack_model = AttackModel(X_train.shape[1])
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
        
        # Train the model
        n_epochs = 50
        best_val_auc = 0
        best_model_state = None
        
        for epoch in range(n_epochs):
            attack_model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = attack_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            # Evaluate on validation set
            attack_model.eval()
            with torch.no_grad():
                val_outputs = attack_model(X_val)
                val_auc = roc_auc_score(y_val, val_outputs.squeeze().numpy())
                
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = {k: v.clone() for k, v in attack_model.state_dict().items()}
                
        # Load best model
        attack_model.load_state_dict(best_model_state)
        return attack_model
            
    def advanced_attack(self, member_data, non_member_data, test_member=None, test_non_member=None):
        """
        Perform an advanced membership inference attack using a meta-classifier.
        
        Args:
            member_data: DataLoader containing member samples
            non_member_data: DataLoader containing non-member samples
            test_member: Optional test set of known members
            test_non_member: Optional test set of known non-members
            
        Returns:
            Dict containing attack metrics
        """
        # Compute features for members and non-members
        member_features = self.compute_loss_values(self.target_model, member_data)
        non_member_features = self.compute_loss_values(self.target_model, non_member_data)
        
        # Train attack model
        attack_model = self.train_attack_model(member_features, non_member_features)
        self.attack_model = attack_model
        
        # Evaluate on training data
        X = np.vstack([member_features, non_member_features])
        y_true = np.concatenate([np.ones(len(member_features)), np.zeros(len(non_member_features))])
        
        # Get predictions
        attack_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            y_pred_proba = attack_model(X_tensor).squeeze().numpy()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
        # Calculate metrics
        train_accuracy = accuracy_score(y_true, y_pred)
        train_precision = precision_score(y_true, y_pred)
        train_recall = recall_score(y_true, y_pred)
        train_f1 = f1_score(y_true, y_pred)
        train_auc = roc_auc_score(y_true, y_pred_proba)
        
        logging.info(f"Training metrics - Accuracy: {train_accuracy}, AUC: {train_auc}")
        
        # If test sets are provided, evaluate attack on them
        if test_member is not None and test_non_member is not None:
            test_member_features = self.compute_loss_values(self.target_model, test_member)
            test_non_member_features = self.compute_loss_values(self.target_model, test_non_member)
            
            X_test = np.vstack([test_member_features, test_non_member_features])
            y_test_true = np.concatenate([np.ones(len(test_member_features)), np.zeros(len(test_non_member_features))])
            
            # Get predictions
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test)
                y_test_pred_proba = attack_model(X_test_tensor).squeeze().numpy()
                y_test_pred = (y_test_pred_proba > 0.5).astype(int)
                
            # Calculate metrics
            test_accuracy = accuracy_score(y_test_true, y_test_pred)
            test_precision = precision_score(y_test_true, y_test_pred)
            test_recall = recall_score(y_test_true, y_test_pred)
            test_f1 = f1_score(y_test_true, y_test_pred)
            test_auc = roc_auc_score(y_test_true, y_test_pred_proba)
            
            logging.info(f"Test metrics - Accuracy: {test_accuracy}, AUC: {test_auc}")
            
            return {
                'train_accuracy': train_accuracy,
                'train_precision': train_precision, 
                'train_recall': train_recall,
                'train_f1': train_f1,
                'train_auc': train_auc,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_auc': test_auc
            }
        
        return {
            'train_accuracy': train_accuracy,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'train_auc': train_auc
        }
    
    def threshold_attack(self, member_data, non_member_data, test_member=None, test_non_member=None):
        """
        Perform a threshold-based membership inference attack.
        
        Args:
            member_data: DataLoader containing member samples
            non_member_data: DataLoader containing non-member samples
            test_member: Optional test set of known members
            test_non_member: Optional test set of known non-members
            
        Returns:
            Dict containing attack metrics
        """
        # For backward compatibility, extract just the loss feature
        member_features = self.compute_loss_values(self.target_model, member_data)
        non_member_features = self.compute_loss_values(self.target_model, non_member_data)
        
        member_losses = member_features[:, 0]  # Loss is the first feature
        non_member_losses = non_member_features[:, 0]
        
        # Create labels: 1 for members, 0 for non-members
        member_labels = [1] * len(member_losses)
        non_member_labels = [0] * len(non_member_losses)
        
        # Combine data for finding optimal threshold
        train_losses = np.concatenate([member_losses, non_member_losses])
        train_labels = np.concatenate([member_labels, non_member_labels])
        
        # Find optimal threshold that maximizes accuracy
        sorted_losses = np.sort(np.unique(train_losses))
        best_threshold = None
        best_accuracy = 0.0
        
        for threshold in sorted_losses:
            predictions = (train_losses <= threshold).astype(int)
            acc = accuracy_score(train_labels, predictions)
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_threshold = threshold
        
        logging.info(f"Best threshold: {best_threshold}, Training accuracy: {best_accuracy}")
        
        # Calculate AUC for ROC curve
        train_auc = roc_auc_score(train_labels, -train_losses)  # Negative because lower loss = more likely member
        
        # If test sets are provided, evaluate attack on them
        if test_member is not None and test_non_member is not None:
            test_member_features = self.compute_loss_values(self.target_model, test_member)
            test_non_member_features = self.compute_loss_values(self.target_model, test_non_member)
            
            test_member_losses = test_member_features[:, 0]
            test_non_member_losses = test_non_member_features[:, 0]
            
            test_losses = np.concatenate([test_member_losses, test_non_member_losses])
            test_labels = np.concatenate([[1] * len(test_member_losses), [0] * len(test_non_member_losses)])
            test_preds = (test_losses <= best_threshold).astype(int)
            
            test_accuracy = accuracy_score(test_labels, test_preds)
            test_precision = precision_score(test_labels, test_preds)
            test_recall = recall_score(test_labels, test_preds)
            test_f1 = f1_score(test_labels, test_preds)
            test_auc = roc_auc_score(test_labels, -test_losses)
            
            logging.info(f"Test metrics - Accuracy: {test_accuracy}, AUC: {test_auc}")
            
            return {
                'threshold': best_threshold,
                'train_accuracy': best_accuracy,
                'train_auc': train_auc,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_auc': test_auc
            }
        
        return {
            'threshold': best_threshold,
            'train_accuracy': best_accuracy,
            'train_auc': train_auc
        }
    
    def evaluate_model_privacy(self, train_data, test_data, k_fold=5, attack_type='advanced'):
        """
        Evaluate the privacy of a model using k-fold cross-validation.
        
        Args:
            train_data: The training dataset
            test_data: The test dataset
            k_fold: Number of folds for cross-validation
            attack_type: Type of attack to perform ('threshold' or 'advanced')
            
        Returns:
            Dict containing average attack metrics
        """
        # Create indices for the k-fold splits
        train_indices = list(range(len(train_data)))
        test_indices = list(range(len(test_data)))
        
        # Use stratified k-fold to ensure balanced data
        skf_train = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
        # Create fake labels for stratification (all 0s)
        y_fake_train = np.zeros(len(train_indices))
        
        skf_test = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
        y_fake_test = np.zeros(len(test_indices))
        
        metrics = []
        
        for (train_fold_indices, _), (test_fold_indices, _) in zip(
            skf_train.split(train_indices, y_fake_train), 
            skf_test.split(test_indices, y_fake_test)
        ):
            logging.info(f"Evaluating fold {len(metrics)+1}/{k_fold}")
            
            # Create member samples (subset of training data)
            member_indices = [train_indices[i] for i in train_fold_indices[:len(train_fold_indices)//2]]
            member_subset = Subset(train_data, member_indices)
            member_loader = DataLoader(member_subset, batch_size=64, shuffle=False)
            
            # Create non-member samples (subset of test data)
            non_member_indices = [test_indices[i] for i in test_fold_indices[:len(test_fold_indices)//2]]
            non_member_subset = Subset(test_data, non_member_indices)
            non_member_loader = DataLoader(non_member_subset, batch_size=64, shuffle=False)
            
            # Create validation sets from remaining data
            val_member_indices = [train_indices[i] for i in train_fold_indices[len(train_fold_indices)//2:]]
            val_member_subset = Subset(train_data, val_member_indices)
            val_member_loader = DataLoader(val_member_subset, batch_size=64, shuffle=False)
            
            val_non_member_indices = [test_indices[i] for i in test_fold_indices[len(test_fold_indices)//2:]]
            val_non_member_subset = Subset(test_data, val_non_member_indices)
            val_non_member_loader = DataLoader(val_non_member_subset, batch_size=64, shuffle=False)
            
            # Perform attack based on selected type
            if attack_type == 'advanced':
                fold_metrics = self.advanced_attack(
                    member_loader, 
                    non_member_loader,
                    val_member_loader,
                    val_non_member_loader
                )
            else:
                fold_metrics = self.threshold_attack(
                    member_loader, 
                    non_member_loader,
                    val_member_loader,
                    val_non_member_loader
                )
            
            metrics.append(fold_metrics)
        
        # Calculate average metrics
        avg_metrics = {}
        for key in metrics[0].keys():
            avg_metrics[key] = np.mean([metric.get(key, 0) for metric in metrics])
        
        logging.info(f"Average attack metrics: {avg_metrics}")
        
        return avg_metrics 