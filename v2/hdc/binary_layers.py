import numpy as np

def ste_sign(x):
    """
    Straight-Through Estimator (STE) pour la fonction Sign.
    Forward : sign(x)
    Backward : identité (gradient passe tel quel)
    """
    return np.sign(x)

class BinaryLinear:
    """
    Couche Linéaire Binaire avec poids latents float32.
    """
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        
        # Poids latents en haute précision
        limit = np.sqrt(6 / (in_features + out_features))
        self.weights_latent = np.random.uniform(-limit, limit, (out_features, in_features)).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)
        
    def forward(self, x):
        # Binarisation des poids pour le forward pass
        # W_b = sign(W - mean(W))
        w_mean = np.mean(self.weights_latent)
        self.w_bin = np.sign(self.weights_latent - w_mean)
        self.w_bin[self.w_bin == 0] = 1 # Pas de zéro
        
        # Inférence binaire : pure addition/soustraction (approximée par dot en NumPy)
        return np.dot(x, self.w_bin.T) + self.bias

    def backward(self, x, grad_output, lr=0.01):
        """
        Calcul du gradient via STE et mise à jour des poids latents.
        """
        # Gradient par rapport aux poids
        grad_w = np.outer(grad_output, x)
        grad_b = grad_output
        
        # Gradient Clipping (Global L2)
        norm = np.sqrt(np.sum(grad_w**2))
        if norm > 1.0:
            grad_w = grad_w / norm
        
        # Mise à jour des poids latents
        self.weights_latent -= lr * grad_w
        self.bias -= lr * (grad_b / (np.linalg.norm(grad_b) + 1e-8))
        
        # Gradient par rapport à l'entrée (pour la couche précédente)
        grad_input = np.dot(grad_output, self.w_bin)
        return grad_input

class BinaryMLP:
    """
    Simple MLP Binaire (Raisonnement)
    """
    def __init__(self, dim, hidden_dim):
        self.l1 = BinaryLinear(dim, hidden_dim)
        self.l2 = BinaryLinear(hidden_dim, dim)
        
    def forward(self, x):
        self.x1 = self.l1.forward(x)
        self.h1 = np.tanh(self.x1) # Activation non-linéaire float pour le prototype
        self.x2 = self.l2.forward(self.h1)
        return self.x2

    def train_step(self, x, target_hv, lr=0.01):
        # Forward
        out = self.forward(x)
        
        # Loss simple (MSE sur HVs)
        loss = np.mean((out - target_hv)**2)
        grad = 2 * (out - target_hv) / out.size
        
        # Backward
        grad_h1 = self.l2.backward(self.h1, grad, lr)
        # Gradient tanh : (1 - tanh^2)
        grad_x1 = grad_h1 * (1 - self.h1**2)
        self.l1.backward(x, grad_x1, lr)
        
        return loss
