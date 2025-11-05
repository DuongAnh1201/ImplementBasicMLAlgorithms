import numpy as np
import matplotlib.pyplot as plt
from typing import List
def Batch_Gradient_Descent(o, x, y, learning_rate = 0.05):
    hypothesis = np.dot(x,o) 
    error = y - hypothesis
    gradient = np.dot(np.transpose(x),error)
    o = o + learning_rate*gradient
    loss = 0.5*np.sum(error**2)
    return hypothesis,o,loss

def Stochastic_Gradient_Descent(o,x,y, learning_rate = 0.05):  #For input 2d matrix only.
    hypothesis = np.dot(x,o) 
    error = y - hypothesis
    losses = []
    for i in range(len(o)):
        for e in range(len(error)):
            o[i] = o[i] + learning_rate*x[e][i]*error[e]
        loss = 0.5*np.sum((y - hypothesis)**2)  
        # assuming MSE as your loss function
        losses.append(loss)
    return hypothesis, o


def main():
    x = np.array([[1, 2], [3, 1]])  # 2 samples, 2 features
    y = np.array([5, 6])             # 2 target values
    o = np.array([2, 1])             # Initial parameters (theta)
    losses = []
    epochs = []
    print("Initial parameters:", o)
    print("-" * 40)
    
    for i in range(10):
        hypothesis, o, l = Batch_Gradient_Descent(o, x, y, learning_rate=0.05)
        losses.append(l)
        print(f"Iteration {i+1}:")
        epochs.append(i+1)
        print(f"  Hypothesis: {hypothesis}")
        print(f"  Error: {y - hypothesis}")
        print(f"  Loss: {l}")
        print(f"  New Parameters: {o}")
        print("\n")
    #plot the loss curve
    plt.figure(figsize=(10,6))
    plt.plot(epochs, losses, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (LSE)', fontsize=12)
    plt.title('Training Loss over Epochs (Least Square Error)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(f"\nFinal Loss: {losses[-1]:.6f}")
    print(f"Final Parameters: {o}")
    

if __name__ == "__main__":
    main()

    


