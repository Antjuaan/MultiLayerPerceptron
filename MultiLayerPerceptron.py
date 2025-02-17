import numpy as np
import matplotlib.pyplot as plt

# Classe del Percettrone Multistrato
class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, momentum):
        # Architettura della rete
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        # Inizializzo i neuroni
        self.input_neurons = np.zeros(self.input_size)
        self.hidden_neurons = np.zeros(self.hidden_size)
        self.output_neurons = np.zeros(self.output_size)

        # Inizializzo i pesi: input -> hidden e hidden -> output
        self.weights_IH = np.random.rand(self.input_size, self.hidden_size) - 0.5
        self.weights_HO = np.random.rand(self.hidden_size, self.output_size) - 0.5

        # Inizializzo i bias: hidden e output
        self.bias_H = np.zeros(self.hidden_size)
        self.bias_O = np.zeros(self.output_size)

        # Termini di velocità per il momentum (per l'aggiornamento dei pesi)
        self.velocity_IH = np.zeros_like(self.weights_IH)
        self.velocity_HO = np.zeros_like(self.weights_HO)
        self.velocity_H = np.zeros_like(self.bias_H)
        self.velocity_O = np.zeros_like(self.bias_O)

    # Funzione di attivazione - Sigmoide
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivata della sigmoide per la modifica dei pesi
    def sigmoid_derivative(self, x):
        return x * (1 - x)


    # Feed forward
    def feed_forward(self, sample: np.ndarray):
        # Salvo l'input che mi arriva (i quattro dati del fiore)
        self.input_neurons = sample

        # INPUT -> HIDDEN
        # Somma pesata degli input per i pesi e il bias per la soglia del layer nascosto
        self.hidden_input = np.dot(self.input_neurons, self.weights_IH) + self.bias_H

        # Applico la funzione di attivazione (sigmoide) per ottenere l'output del layer nascosto
        self.hidden_neurons = self.sigmoid(self.hidden_input)

        # HIDDEN -> OUTPUT
        # Somma pesata dei neuroni nascosti per i pesi e il bias per la soglia del layer di output
        self.output_input = np.dot(self.hidden_neurons, self.weights_HO) + self.bias_O

        # Applico la funzione di attivazione (sigmoide) per ottenere l'output finale
        self.output_neurons = self.sigmoid(self.output_input)

        # Restituisco l'output che contiene un array di 3 valori (probabilità di appartenenza a ciascuna classe)
        return self.output_neurons


    # Backpropagation
    def backpropagation(self, sample: np.ndarray, target: np.ndarray):
        # Calcolo l'errore dell'output (differenza tra output e target)
        output_error = self.output_neurons - target

        # Calcolo del delta del livello di output (derivata della sigmoide) per vedere di quanto modificare i pesi
        output_delta = output_error * self.sigmoid_derivative(self.output_neurons)

        # Calcolo dell'errore dell'hidden layer - Distribuisco l'errore indietro ai neuroni nascosti per capire 
        # quanto ciascuno di essi ha contribuito all'errore dell'output
        hidden_error = np.dot(output_delta, self.weights_HO.T)

        # Calcolo del delta del livello nascosto (derivata della sigmoide) per vedere di quanto modificare i pesi
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_neurons)


        # Aggiorno i pesi - Hidden -> Output
        self.velocity_HO = self.momentum * self.velocity_HO - self.learning_rate * np.outer(self.hidden_neurons, output_delta)
        self.weights_HO += self.velocity_HO

        # Aggiorno il bias del livello di output
        self.velocity_O = self.momentum * self.velocity_O - self.learning_rate * output_delta
        self.bias_O += self.velocity_O

        # Aggiorno i pesi - Input -> Hidden
        self.velocity_IH = self.momentum * self.velocity_IH - self.learning_rate * np.outer(sample, hidden_delta)
        self.weights_IH += self.velocity_IH

        # Aggiorno il bias del livello nascosto
        self.velocity_H = self.momentum * self.velocity_H - self.learning_rate * hidden_delta
        self.bias_H += self.velocity_H


    # Calcolo dell'errore
    def get_loss(self, target: np.ndarray, prediction: np.ndarray):
        return np.mean((target - prediction) ** 2)
    

    # Training e Test
    def train_test(self, X_train, y_train, X_test, y_test, epochs, debug=False):
        # Array per salvare le loss
        train_loss = []
        test_loss = []

        # Addestramento
        for epoch in range(epochs):

            # Riduco il learning rate (in questo caso del 5%) ad ogni epoca per addestrare la rete in modo più preciso
            self.learning_rate *= 0.95

            for i in range(len(X_train)):
                self.feed_forward(X_train[i])
                self.backpropagation(X_train[i], y_train[i])
            
            # Calcolo la loss per ogni epoca
            train_loss.append(self.get_loss(y_train, np.array([self.feed_forward(x) for x in X_train])))

            # Calcolo la loss sul test set
            test_predictions = np.array([self.feed_forward(x) for x in X_test])
            test_loss.append(self.get_loss(y_test, test_predictions))

            print(f"Epoch {epoch+1}/{epochs}: Training Loss = {train_loss[-1]:.4f}, Test Loss = {test_loss[-1]:.4f}, Learning Rate = {self.learning_rate:.5f}")

        # Risultati del Test
        print("\n--- Risultati del Test ---")
        correct_predictions = 0
        for i in range(len(X_test)):
            # Calcolo il valore predetto dalla rete
            pred = self.feed_forward(X_test[i])

            # Estraggo il valore più alto (quindi la classe predetta)
            predicted_class = np.argmax(pred)

            # Estraggo la classe reale
            actual_class = np.argmax(y_test[i])

            # Se la classe predetta è uguale a quella reale allora la predizione è corretta
            is_correct = predicted_class == actual_class
            correct_predictions += int(is_correct)

            print(f"Test {i+1}/{len(X_test)} - Predetto: {predicted_class}, Reale: {actual_class} - {'✔ CORRETTO' if is_correct else '✘ ERRATO'}")

        # Calcolo l'accuratezza in percentuale
        accuracy = (correct_predictions / len(X_test)) * 100
        print(f"\nAccuracy Finale sul Test Set: {accuracy:.2f}%")


        # Grafico Training & Test Loss
        if debug:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, epochs+1), train_loss, label="Training Loss", marker="o", linestyle="-", color="blue")
            plt.plot(range(1, epochs+1), test_loss, label="Test Loss", marker="x", linestyle="--", color="red")

            plt.title("Training vs Test Loss Over Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.show()