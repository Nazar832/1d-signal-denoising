import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

signal_length = 512
t = np.linspace(0, 1, signal_length)


def generate_periodic_signal(signal_type, freq, amp, D, phase):
    T = 1 / freq

    if signal_type == "sine":
        return amp * np.sin(2 * np.pi * freq * t + phase) + D
    elif signal_type == "square":
        return amp * np.sign(np.sin(2 * np.pi * freq * t + phase)) + D
    elif signal_type == "sawtooth":
        return amp * (2 * ((t + phase / (2 * np.pi * freq)) / T % 1) - 1) + D
    elif signal_type == "triangle":
        return (4 * amp / T) * np.abs((t + phase / (2 * np.pi * freq) - T / 4) % T - T / 2) - amp + D
    else:
        raise ValueError("Unknown signal type")


def generate_poliharmonic_signal(n_components):
    signal = np.zeros(signal_length)
    for _ in range(n_components):
        freq = np.random.uniform(5, 10)
        phase = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(0.5, 1.5)
        D = 0
        signal += amp * np.sin(2 * np.pi * freq * t + phase) + D
    return signal / n_components


def generate_aperiodic_signal():
    freq = np.random.uniform(2, 10)
    damping_factor = np.exp(-t * np.random.uniform(1, 4))
    return np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi)) * damping_factor


def add_noise(signal, noise_type):
    if noise_type == "white":
        noise = np.random.uniform(-0.5, 0.5, signal_length)
    elif noise_type == "harmonic":
        freq = np.random.uniform(20, 30)
        noise = np.sin(2 * np.pi * freq * t) / 3
    elif noise_type == "impulse":
        noise = np.zeros(signal_length)
        num_peaks = np.random.randint(5, 11)
        positions = np.random.randint(0, signal_length, size=num_peaks)
        noise[positions] = np.random.uniform(-5, 5, size=num_peaks)
    elif noise_type == "mixed":
        types = ["white", "harmonic", "impulse"]
        noise = sum(add_noise(np.zeros_like(signal), nt) - np.zeros_like(signal) for nt in
                    np.random.choice(types, 2, replace=False)) / 1.5
    else:
        raise ValueError("Unknown noise type")

    return signal + noise


def generate_dataset(n_samples=30000):
    def rand_params():
        return {
            "freq": np.random.uniform(1, 10),
            "amp": np.random.uniform(0.5, 1.5),
            "D": 0,
            "phase": np.random.uniform(0, 2 * np.pi),
        }

    signal_generators = {
        "sine": lambda: generate_periodic_signal("sine", **rand_params()),
        "square": lambda: generate_periodic_signal("square", **rand_params()),
        "sawtooth": lambda: generate_periodic_signal("sawtooth", **rand_params()),
        "triangle": lambda: generate_periodic_signal("triangle", **rand_params()),
        "poliharmonic": lambda: generate_poliharmonic_signal(np.random.randint(2, 10)),
        "aperiodic": lambda: generate_aperiodic_signal(),}
    noise_types = ["white", "harmonic", "impulse", "mixed"]

    X_clean, X_noisy = [], []
    for _ in range(n_samples):
        sig_type = np.random.choice(list(signal_generators.keys()))
        noise_type = np.random.choice(noise_types)

        clean = signal_generators[sig_type]()
        noisy = add_noise(clean, noise_type)

        X_clean.append(clean)
        X_noisy.append(noisy)

    X_clean = np.array(X_clean, dtype=np.float32)[..., np.newaxis]
    X_noisy = np.array(X_noisy, dtype=np.float32)[..., np.newaxis]

    return X_noisy, X_clean


# Генерування даних
X_noisy, X_clean = generate_dataset()
X_train, X_val, X_test = X_noisy[:24000], X_noisy[24000:27000], X_noisy[27000:]
Y_train, Y_val, Y_test = X_clean[:24000], X_clean[24000:27000], X_clean[27000:]

print(len(X_val))

# Створення моделі
def build_conv_autoencoder(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Кодувальник
    x = layers.Conv1D(64, 10, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)

    # Декодувальник
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 5, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(64, 10, activation='relu', padding='same')(x)
    outputs = layers.Conv1D(1, 3, activation='linear', padding='same')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


autoencoder = build_conv_autoencoder(X_train.shape[1:])

# Навчання
history = autoencoder.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=30,
    batch_size=64,
)

# ========== 4. Графік втрат ==========
plt.figure(figsize=(8, 4))

epochs = range(1, len(history.history['loss']) + 1)  # Нумерація епох від 1

plt.plot(epochs, history.history['loss'], label='Похибка на навчальній вибірці')
plt.plot(epochs, history.history['val_loss'], label='Похибка на валідаційній вибірці')

plt.xlabel('Епоха')
plt.ylabel('MSE')
plt.legend()
plt.title('Динаміка похибок під час навчання')
plt.tight_layout()
plt.show()


#autoencoder = load_model("denoising_autoencoder.keras")


def compute_snr(clean, test):
    signal_power = np.mean(clean ** 2, axis=1)
    noise_power = np.mean((clean - test) ** 2, axis=1)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))  # додаємо eps, щоб уникнути ділення на 0
    return snr


# Отримуємо вихідні сигнали мережі
X_denoised = autoencoder.predict(X_test)

# Обчислюємо SNR до і після фільтрації
snr_before = compute_snr(Y_test.squeeze(), X_test.squeeze())
snr_after = compute_snr(Y_test.squeeze(), X_denoised.squeeze())

# Виводимо статистику
print("ССШ до фільтрації (дБ):")
print(f"  Середнє: {np.mean(snr_before):.2f}, Мін: {np.min(snr_before):.2f}, Макс: {np.max(snr_before):.2f}")
print("ССШ після фільтрації (дБ):")
print(f"  Середнє: {np.mean(snr_after):.2f}, Мін: {np.min(snr_after):.2f}, Макс: {np.max(snr_after):.2f}")

plt.hist(snr_before, bins=50, alpha=0.5, label='До фільтрації')
plt.hist(snr_after, bins=50, alpha=0.5, label='Після фільтрації')
plt.xlabel('SNR (дБ)')
plt.ylabel('Кількість сигналів')
plt.title('Розподіл SNR до і після фільтрації')
plt.legend()
plt.grid(True)
plt.show()


# Візуалізація випадкових прикладів із тестової вибірки
num_pictures = 20
num_examples = 3
for _ in range(num_pictures):
    indices = np.random.choice(len(X_test), size=num_examples, replace=False)

    fig, axes = plt.subplots(num_examples, 1, figsize=(12, 3 * num_examples))

    for i, idx in enumerate(indices):
        noisy = X_test[idx].squeeze()
        clean = Y_test[idx].squeeze()
        reconstructed = autoencoder.predict(X_test[idx][np.newaxis])[0].squeeze()

        ax = axes[i]

        ax.plot(t, noisy, label='Спотворений сигнал', color='orange', alpha=0.6)
        ax.plot(t, reconstructed, label='Відновлений сигнал', color='green')
        ax.plot(t, clean, label='Чистий сигнал', color='blue', linestyle='dashed')

        ax.set_title(f'Приклад {i + 1}', fontsize=13)
        ax.set_xlabel('Час', fontsize=12, labelpad=2)
        ax.set_ylabel('Значення сигналу', fontsize=12, labelpad=10)
        ax.legend(loc='upper right')
        ax.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.42)  # hspace — відстань між рядками (vertical space)
    plt.show()

# Збереження моделі
autoencoder.save("denoising_autoencoder.keras")
