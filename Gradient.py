import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Malgun Gothic'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(42)
tf.random.set_seed(42)

# 데이터 생성
def get_data(samples=1000):
    X = np.random.rand(samples, 10)
    y = (np.sum(X, axis=1) > 5).astype(int)
    return X, y

# 모델 생성 함수
def create_model(depth, activation='relu', initializer='he_normal', use_clip=False, use_norm=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(10,)))
    for _ in range(depth):
        model.add(tf.keras.layers.Dense(64, activation=None, kernel_initializer=initializer))
        if use_norm:
            model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.Activation(activation))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

# 모델과 X_sample을 직접 받아 히스토그램 그리기
def plot_gradients_from_model(model, X_sample, title='Gradient Histogram'):
    fig, ax = plt.subplots(figsize=(8, 5))
    with tf.GradientTape() as tape:
        preds = model(X_sample, training=True)
        loss = tf.keras.losses.binary_crossentropy(tf.ones_like(preds), preds)
    grads = tape.gradient(loss, model.trainable_variables)

    dense_grads = [g.numpy().flatten() for g in grads if len(g.shape) > 1]
    for i, g in enumerate(dense_grads):
        label = f'Layer {i+1}' if i % 10 == 0 else None
        ax.hist(g, bins=50, alpha=0.5, label=label)

    all_grads = np.concatenate(dense_grads)
    mean_grad = np.mean(all_grads)
    ax.axvline(mean_grad, color='green', linestyle='--', label=f'Mean={mean_grad:.1e}')
    ax.set_title(title)
    ax.set_xlabel('Gradient Value')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize='x-small')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# 여러 활성화 함수에 대한 비교용 함수
def plot_gradients_by_activation(depth=20, activations=['sigmoid', 'tanh', 'relu']):
    fig, axs = plt.subplots(1, len(activations), figsize=(18, 6), sharey=True)
    X_sample, _ = get_data(samples=1)

    for ax, act in zip(axs, activations):
        model = create_model(depth, act)
        with tf.GradientTape() as tape:
            preds = model(X_sample, training=True)
            loss = tf.keras.losses.binary_crossentropy(tf.ones_like(preds), preds)
        grads = tape.gradient(loss, model.trainable_variables)
        dense_grads = [g.numpy().flatten() for g in grads if len(g.shape) > 1]

        for i, g in enumerate(dense_grads):
            label = f'Layer {i+1}' if i % 10 == 0 else None
            ax.hist(g, bins=50, alpha=0.5, label=label)

        ax.set_title(f'Activation: {act}')
        ax.set_xlabel('Gradient Value')
        ax.grid(True)

        all_grads = np.concatenate(dense_grads)
        mean_grad = np.mean(all_grads)
        ax.axvline(mean_grad, color='green', linestyle='--', label=f'Mean={mean_grad:.1e}')
        ax.legend(fontsize='x-small', loc='upper right')

    axs[0].set_ylabel('Frequency')
    fig.suptitle(f'[기울기 분포 비교] sigmoid, tanh, relu (Depth={depth})')
    plt.tight_layout()
    plt.show()

# 훈련 함수
def train_model(depth, activation, initializer, lr, use_clip=False, use_norm=False, epochs=20):
    X, y = get_data()
    model = create_model(depth, activation, initializer, use_clip, use_norm)

    if use_clip:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    history = model.fit(X, y, epochs=epochs, verbose=0)
    return model, history.history['loss']

# 실험 실행 함수
def run_all_experiments():
    X_sample, _ = get_data(samples=1)

    # 실험 1: 깊이+활성화 함수 → 기울기 분포 
    depths = [5, 20, 50]
    activations = ['sigmoid', 'tanh', 'relu']
    for depth in depths:
        plot_gradients_by_activation(depth=depth, activations=activations)

    # 실험 2: 학습률 변화에 따른 손실
    lrs = [0.01, 0.1, 1.0, 10.0]
    for lr in lrs:
        _, loss = train_model(20, 'relu', 'he_normal', lr)
        plt.plot(loss, label=f'LR={lr}')
    plt.title('[실험2] 학습률별 손실 변화')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 실험 3: 초기화 방법
    initializers = ['random_normal', 'glorot_uniform', 'he_normal']
    for init in initializers:
        _, loss = train_model(20, 'relu', init, 0.1)
        plt.plot(loss, label=f'Init={init}')
    plt.title('[실험3] 초기화 방법에 따른 손실 변화')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 실험 4: Gradient Clipping & Layer Normalization
    configs = [
        ('Baseline', False, False),
        ('Gradient Clipping', True, False),
        ('Layer Norm', False, True),
        ('Both', True, True)
    ]
    for name, clip, norm in configs:
        _, loss = train_model(20, 'relu', 'he_normal', 0.1, use_clip=clip, use_norm=norm)
        plt.plot(loss, label=name)
    plt.title('[실험4] 기울기 폭주/소실 방지 기법 비교')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 실험 5: 각 층 기울기 히스토그램 (사용자 모델 직접 지정)
    deep_model = create_model(50, activation='sigmoid')
    plot_gradients_from_model(deep_model, X_sample, '50층 모델 - sigmoid (기울기 소실 확인)')

    expl_model = create_model(20, activation='relu', initializer='random_normal')
    plot_gradients_from_model(expl_model, X_sample, '20층 모델 - relu + random_normal (기울기 폭주 확인)')

    # 실험 6: 여러 활성화 함수 비교
    plot_gradients_by_activation(depth=20, activations=['sigmoid', 'tanh', 'relu'])

# 전체 실험 실행
run_all_experiments()
