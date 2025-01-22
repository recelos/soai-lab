import numpy as np
import ssl
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential, layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    ssl._create_default_https_context = ssl._create_unverified_context

    (X_train, y_train), (X_test, y_test) = load_data()

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    img_height, img_width = X_train.shape[1], X_train.shape[2]
    num_classes = y_train.shape[1]

    model = Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 1)),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test), batch_size=32)
    _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    extractor = Sequential(model.layers[:-2])
    X_train_extracted = extractor.predict(X_train)
    X_train_extracted = X_train_extracted.reshape(X_train_extracted.shape[0], -1)

    X_test_extracted = extractor.predict(X_test)
    X_test_extracted = X_test_extracted.reshape(X_test_extracted.shape[0], -1)

    y_train_rf = np.argmax(y_train, axis=1)
    y_test_rf = np.argmax(y_test, axis=1)

    rf_classifier = RandomForestClassifier(n_estimators=10, random_state=0)
    rf_classifier.fit(X_train_extracted, y_train_rf)
    y_pred_rf = rf_classifier.predict(X_test_extracted)

    rf_accuracy = accuracy_score(y_test_rf, y_pred_rf)

    print(f'Dokładność Sequential: {test_accuracy * 100:.2f}%')
    print(f'Dokładność Random Forest: {rf_accuracy * 100:.2f}%')

if __name__=="__main__":
    main()
