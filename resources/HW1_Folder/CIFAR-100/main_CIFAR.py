import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs

import util_CIFAR
import model_CIFAR
import keras


train_images, train_labels, validation_images, validation_labels, validation_labels_save = util_CIFAR.load_data()

img_rows, img_cols, img_width = 32, 32, 3

input_shape = (img_rows, img_cols, img_width)

#model = model_CIFAR.create_model1(input_shape)
model = model_CIFAR.create_model2(input_shape)

model.summary()

optimizer = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
history = model.fit(train_images, train_labels, batch_size=32, epochs=1000,
                    validation_data=(validation_images, validation_labels_save), callbacks=[es], shuffle=True,
                    use_multiprocessing=(True))


validation_evaluation = model.evaluate(validation_images, validation_labels_save)

y_predict = model.predict(validation_images)

confusion_matrix = util_CIFAR.print_confusion_matrix(validation_labels, y_predict)

print(confusion_matrix)

util_CIFAR.print_accuracy_graph(history)

util_CIFAR.print_loss_graph(history)

lower_bound_interval, upper_bound_interval = util_CIFAR.confidence_interval(validation_evaluation, len(validation_labels))

print("The 95% Confidence interval for error hypothesis based on the normal distribution estimator: ",
      (lower_bound_interval, upper_bound_interval))

tf.saved_model.save(model, 'homework1/cifar100')





