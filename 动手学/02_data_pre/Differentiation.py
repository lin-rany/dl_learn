
import tensorflow as tf


def MyPrint(name, tensor):
    print(f"{name}:{str(tensor.numpy())}")

def eg1():
    x = tf.range(4, dtype=tf.float32)
    print(f"x:{str(x)}")
    x = tf.Variable(x)
    with tf.GradientTape() as t:
        y = 2 * tf.tensordot(x, x, axes=1)
        print(f"y:{str(y)}")
        x_grad = t.gradient(y, x)
        print(f"x_grad:{str(x_grad)}")
    with tf.GradientTape() as t:
        y = tf.reduce_sum(x)
    new_x_grad=t.gradient(y, x)
    print(f"new_x_grad:{str(new_x_grad)}")




if __name__ == '__main__':
    eg1()