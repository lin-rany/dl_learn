import tensorflow as tf

def TF_Range(size):
    #TFRange(10)=>TFRange:[0 1 2 3 4 5 6 7 8 9]
    a=tf.range(size)
    print(f"TFRange:{str(a)}")

def TF_Range_Shape(size):
    #TFRange(10)=>TFRange:[0 1 2 3 4 5 6 7 8 9]
    a=tf.range(size)
    print(f"TFRange.shape:{a.shape}")

def TF_Range_Reshape(size):
    #TFRange(10)=>TFRange:[0 1 2 3 4 5 6 7 8 9]
    a=tf.range(size)
    x=tf.reshape(a,(2,5))
    print(f"TFRange.reshape:{str(x)}")

def TF_zeros(size=16):
    print(f"TF_zero_shape:{str(tf.zeros(size))}")

def TF_ones(size=16):
    print(f"TF_ones:{str(tf.ones(size))}")

def TF_random(size=16):
    print(f"TF_random:{str(tf.random.normal(shape=[size,1]))}")

def TF_compare_range(size=8):
    a=tf.constant([1,2,3,4,5,6,7,8])
    b= tf.constant([2, 2, 3, 4, 5,10,11,12])
    # print(f"[TF_compare_range] a:{str(a)} b:{str(b)}")
    # newb=tf.zeros_like(a)+b
    # print(f"[TF_compare_range] newb:{str(newb)}")
    cmp=b==a
    print(f"[TF_compare_range] cmp: {str(cmp)}")

if __name__ == '__main__':
    TF_Range(10)
    TF_Range_Shape(10)
    TF_Range_Reshape(10)
    TF_zeros()
    TF_ones()
    TF_random()
    TF_compare_range()