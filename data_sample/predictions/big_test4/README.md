predictions with :  (['conv',(3, 5, 16), tf.nn.relu], 1), #layer 1: convolution layer with a filter of 1*25 and a depth of 32 using relu as an activation function
                                 (['conv',(3, 5, 16), tf.nn.relu], 1),
                                 (['pooling',(1, 2), None], 1),
                                 (['conv',(3, 5, 32), tf.nn.relu], 1),
                                 (['conv',(3, 5, 32), tf.nn.relu], 1),
                                 (['pooling',(1,2), None], 1),
                                 (['conv',(3, 5, 64), tf.nn.relu], 1),
                                 (['conv',(13, 5, 64), tf.nn.relu], 1),
                                # (['conv',(13,10, 3), tf.nn.relu], 1),
                                 (['conv',(13,10, 3), None], 1)
