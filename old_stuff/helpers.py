
def ease_in_quad( current_time, start_value, change_value, duration ):
    """
    current_time: float or Tensor
        The current time

    start_value: float or Tensor
        The start value

    change_value: float or Tensor
        The value change of the duration. The final value is start_value + change_value

    duration: float or Tensor
        The duration

    Returns the value for the current time
    """
    with tf.name_scope("ease_in_quad"):
        t = tf.clip_by_value(current_time/duration, 0, 1)
        return tf.to_float(change_value*t*t + start_value)


def ease_out_quad( current_time, start_value, change_value, duration ):
    """
    current_time: float or Tensor
        The current time

    start_value: float or Tensor
        The start value

    change_value: float or Tensor
        The value change of the duration. The final value is start_value + change_value

    duration: float or Tensor
        The duration

    Returns the value for the current time
    """
    with tf.name_scope("ease_out_quad"):
        t = tf.clip_by_value(current_time/duration, 0, 1)
        return tf.to_float(-change_value*t*(t-2) + start_value)