import tensorflow as tf

def optimistic_restore(session, save_file, ignore_vars=None, verbose=False, ignore_incompatible_shapes=False):
    """This function tries to restore all variables in the save file.
    This function ignores variables that do not exist or have incompatible shape.
    Raises TypeError if the there is a type mismatch for compatible shapes.
    
    session: tf.Session
        The tf session
    save_file: str
        Path to the checkpoint without the .index, .meta or .data extensions.
    ignore_vars: list, tuple or set of str
        These variables will be ignored.
    verbose: bool
        If True prints which variables will be restored
    ignore_incompatible_shapes: bool
        If True ignores variables with incompatible shapes.
        If False raises a runtime error f shapes are incompatible.
    """
    def vprint(*args, **kwargs): 
        if verbose: print(*args, flush=True, **kwargs)
    # def dbg(*args, **kwargs): print(*args, flush=True, **kwargs)
    def dbg(*args, **kwargs): pass
    if ignore_vars is None:
        ignore_vars = []

    reader = tf.train.NewCheckpointReader(save_file)
    var_to_shape_map = reader.get_variable_to_shape_map()

    var_list = []
    for key in sorted(var_to_shape_map):
        if not 'Adam' in key: 
            var_list.append(key)
            print(key)

    return var_list 


optimistic_restore(tf.InteractiveSession(),'./ckpt/driving/one_at_a_time_training_flying/train/model.ckpt-149999',verbose=True)