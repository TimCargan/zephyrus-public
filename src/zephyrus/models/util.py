import keras_tuner as kt

def order_input(x_input):
    #'APM', 'CL', 'HRD0', 'TA2', 'WND',  #'convhead/APM', 'convhead/CL', 'convhead/HRD0', 'convhead/TA2', 'convhead/WND',
    model_input_order = ['doy', 'hour', 'lon', 'lat', 'clearsky']

    return dict([(x.replace('convhead/', ''), x_input[x]) for x in model_input_order])


## Pariton data for plant folds ##
# Create data path splits
plants = {'far dane%27s': 0, 'asfordby b': 0, 'asfordby a': 0, 'kirton': 0,         'nailstone':    5,
          'kelly green': 1, 'bake solar farm': 1, 'newnham': 1, 'caegarw': 1,       'rosedew':      5,
          'moss electrical': 2, 'caldecote': 2, 'clapham': 2, 'lains farm': 2,      'magazine':     5,
          'roberts wall solar farm': 3, 'crumlin': 3, 'moor': 3, 'soho farm': 3,    'box road':     5,
          'grange farm': 4, 'ashby': 4, 'somersal solar farm': 4, 'combermere farm': 4}

def split_data(fold, root_data_path, num_folds=5):
    folds = [root_data_path.replace("$FOLD$", str(i)) for i in range(num_folds)]
    folds = [[f.replace('plant=*', f'plant={x}') for (x, y) in plants.items() if y == i] for (i, f) in enumerate(folds)]

    test_split = folds[fold]
    l = folds[:fold] + folds[fold + 1:]
    # TODO: make this a bit better see https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    train_split = [item for sublist in l for item in sublist]

    return (test_split, train_split)


#########################################
#########        HP utils      #########
#########################################

def combination(hp:kt.HyperParameters, name, values, parent_name=None, parent_values=None):
    res = []
    for i,v in enumerate(values):
        if hp.Boolean(f"{name}_c_{i}", default=True, parent_name=parent_name, parent_values=parent_values):
            res.append(v)
    return res

def array_set(hp:kt.HyperParameters, name, arr):
    hp.Fixed(f"_{name}_len", len(arr))
    for i, el in enumerate(arr):
        hp.Fixed(f"_{name}_el_{i}", el)

def array_get(hp:kt.HyperParameters, name):
    num_els = hp.get(f"_{name}_len")
    arr = []
    for i in range(num_els):
        arr.append(hp.get(f"_{name}_el_{i}"))
    return arr