import matplotlib.pyplot as plt

target_name = {0: 'Bread', 1: 'Dairy products', 2: 'Dessert', 3: 'Egg', 4: 'Fried food', 5: 'Meat', 6: 'Noodles/Pasta',
               7: 'Rice', 8: 'Seafood', 9: 'Soup', 10: 'Vegetable/Fruit'}


def plot_hidden_feature(units, dim, title):
    imgs = units.shape[3]
    fig = plt.figure()
    fig.set_size_inches(6, 6)
    n_columns = dim
    n_rows = imgs // n_columns + 1
    for i in range(imgs):
        plt.subplot(n_rows, n_columns, i+1)
        plt.imshow(units[0,:,:,i], aspect='auto')
        plt.tick_params(labelbottom=False, labelleft=False)
    fig.suptitle(title)
    plt.show()